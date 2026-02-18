# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1.

Training (backward): use context_attention() or ContextAttentionFunc for autograd,
or context_attention_fwd(..., return_lse=True) + context_attention_bwd() manually.
Inference: use context_attention_fwd(..., return_lse=False) (default), unchanged.
Decode and extend backward are not implemented (not needed for True On-Policy).
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda or _is_hip:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )


@triton.jit
def _fwd_kernel_with_lse(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    LSE,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_s,
    stride_lse_h,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    """Forward kernel that also writes LSE for backward. Same as _fwd_kernel plus LSE output."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )

    # LSE = m_i + log(l_i), safe for l_i=0
    lse_i = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    off_lse = (cur_batch_in_all_start_index + offs_m) * stride_lse_s + cur_head * stride_lse_h
    tl.store(LSE + off_lse, lse_i, mask=offs_m < cur_batch_seq_len)


def context_attention_fwd(
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    max_input_len,
    is_causal=True,
    return_lse=False,
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    When return_lse=True, returns (o, lse) with lse of shape [total_tokens, num_heads], float32.
    """
    if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    if return_lse:
        total_tokens = q.shape[0]
        lse = torch.empty(
            (total_tokens, head), dtype=torch.float32, device=q.device
        )
        _fwd_kernel_with_lse[grid](
            q,
            k,
            v,
            sm_scale,
            b_start_loc,
            b_seq_len,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            o.stride(0),
            o.stride(1),
            lse.stride(0),
            lse.stride(1),
            kv_group_num=kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=triton.next_power_of_2(Lk),
            BLOCK_N=BLOCK,
            IS_CAUSAL=is_causal,
            num_warps=num_warps,
            num_stages=1,
            Lk=Lk,
        )
        return o, lse

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


# ------------------------------------------------------------------------------
# Backward (for training): Flash-style, deterministic (key-block parallel)
# ------------------------------------------------------------------------------

@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    dO,
    LSE,
    Delta,
    dQ_buffer,
    dK,
    dV,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_dobs,
    stride_doh,
    stride_lse_s,
    stride_lse_h,
    stride_delta_s,
    stride_delta_h,
    stride_dqb_n,
    stride_dqb_s,
    stride_dqb_h,
    stride_dkbs,
    stride_dkh,
    stride_dvbs,
    stride_dvh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    num_blocks_m: tl.constexpr,
):
    """
    Deterministic backward: each program (batch, head, start_n) owns one key block.
    Loops over all query blocks (start_m), accumulates dK/dV in registers and stores once.
    dQ contributions are written to dQ_buffer[start_n, ...]; caller sums over start_n.
    No atomic operations -> bitwise reproducible.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    # This key block covers indices [start_n*BLOCK_N, (start_n+1)*BLOCK_N)
    if start_n * BLOCK_N >= cur_batch_seq_len:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)

    # Load K, V block once for this key block
    k_ptrs = (
        K
        + (cur_batch_in_all_start_index + start_n * BLOCK_N + offs_n[:, None]) * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + (cur_batch_in_all_start_index + start_n * BLOCK_N + offs_n[:, None]) * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :]
    )
    mask_n = (start_n * BLOCK_N + offs_n) < cur_batch_seq_len
    mask_d = offs_d < Lk
    kv_mask = mask_n[:, None] & mask_d[None, :]
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    # k [BLOCK_N, D], v [BLOCK_N, D]; we need k^T for qk = q @ k^T
    kT = tl.trans(k)

    # Accumulate dK, dV for this key block (single writer -> deterministic)
    dk_block = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Causal: query block start_m sees this key block iff (start_m+1)*BLOCK_M > start_n*BLOCK_N
    lo_m = (start_n * BLOCK_N + BLOCK_M - 1) // BLOCK_M

    for start_m in range(0, num_blocks_m):
        start_m = tl.multiple_of(start_m, 1)
        # Only process if this query block is valid and (when causal) can see this key block
        do_block = start_m * BLOCK_M < cur_batch_seq_len
        if IS_CAUSAL:
            do_block = do_block & (start_m >= lo_m)
        if do_block:
            offs_m_cur = start_m * BLOCK_M + offs_m
            mask_m = offs_m_cur < cur_batch_seq_len

            off_q = (
                (cur_batch_in_all_start_index + offs_m_cur[:, None]) * stride_qbs
                + cur_head * stride_qh
                + offs_d[None, :]
            )
            off_do = (
                (cur_batch_in_all_start_index + offs_m_cur[:, None]) * stride_dobs
                + cur_head * stride_doh
                + offs_d[None, :]
            )
            off_lse = (cur_batch_in_all_start_index + offs_m_cur) * stride_lse_s + cur_head * stride_lse_h
            off_delta = (cur_batch_in_all_start_index + offs_m_cur) * stride_delta_s + cur_head * stride_delta_h

            q = tl.load(Q + off_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            do = tl.load(dO + off_do, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            lse = tl.load(LSE + off_lse, mask=mask_m, other=-float("inf"))
            delta_i = tl.load(Delta + off_delta, mask=mask_m, other=0.0)

            # S = Q @ K^T * scale
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, kT)
            qk *= sm_scale

            key_pos = start_n * BLOCK_N + offs_n[None, :]
            if IS_CAUSAL:
                qk += tl.where(
                    (key_pos < cur_batch_seq_len) & (offs_m_cur[:, None] >= key_pos),
                    0,
                    float("-inf"),
                )
            else:
                qk += tl.where(key_pos < cur_batch_seq_len, 0, float("-inf"))

            # P = exp(S - LSE)
            p = tl.exp(qk - lse[:, None])
            p = tl.where(qk == float("-inf"), 0.0, p)
            p = tl.where(mask_m[:, None], p, 0.0)

            # dP = dO @ V^T, dS = P * (dP - Delta)
            dp = tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = p * (dp - delta_i[:, None])

            # Accumulate dK, dV for this key block
            dk_block += tl.dot(tl.trans(ds), q, allow_tf32=False) * sm_scale
            dv_block += tl.dot(tl.trans(p), do, allow_tf32=False)

            # dQ contribution from this (start_n, start_m): store to dQ_buffer
            # ds [BLOCK_M, BLOCK_N], k [BLOCK_N, BLOCK_DMODEL] -> dq_block [BLOCK_M, BLOCK_DMODEL]
            dq_block = tl.dot(ds, k, allow_tf32=False) * sm_scale
            dqb_base = (
                dQ_buffer
                + start_n * stride_dqb_n
                + (cur_batch_in_all_start_index + start_m * BLOCK_M) * stride_dqb_s
                + cur_head * stride_dqb_h
            )
            dqb_ptrs = dqb_base + offs_m_cur[:, None] * stride_dqb_s + offs_d[None, :]
            tl.store(dqb_ptrs, dq_block.to(dQ_buffer.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    # Store dK, dV for this key block (vectorized, no atomic)
    dk_ptrs = (
        dK
        + (cur_batch_in_all_start_index + start_n * BLOCK_N + offs_n[:, None]) * stride_dkbs
        + cur_kv_head * stride_dkh
        + offs_d[None, :]
    )
    dv_ptrs = (
        dV
        + (cur_batch_in_all_start_index + start_n * BLOCK_N + offs_n[:, None]) * stride_dvbs
        + cur_kv_head * stride_dvh
        + offs_d[None, :]
    )
    mask_nd = (start_n * BLOCK_N + offs_n < cur_batch_seq_len)[:, None] & (offs_d < Lk)[None, :]
    tl.store(dk_ptrs, dk_block.to(dK.dtype.element_ty), mask=mask_nd)
    tl.store(dv_ptrs, dv_block.to(dV.dtype.element_ty), mask=mask_nd)


def context_attention_bwd(
    dO,
    O,
    LSE,
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    max_input_len,
    is_causal=True,
):
    """
    Backward of context_attention_fwd.
    dO: [total_tokens, head, head_dim], gradient of output
    O, LSE: saved from forward (LSE from return_lse=True)
    q, k, v: [total_tokens, head, head_dim] or varlen layout
    Returns: dq, dk, dv
    """
    if not (_is_cuda or _is_hip):
        raise RuntimeError("context_attention_bwd requires CUDA or HIP")
    if CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Delta = (O * dO).sum(dim=-1) = rowsum(P * dP) over all keys; same layout as LSE [total_tokens, head]
    delta = (O.float() * dO.float()).sum(dim=-1)
    if not delta.is_contiguous():
        delta = delta.contiguous()

    num_blocks_m = triton.cdiv(max_input_len, BLOCK)
    num_blocks_n = triton.cdiv(max_input_len, BLOCK)
    total_tokens = q.shape[0]
    # dQ buffer: (num_blocks_n, total_tokens, head, head_dim) for deterministic sum over blocks
    dq_buffer = torch.zeros(
        (num_blocks_n, total_tokens, head, Lk),
        dtype=torch.float32,
        device=q.device,
    )
    if not dq_buffer.is_contiguous():
        dq_buffer = dq_buffer.contiguous()

    grid = (batch, head, num_blocks_n)
    num_warps = 4 if Lk <= 64 else 8

    _bwd_kernel[grid](
        q,
        k,
        v,
        dO,
        LSE,
        delta,
        dq_buffer,
        dk,
        dv,
        sm_scale,
        b_start_loc,
        b_seq_len,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        dO.stride(0),
        dO.stride(1),
        LSE.stride(0),
        LSE.stride(1),
        delta.stride(0),
        delta.stride(1),
        dq_buffer.stride(0),
        dq_buffer.stride(1),
        dq_buffer.stride(2),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_blocks_m=num_blocks_m,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
    dq = dq_buffer.sum(dim=0).to(q.dtype)
    return dq, dk, dv


class ContextAttentionFunc(torch.autograd.Function):
    """
    Autograd wrapper for context attention (prefill). Use this for training
    so that backward uses the same Triton prefill backward (LSE-based).
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal,
    ):
        o = torch.empty_like(q)
        o, lse = context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_input_len,
            is_causal=is_causal,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.b_start_loc = b_start_loc
        ctx.b_seq_len = b_seq_len
        ctx.max_input_len = max_input_len
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = context_attention_bwd(
            dO, o, lse, q, k, v,
            ctx.b_start_loc,
            ctx.b_seq_len,
            ctx.max_input_len,
            is_causal=ctx.is_causal,
        )
        return dq, dk, dv, None, None, None, None


def context_attention(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal=True):
    """
    Prefill context attention with autograd support (for training).
    Use this when you need gradients. For inference-only, use context_attention_fwd
    with return_lse=False.
    q, k, v: [total_tokens, head, head_dim]
    b_start_loc: [batch], int32
    b_seq_len: [batch], int32
    max_input_len: int, max sequence length in the batch
    """
    return ContextAttentionFunc.apply(
        q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
    )
