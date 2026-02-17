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
# Backward (for training): Flash-style, recompute P from LSE
# ------------------------------------------------------------------------------

@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    dO,
    LSE,
    dQ,
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
    stride_dqbs,
    stride_dqh,
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
):
    """
    Backward kernel: each program (batch, head, block_m) computes dQ for one Q block
    and accumulates dK, dV with atomic add.
    """
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
    off_do = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_dobs
        + cur_head * stride_doh
        + offs_d[None, :]
    )
    off_lse = (cur_batch_in_all_start_index + offs_m) * stride_lse_s + cur_head * stride_lse_h

    mask_d = offs_d < Lk
    mask_m = offs_m < cur_batch_seq_len

    q = tl.load(Q + off_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(dO + off_do, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    lse = tl.load(LSE + off_lse, mask=mask_m, other=-float("inf"))

    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    v_ptrs = V + offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

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
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        # S = Q @ K^T * scale
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

        # P = exp(S - LSE), causal mask already in S (-inf -> 0)
        p = tl.exp(qk - lse[:, None])
        p = tl.where(qk == float("-inf"), 0.0, p)

        # dP = dO @ V^T  (do [BLOCK_M, D] @ v^T [D, BLOCK_N] -> [BLOCK_M, BLOCK_N])
        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        # dS = P * (dP - rowsum(P * dP))
        p_dp = p * dp
        rowsum_p_dp = tl.sum(p_dp, 1)
        ds = p * (dp - rowsum_p_dp[:, None])

        # dQ += dS @ K * scale  (ds [BLOCK_M, BLOCK_N], k loaded as [D, N] -> use k^T)
        dq_acc += tl.dot(ds, tl.trans(k), allow_tf32=False) * sm_scale

        # dK += dS^T @ Q * scale  [BLOCK_N, D]
        dk_block = tl.dot(
            tl.trans(ds), q, allow_tf32=False
        ) * sm_scale
        # dV += P^T @ dO  [BLOCK_N, D]
        dv_block = tl.dot(tl.trans(p), do, allow_tf32=False)

        # Atomic add dK, dV (element-wise), only for valid key positions.
        # Use tl.range (runtime loop) instead of tl.static_range to avoid 128*64 compile-time
        # unrolling which causes very slow compilation. Extract scalar via mask+sum.
        dk_block = dk_block.to(dK.dtype.element_ty)
        dv_block = dv_block.to(dV.dtype.element_ty)
        for nn in tl.range(BLOCK_N):
            n_idx = start_n + nn
            if n_idx < cur_batch_seq_len:
                mask_n = (tl.arange(0, BLOCK_N) == nn)
                for dd in tl.range(BLOCK_DMODEL):
                    if dd < Lk:
                        mask_d = (tl.arange(0, BLOCK_DMODEL) == dd)
                        dk_val = tl.sum(dk_block * mask_n[:, None] * mask_d[None, :])
                        dv_val = tl.sum(dv_block * mask_n[:, None] * mask_d[None, :])
                        dk_ptr = (
                            dK
                            + (cur_batch_in_all_start_index + n_idx) * stride_dkbs
                            + cur_kv_head * stride_dkh
                            + dd
                        )
                        dv_ptr = (
                            dV
                            + (cur_batch_in_all_start_index + n_idx) * stride_dvbs
                            + cur_kv_head * stride_dvh
                            + dd
                        )
                        tl.atomic_add(dk_ptr, dk_val)
                        tl.atomic_add(dv_ptr, dv_val)

    # Store dQ
    off_dq = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_dqbs
        + cur_head * stride_dqh
        + offs_d[None, :]
    )
    dq_acc = dq_acc.to(dQ.dtype.element_ty)
    tl.store(dQ + off_dq, dq_acc, mask=mask_m[:, None] & mask_d[None, :])


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

    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _bwd_kernel[grid](
        q,
        k,
        v,
        dO,
        LSE,
        dq,
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
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
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
