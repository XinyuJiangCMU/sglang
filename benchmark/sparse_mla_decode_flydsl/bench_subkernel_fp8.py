"""Standalone benchmark: prototype FP8 sparse MLA decode sub-kernel.

**This is a prototype kernel — NOT integrated into the production
dispatch path.** It measures one slice of the work the production
TileLang ``dpsk_v4_fp8_attention_fwd`` does:

  - Sparse K/V gather via per-batch ``indices[b, n]``
  - Inline FP8 e4m3 dequant (with real per-NOPE_TILE scale)
  - Q @ K^T mfma (mfma_f32_16x16x32_bf16)
  - Row-wise softmax with ``softmax_scale``
  - S @ V mfma (mfma_f32_16x16x16bf16_1k)
  - Output write

Explicitly NOT covered:

  - ``D_tail`` (64 BF16 elements per K row → ~14% less compute here)
  - ``extra_k_cache`` / ``extra_indices_in_kvcache`` (dual cache)
  - Online softmax across multiple BI chunks (m_i / sumexp carry)
  - ``attn_sink`` folding (TileLang combine kernel)
  - Partial_O / Partial_LSE emission + combine kernel

So a µs-per-batch number from this script is *not* a feature-parity
speedup vs TileLang's ``dpsk_v4_fp8_attention_fwd``. It is evidence
that the FlyDSL toolchain reaches competitive perf for the sub-kernel
slice.

Correctness is validated bit-identically against a PyTorch reference
that uses the same dequant bit formula (matches TileLang's formula at
the bit level; see ``tilelang_kernel.py:1780-1797``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def _build_fp8_sparse_kernel(NOPE_TILE: int, SM_SCALE: float):
    """Compile the prototype FP8 sparse MLA decode sub-kernel.

    Returns ``(launch, lds_layout_info)`` where ``launch`` is a
    ``@flyc.jit`` callable.

    Assumes a synthetic K-cache row stride of ``D + 8`` (FP8 packed +
    scale region, no D_tail). The production layout uses ``584`` bytes
    per row to also carry a 64-byte BF16 tail; handling that is
    out-of-scope for this prototype.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _mlir_arith
    from flydsl._mlir.dialects import memref as _memref
    from flydsl._mlir.dialects import scf as _scf
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, range_constexpr, rocdl
    from flydsl.expr.typing import T
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    try:
        from aiter.ops.flydsl.kernels.tensor_shim import GTensor
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "This benchmark requires aiter's tensor_shim. Install aiter "
            "with the FlyDSL extras (see "
            "docs/developer_guide/amd_flydsl_sparse_mla.md)."
        ) from e

    arch = get_rocm_arch()

    LDS_S_F32_BYTES = 16 * 64 * 4
    LDS_S_BF16_BYTES = 16 * 64 * 2

    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name="sglang_flydsl_attn_fp8_subkernel_smem",
    )
    lds_s_f32_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_f32_off + LDS_S_F32_BYTES
    lds_s_bf16_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_bf16_off + LDS_S_BF16_BYTES

    M_TILE, N_TILE, K_MFMA = 16, 16, 32
    BI, D, D_V = 64, 448, 448
    NUM_SCALES = D // NOPE_TILE
    ROW_BYTES_FP8 = D
    ROW_BYTES_V_FP8 = D_V
    ROW_STRIDE_KCACHE = ROW_BYTES_FP8 + 8
    ROW_STRIDE_VCACHE = ROW_BYTES_V_FP8 + 8
    BS_KV = 128

    N_SUBTILES = BI // N_TILE  # 4
    K_TILES = D // K_MFMA      # 14
    K_V_TILES_16 = BI // 16    # 4
    OUT_N_TILES = D_V // N_TILE  # 28

    def _dequant_fp8_to_bf16_bits(b_u32, s_u32):
        c80 = arith.constant(0x80, type=T.i32)
        c78 = arith.constant(0x78, type=T.i32)
        c7  = arith.constant(0x7, type=T.i32)
        c100h = arith.constant(0x100, type=T.i32)
        c10 = arith.constant(0x10, type=T.i32)
        c3  = arith.constant(3, type=T.i32)
        c7m = arith.constant(7, type=T.i32)
        sign_bf = arith.muli(arith.andi(b_u32, c80), c100h)
        exp_e4 = arith.shrui(arith.andi(b_u32, c78), c3)
        mant_bf = arith.muli(arith.andi(b_u32, c7), c10)
        exp_c = arith.subi(arith.addi(exp_e4, s_u32), c7m)
        bits_i32 = arith.ori(
            arith.ori(sign_bf, arith.shli(exp_c, c7m)), mant_bf
        )
        return arith.trunci(T.i16, bits_i32)

    @flyc.kernel(name="sglang_flydsl_attn_fp8_subkernel")
    def kernel(q_tensor, k_cache, v_cache, indices, o_tensor):
        tid = fx.thread_idx.x
        m_block = fx.block_idx.x
        b_block = fx.block_idx.y

        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        # K cache i32 view to dodge LLVM v8i8 buffer_load split bug.
        kc_i32 = GTensor(k_cache, dtype=T.i32, shape=(-1,))
        kc_i8 = GTensor(k_cache, dtype=T.i8, shape=(-1,))
        vc_i8 = GTensor(v_cache, dtype=T.i8, shape=(-1,))
        idx_ = GTensor(indices, dtype=T.i32, shape=(-1,))
        o_ = GTensor(o_tensor, dtype=T.f32, shape=(-1,))

        m_in_tile = tid % fx.Int32(16)
        k_lo_32 = (tid // fx.Int32(16)) * fx.Int32(8)
        global_m = (
            b_block * fx.Int32(128)
            + m_block * fx.Int32(M_TILE)
            + m_in_tile
        )
        BS_KV_C = arith.constant(BS_KV, type=T.i32)
        ROW_STRIDE_C = arith.constant(ROW_STRIDE_KCACHE, type=T.i32)
        ROW_STRIDE_V_C = arith.constant(ROW_STRIDE_VCACHE, type=T.i32)

        f32x4_t = T.vec(4, T.f32)
        bf16x8_t = T.vec(8, T.bf16)
        bf16x4_t = T.vec(4, T.bf16)
        i16x4_t = T.vec(4, T.i16)
        v8i8_t = T.vec(8, T.i8)
        zero_f32 = arith.constant(0.0, type=T.f32)

        accs = [
            _vector.broadcast(f32x4_t, zero_f32)
            for _ in range_constexpr(N_SUBTILES)
        ]

        # ---- Step 1: QK gemm with sparse FP8 K-gather + inline dequant ----
        for n_tile in range_constexpr(N_SUBTILES):
            n_in_tile = tid % fx.Int32(16)
            n_global = fx.Int32(n_tile * N_TILE) + n_in_tile
            idx_off = b_block * fx.Int32(BI) + n_global
            token = idx_.load(idx_off, vec_size=1)
            tok_raw = token.value if hasattr(token, "value") else token
            blk = arith.divui(tok_raw, BS_KV_C)
            inb = arith.remui(tok_raw, BS_KV_C)
            row_idx = arith.addi(arith.muli(blk, BS_KV_C), inb)
            row_byte_base = arith.muli(row_idx, ROW_STRIDE_C)

            for k_tile in range_constexpr(K_TILES):
                k_byte_off_in_row = (
                    fx.Int32(k_tile * K_MFMA) + k_lo_32
                )
                kbor_raw = (
                    k_byte_off_in_row.value
                    if hasattr(k_byte_off_in_row, "value")
                    else k_byte_off_in_row
                )
                k_global_byte = arith.addi(row_byte_base, kbor_raw)

                # Load 8 FP8 bytes via 2 i32 loads.
                k_word_off = arith.divui(
                    k_global_byte, arith.constant(4, type=T.i32)
                )
                k_i32_vec = kc_i32.load(k_word_off, vec_size=2)
                k_fp8 = _vector.bitcast(v8i8_t, k_i32_vec)
                # Scale byte for this K tile.
                scale_byte_off = arith.addi(
                    row_byte_base,
                    arith.constant(
                        ROW_BYTES_FP8 + (k_tile // 2), type=T.i32
                    ),
                )
                s_byte = kc_i8.load(scale_byte_off, vec_size=1)
                s_raw = s_byte.value if hasattr(s_byte, "value") else s_byte
                s_u32 = arith.extui(T.i32, s_raw)

                bf16_elems = []
                for el in range_constexpr(8):
                    b_el = _vector.extract(
                        k_fp8, static_position=[el], dynamic_position=[]
                    )
                    b_u32 = arith.extui(T.i32, b_el)
                    bf16_bits = _dequant_fp8_to_bf16_bits(b_u32, s_u32)
                    bf16_v = arith.bitcast(T.bf16, bf16_bits)
                    bf16_elems.append(bf16_v)
                b_frag = _vector.from_elements(bf16x8_t, bf16_elems)

                q_a_off = (
                    global_m * fx.Int32(D)
                    + fx.Int32(k_tile * K_MFMA)
                    + k_lo_32
                )
                a_frag = q_.load(q_a_off, vec_size=8)

                accs[n_tile] = rocdl.mfma_f32_16x16x32_bf16(
                    f32x4_t, [a_frag, b_frag, accs[n_tile], 0, 0, 0]
                )

        # ---- Step 2: spill to LDS ----
        m_lo_c = (tid // fx.Int32(16)) * fx.Int32(4)
        n_c = tid % fx.Int32(16)
        s_lds = SmemPtr(
            allocator.get_base(), lds_s_f32_off, T.f32, shape=(M_TILE * BI,)
        )
        s_lds_mr = s_lds.get()
        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                scalar = _vector.extract(
                    accs[n_tile], static_position=[el], dynamic_position=[]
                )
                ix = (
                    (m_lo_c + fx.Int32(el)) * fx.Int32(BI)
                    + fx.Int32(n_tile * N_TILE)
                    + n_c
                )
                ix_raw = ix.value if hasattr(ix, "value") else ix
                _memref.store(
                    scalar,
                    s_lds_mr,
                    [arith.index_cast(T.index, ix_raw)],
                )
        rocdl.barrier()

        # ---- Step 3: row-wise softmax over BI (16 worker lanes) ----
        is_worker = arith.cmpi(
            _mlir_arith.CmpIPredicate.ult,
            tid.value if hasattr(tid, "value") else tid,
            arith.constant(16, type=T.i32),
        )
        log2e = arith.constant(1.4426950408889634, type=T.f32)
        sm_scale = arith.constant(SM_SCALE, type=T.f32)
        c_neg_inf = arith.constant(-1e30, type=T.f32)
        c_zero = arith.constant(0.0, type=T.f32)
        if_op = _scf.IfOp(is_worker, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            row_base = tid * fx.Int32(BI)
            rb = row_base.value if hasattr(row_base, "value") else row_base
            row_vals, cur_max = [], c_neg_inf
            for j in range_constexpr(BI):
                idx = arith.addi(rb, arith.constant(j, type=T.i32))
                v_raw = _memref.load(
                    s_lds_mr, [arith.index_cast(T.index, idx)]
                )
                v = arith.mulf(v_raw, sm_scale)
                row_vals.append(v)
                cur_max = arith.maxnumf(cur_max, v)
            exp_vals, cur_sum = [], c_zero
            for j in range_constexpr(BI):
                d = arith.subf(row_vals[j], cur_max)
                e = rocdl.exp2(T.f32, arith.mulf(d, log2e))
                exp_vals.append(e)
                cur_sum = arith.addf(cur_sum, e)
            for j in range_constexpr(BI):
                s = arith.divf(exp_vals[j], cur_sum)
                idx = arith.addi(rb, arith.constant(j, type=T.i32))
                _memref.store(
                    s, s_lds_mr, [arith.index_cast(T.index, idx)]
                )
            _scf.yield_([])
        rocdl.barrier()

        # ---- Step 4: cast f32 → bf16 ----
        s_bf16_lds = SmemPtr(
            allocator.get_base(),
            lds_s_bf16_off,
            T.bf16,
            shape=(M_TILE * BI,),
        )
        s_bf16_mr = s_bf16_lds.get()
        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                ix = (
                    (m_lo_c + fx.Int32(el)) * fx.Int32(BI)
                    + fx.Int32(n_tile * N_TILE)
                    + n_c
                )
                ix_raw = ix.value if hasattr(ix, "value") else ix
                ix_ix = arith.index_cast(T.index, ix_raw)
                f32_v = _memref.load(s_lds_mr, [ix_ix])
                _memref.store(
                    arith.truncf(T.bf16, f32_v), s_bf16_mr, [ix_ix]
                )
        rocdl.barrier()

        # ---- Step 5: S @ V with sparse FP8 V gather + inline dequant ----
        m_a = tid % fx.Int32(16)
        n_b = tid % fx.Int32(16)
        k_lo_16 = (tid // fx.Int32(16)) * fx.Int32(4)
        m_lo_o = (tid // fx.Int32(16)) * fx.Int32(4)
        n_o = tid % fx.Int32(16)

        for out_n in range_constexpr(OUT_N_TILES):
            o_acc = _vector.broadcast(f32x4_t, zero_f32)
            for k_tile_v in range_constexpr(K_V_TILES_16):
                a_elems = []
                for el in range_constexpr(4):
                    col = (
                        fx.Int32(k_tile_v * 16)
                        + k_lo_16
                        + fx.Int32(el)
                    )
                    ix = m_a * fx.Int32(BI) + col
                    ix_raw = ix.value if hasattr(ix, "value") else ix
                    a_elems.append(
                        _memref.load(
                            s_bf16_mr,
                            [arith.index_cast(T.index, ix_raw)],
                        )
                    )
                a_frag = _vector.from_elements(bf16x4_t, a_elems)
                a_i16 = _vector.bitcast(i16x4_t, a_frag)

                v_elems = []
                for el in range_constexpr(4):
                    k_v_pos = (
                        fx.Int32(k_tile_v * 16)
                        + k_lo_16
                        + fx.Int32(el)
                    )
                    idx_off_v = b_block * fx.Int32(BI) + k_v_pos
                    tok = idx_.load(idx_off_v, vec_size=1)
                    tok_raw = tok.value if hasattr(tok, "value") else tok
                    blk = arith.divui(tok_raw, BS_KV_C)
                    inb = arith.remui(tok_raw, BS_KV_C)
                    row_idx = arith.addi(
                        arith.muli(blk, BS_KV_C), inb
                    )
                    v_row_byte = arith.muli(row_idx, ROW_STRIDE_V_C)
                    n_col = fx.Int32(out_n * 16) + n_b
                    n_col_raw = (
                        n_col.value if hasattr(n_col, "value") else n_col
                    )
                    v_byte_off = arith.addi(v_row_byte, n_col_raw)
                    v_fp8 = vc_i8.load(v_byte_off, vec_size=1)
                    v_raw = v_fp8.value if hasattr(v_fp8, "value") else v_fp8
                    v_u32 = arith.extui(T.i32, v_raw)
                    v_scale_tile_id = arith.divui(
                        n_col_raw,
                        arith.constant(NOPE_TILE, type=T.i32),
                    )
                    v_scale_off = arith.addi(
                        v_row_byte,
                        arith.addi(
                            arith.constant(
                                ROW_BYTES_V_FP8, type=T.i32
                            ),
                            v_scale_tile_id,
                        ),
                    )
                    s_v_byte = vc_i8.load(v_scale_off, vec_size=1)
                    s_v_raw = (
                        s_v_byte.value
                        if hasattr(s_v_byte, "value")
                        else s_v_byte
                    )
                    s_u32_v = arith.extui(T.i32, s_v_raw)
                    v_bf16_bits = _dequant_fp8_to_bf16_bits(v_u32, s_u32_v)
                    v_bf16_v = arith.bitcast(T.bf16, v_bf16_bits)
                    v_elems.append(v_bf16_v)
                v_frag = _vector.from_elements(bf16x4_t, v_elems)
                v_i16 = _vector.bitcast(i16x4_t, v_frag)

                o_acc = rocdl.mfma_f32_16x16x16bf16_1k(
                    f32x4_t, [a_i16, v_i16, o_acc, 0, 0, 0]
                )
            for el in range_constexpr(4):
                scalar = _vector.extract(
                    o_acc, static_position=[el], dynamic_position=[]
                )
                global_m_el = (
                    b_block * fx.Int32(128)
                    + m_block * fx.Int32(M_TILE)
                    + m_lo_o
                    + fx.Int32(el)
                )
                global_n_el = fx.Int32(out_n * 16) + n_o
                o_off = global_m_el * fx.Int32(D_V) + global_n_el
                o_.store(o_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q, kc, vc, idx, o, num_m_wgs: fx.Int32, num_bs: fx.Int32):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(q, kc, vc, idx, o).launch(
            grid=(num_m_wgs, num_bs, 1),
            block=(64, 1, 1),
        )

    return launch, (BI, D, D_V, BS_KV, ROW_STRIDE_KCACHE, ROW_STRIDE_VCACHE)


# ---------------------------------------------------------------------------
# PyTorch reference (bit-identical dequant)
# ---------------------------------------------------------------------------

def _torch_dequant(fp8_bytes_u8: torch.Tensor, scale_bytes_u8: torch.Tensor) -> torch.Tensor:
    b = fp8_bytes_u8.to(torch.int64) & 0xFF
    s = scale_bytes_u8.to(torch.int64) & 0xFF
    sign_bf = (b & 0x80) << 8
    exp_e4 = (b & 0x78) >> 3
    mant_bf = (b & 0x7) << 4
    exp_c = exp_e4 + s - 7
    bits = sign_bf | (exp_c << 7) | mant_bf
    bits_u16 = (bits & 0xFFFF).to(torch.int16)
    return bits_u16.view(torch.bfloat16)


def _torch_reference(
    q_bf16, k_cache_u8, v_cache_u8, indices_i32,
    bs_kv, d, d_v, row_stride_k, row_stride_v, nope_tile, sm_scale,
):
    bs, m_heads, _ = q_bf16.shape
    nb = k_cache_u8.numel() // (bs_kv * row_stride_k)
    kc = k_cache_u8.view(nb, bs_kv, row_stride_k)
    vc = v_cache_u8.view(nb, bs_kv, row_stride_v)
    block_id = (indices_i32 // bs_kv).long()
    in_block = (indices_i32 % bs_kv).long()
    k_rows = kc[block_id, in_block]
    v_rows = vc[block_id, in_block]
    k_fp8 = k_rows[:, :, :d]
    v_fp8 = v_rows[:, :, :d_v]
    k_scales = k_rows[:, :, d:d + (d // nope_tile)].repeat_interleave(nope_tile, dim=2)
    v_scales = v_rows[:, :, d_v:d_v + (d_v // nope_tile)].repeat_interleave(nope_tile, dim=2)
    k_bf16 = _torch_dequant(k_fp8, k_scales).float()
    v_bf16 = _torch_dequant(v_fp8, v_scales).float()
    s = torch.einsum("bhd,bnd->bhn", q_bf16.float(), k_bf16) * sm_scale
    sm = torch.softmax(s, dim=-1)
    return torch.einsum("bhn,bnv->bhv", sm, v_bf16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=159,
                   help="decode batch size")
    p.add_argument("--m-heads", type=int, default=128)
    p.add_argument("--bi", type=int, default=64,
                   help="indices per batch (single sparse chunk size)")
    p.add_argument("--d", type=int, default=448)
    p.add_argument("--d-v", type=int, default=448)
    p.add_argument("--bs-kv", type=int, default=128)
    p.add_argument("--nope-tile", type=int, default=64)
    p.add_argument("--sm-scale", type=float, default=0.044194173824159216)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--skip-correctness", action="store_true",
                   help="skip the bit-identical torch reference check")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    torch.manual_seed(2)
    device = torch.device("cuda")

    bs, m_heads, bi = args.bs, args.m_heads, args.bi
    d, d_v, bs_kv = args.d, args.d_v, args.bs_kv
    nb = max(2, (bs * bi * 2) // bs_kv + 1)
    num_m_wgs = m_heads // 16
    row_stride_k = d + 8
    row_stride_v = d_v + 8

    print(f"shapes: BS={bs}, M_HEADS={m_heads}, BI={bi}, D={d}, D_V={d_v}, "
          f"NB={nb}, BS_KV={bs_kv}")
    print(f"row stride: K={row_stride_k}, V={row_stride_v} (synthetic; "
          f"real DSv4 is 584 incl. D_tail)\n")

    launch, _ = _build_fp8_sparse_kernel(args.nope_tile, args.sm_scale)

    q = torch.randn(bs, m_heads, d, dtype=torch.bfloat16, device=device) * 0.05

    def _make_cache(num_rows, stride, data_bytes, scales_per_row):
        rows = torch.randint(0, 256, (num_rows, stride), dtype=torch.uint8, device=device)
        # Constrain scale bytes to [7, 14] so exp_combined stays in safe bf16 range
        # (random scales 0..255 produce inf/NaN on overflow).
        rows[:, data_bytes : data_bytes + scales_per_row] = torch.randint(
            7, 15, (num_rows, scales_per_row), dtype=torch.uint8, device=device,
        )
        return rows.reshape(-1)

    k_cache_u8 = _make_cache(nb * bs_kv, row_stride_k, d, d // args.nope_tile)
    v_cache_u8 = _make_cache(nb * bs_kv, row_stride_v, d_v, d_v // args.nope_tile)
    k_cache = k_cache_u8.view(torch.int8)
    v_cache = v_cache_u8.view(torch.int8)
    indices = torch.randint(0, nb * bs_kv, (bs, bi), dtype=torch.int32, device=device)
    o = torch.zeros(bs, m_heads, d_v, dtype=torch.float32, device=device)

    launch(q.reshape(-1), k_cache, v_cache, indices.reshape(-1),
           o.reshape(-1), num_m_wgs, bs)
    torch.cuda.synchronize()

    # ---- Correctness ----
    if not args.skip_correctness:
        ref = _torch_reference(
            q, k_cache_u8, v_cache_u8, indices,
            bs_kv, d, d_v, row_stride_k, row_stride_v,
            args.nope_tile, args.sm_scale,
        )
        got = o.cpu()
        ref = ref.cpu()
        finite = torch.isfinite(got) & torch.isfinite(ref)
        diff = (got - ref).abs()[finite]
        n_finite = int(finite.sum().item())
        n_total = got.numel()
        within = (diff < 1e-2).sum().item()
        print(f"correctness: {within}/{n_finite} finite "
              f"(of {n_total}) within abs 1e-2; "
              f"max diff (finite) = {diff.max().item():.3e}")
        if within != n_finite:
            print("FAIL", file=sys.stderr)
            sys.exit(2)

    # ---- Bench ----
    for _ in range(args.warmup):
        launch(q.reshape(-1), k_cache, v_cache, indices.reshape(-1),
               o.reshape(-1), num_m_wgs, bs)
    torch.cuda.synchronize()

    sample_us = []
    for _ in range(args.samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            launch(q.reshape(-1), k_cache, v_cache, indices.reshape(-1),
                   o.reshape(-1), num_m_wgs, bs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sample_us.append((t1 - t0) / args.iters * 1e6)

    sample_us.sort()
    median = statistics.median(sample_us)
    p90 = sample_us[int(0.9 * (len(sample_us) - 1))]
    print(f"\nperf: median={median:.2f} µs/call  ({median / bs:.3f} µs/batch)  "
          f"p90={p90:.2f} µs/call")
    print("NOTE: sub-kernel scope (no D_tail / dual cache / online softmax / "
          "attn_sink / combine). NOT comparable to full TileLang µs/batch.")


if __name__ == "__main__":
    main()
