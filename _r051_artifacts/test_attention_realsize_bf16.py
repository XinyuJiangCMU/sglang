"""r051 Round 13: full-size BF16 attention kernel matching dpsk_v4 partial shape.

Shape matches dpsk_v4_fp8_partial_kernel (without FP8 + without sparse for now):
  M = H_per_block = 16   (one m-tile per workgroup; tilelang uses 64-128)
  N = BI = 64            (4 n-tiles processed inside each WG for row softmax)
  K = D = 448            (14 K-tiles per gemm, accumulated)
  K_v = BI = 64          (S@V K dim)
  D_v = 16               (V output dim — simplified; real DSv4 is 448)

Grid: (M_tiles, 1) = (M/16, 1) = (8, 1) workgroups when M_total = 128.

Per workgroup:
  1. QK gemm: loop n_tile ∈ [0,4), k_tile ∈ [0,14), mfma into acc[n_tile]
     → 4 × f32x4 per lane (4 accumulators, each 16x16 = 4 per lane)
  2. Softmax across N=64:
     spill all acc to LDS (16x64 f32), barrier
     lanes 0..15 each do one row's 64-element softmax, write back
  3. S@V gemm: BS in LDS bf16, V from HBM bf16
     For simplicity D_v = 16 → one mfma 16x16x16 per m-tile per n_subtile
     Sum across K=BI=64 via 4 mfma calls (K split 64/16=4)
  4. Write output to HBM.

This is the "real-shape" kernel, no FP8 yet, no sparse yet, no online
softmax across multiple K iterations (just one pass over full K).
Correctness vs torch reference.
"""
import os
import sys
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith, vector as _fvec, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def build_attention():
    arch = get_hip_arch()

    # Per WG: 16x64 S tile (after QK), then 16x16 O tile (after SV).
    LDS_S_F32_BYTES = 16 * 64 * 4   # 4096
    LDS_S_BF16_BYTES = 16 * 64 * 2  # 2048

    allocator = SmemAllocator(None, arch=arch, global_sym_name="attn_real_smem_v1")
    lds_s_f32_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_f32_off + LDS_S_F32_BYTES
    lds_s_bf16_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_bf16_off + LDS_S_BF16_BYTES

    M_TILE = 16
    N_TILE = 16
    K_MFMA = 32
    BI = 64
    D = 448
    D_V = 448           # real DSv4 output dim
    N_SUBTILES = BI // N_TILE       # 4 (QK side)
    K_TILES = D // K_MFMA            # 14
    K_V_TILES_16 = BI // 16          # 4 (for S@V using mfma 16x16x16)
    OUT_N_TILES = D_V // N_TILE      # 28 (output sub-tiles in D_V direction)

    @flyc.kernel(name="attn_real_bf16_kernel")
    def kernel(
        q_tensor: fx.Tensor,    # bf16 [M_total * D]
        k_tensor: fx.Tensor,    # bf16 [BI * D]   (one wg handles same K for all rows)
        v_tensor: fx.Tensor,    # bf16 [BI * D_V]
        o_tensor: fx.Tensor,    # f32  [M_total * D_V]
    ):
        tid = fx.thread_idx.x  # 0..63
        m_block = fx.block_idx.x  # 0..M_total/M_TILE - 1

        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        o_ = GTensor(o_tensor, dtype=T.f32, shape=(-1,))

        # Lane mapping for mfma A/B (K=32) load.
        m_in_tile = tid % fx.Int32(16)
        k_lo_32 = (tid // fx.Int32(16)) * fx.Int32(8)
        global_m = m_block * fx.Int32(M_TILE) + m_in_tile

        # ======================================================
        # Step 1: QK gemm — accumulate 4 N-tiles × 14 K-tiles
        # ======================================================
        f32x4_t = T.vec(4, T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)

        # 4 accumulators, one per n_tile.
        accs = [_vector.broadcast(f32x4_t, zero_f32) for _ in range_constexpr(N_SUBTILES)]

        for n_tile in range_constexpr(N_SUBTILES):
            n_in_tile = tid % fx.Int32(16)
            global_n = fx.Int32(n_tile * N_TILE) + n_in_tile
            for k_tile in range_constexpr(K_TILES):
                k_off = fx.Int32(k_tile * K_MFMA) + k_lo_32
                a_off = global_m * fx.Int32(D) + k_off
                b_off = global_n * fx.Int32(D) + k_off
                a_frag = q_.load(a_off, vec_size=8)
                b_frag = k_.load(b_off, vec_size=8)
                accs[n_tile] = rocdl.mfma_f32_16x16x32_bf16(
                    f32x4_t, [a_frag, b_frag, accs[n_tile], 0, 0, 0]
                )

        # ======================================================
        # Step 2: spill all 4 accumulators to LDS in (m, n) layout
        # ======================================================
        # acc[n_tile] lane (m/4)*16 + n_in_tile holds C[m_lo:m_lo+4, n_in_tile]
        # where columns of acc[n_tile] are global_n = n_tile * 16 + n_in_tile
        m_lo_c = (tid // fx.Int32(16)) * fx.Int32(4)
        n_c = tid % fx.Int32(16)

        s_lds = SmemPtr(allocator.get_base(), lds_s_f32_off, T.f32,
                        shape=(M_TILE * BI,))
        s_lds_mr = s_lds.get()

        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                scalar = _vector.extract(accs[n_tile], static_position=[el], dynamic_position=[])
                # LDS layout: [m_tile_row][n_global] = m * BI + (n_tile*16 + n_c)
                ix = (m_lo_c + fx.Int32(el)) * fx.Int32(BI) + fx.Int32(n_tile * N_TILE) + n_c
                ix_raw = ix.value if hasattr(ix, "value") else ix
                ix_ix = arith.index_cast(T.index, ix_raw)
                _memref.store(scalar, s_lds_mr, [ix_ix])

        rocdl.barrier()

        # ======================================================
        # Step 3: softmax row-wise across N=BI=64. Lanes 0..15 work.
        # ======================================================
        is_worker = arith.cmpi(
            _mlir_arith.CmpIPredicate.ult,
            tid.value if hasattr(tid, "value") else tid,
            arith.constant(16, type=T.i32),
        )
        log2e = arith.constant(1.4426950408889634, type=T.f32)
        c_neg_inf = arith.constant(-1e30, type=T.f32)
        c_zero = arith.constant(0.0, type=T.f32)

        if_op = _scf.IfOp(is_worker, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            row_base = tid * fx.Int32(BI)
            row_base_raw = row_base.value if hasattr(row_base, "value") else row_base

            row_vals = []
            cur_max = c_neg_inf
            for j in range_constexpr(BI):
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                v = _memref.load(s_lds_mr, [idx_ix])
                row_vals.append(v)
                cur_max = arith.maxnumf(cur_max, v)

            exp_vals = []
            cur_sum = c_zero
            for j in range_constexpr(BI):
                d = arith.subf(row_vals[j], cur_max)
                d_l = arith.mulf(d, log2e)
                e = rocdl.exp2(T.f32, d_l)
                exp_vals.append(e)
                cur_sum = arith.addf(cur_sum, e)

            for j in range_constexpr(BI):
                s = arith.divf(exp_vals[j], cur_sum)
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                _memref.store(s, s_lds_mr, [idx_ix])
            _scf.yield_([])

        rocdl.barrier()

        # ======================================================
        # Step 4: cast S f32 → bf16 to second LDS region
        # ======================================================
        s_bf16_lds = SmemPtr(
            allocator.get_base(), lds_s_bf16_off, T.bf16, shape=(M_TILE * BI,)
        )
        s_bf16_mr = s_bf16_lds.get()

        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                ix = (m_lo_c + fx.Int32(el)) * fx.Int32(BI) + fx.Int32(n_tile * N_TILE) + n_c
                ix_raw = ix.value if hasattr(ix, "value") else ix
                ix_ix = arith.index_cast(T.index, ix_raw)
                f32_v = _memref.load(s_lds_mr, [ix_ix])
                bf16_v = arith.truncf(T.bf16, f32_v)
                _memref.store(bf16_v, s_bf16_mr, [ix_ix])

        rocdl.barrier()

        # ======================================================
        # Step 5: S @ V using mfma_f32_16x16x16bf16_1k
        # S is 16xBI (M_TILE x 64), V is BIxD_V (64x448) → O is 16xD_V (16x448)
        # Loop over OUT_N_TILES=28 output sub-tiles in D_V direction; for each,
        # do K_V_TILES_16=4 mfma's accumulating into a per-tile o_acc, then
        # write 16x16 tile to HBM.
        # ======================================================
        i16x4_t = T.vec(4, T.i16)
        bf16x4_t = T.vec(4, T.bf16)
        m_a = tid % fx.Int32(16)
        n_b = tid % fx.Int32(16)
        k_lo_16 = (tid // fx.Int32(16)) * fx.Int32(4)
        m_lo_o = (tid // fx.Int32(16)) * fx.Int32(4)
        n_o = tid % fx.Int32(16)

        for out_n in range_constexpr(OUT_N_TILES):
            o_acc = _vector.broadcast(f32x4_t, zero_f32)
            for k_tile_v in range_constexpr(K_V_TILES_16):
                # A (S) frag — same for every out_n
                a_elems = []
                for el in range_constexpr(4):
                    col = fx.Int32(k_tile_v * 16) + k_lo_16 + fx.Int32(el)
                    ix = m_a * fx.Int32(BI) + col
                    ix_raw = ix.value if hasattr(ix, "value") else ix
                    ix_ix = arith.index_cast(T.index, ix_raw)
                    a_elems.append(_memref.load(s_bf16_mr, [ix_ix]))
                a_frag = _vector.from_elements(bf16x4_t, a_elems)
                a_i16 = _vector.bitcast(i16x4_t, a_frag)

                # B (V) frag: V[k_lo+0..3, out_n*16 + n_b]
                v_elems = []
                for el in range_constexpr(4):
                    k_row = fx.Int32(k_tile_v * 16) + k_lo_16 + fx.Int32(el)
                    n_col = fx.Int32(out_n * 16) + n_b
                    v_off_el = k_row * fx.Int32(D_V) + n_col
                    v_elems.append(v_.load(v_off_el, vec_size=1))
                v_frag = _vector.from_elements(bf16x4_t, v_elems)
                v_i16 = _vector.bitcast(i16x4_t, v_frag)

                o_acc = rocdl.mfma_f32_16x16x16bf16_1k(
                    f32x4_t, [a_i16, v_i16, o_acc, 0, 0, 0]
                )

            # Write this 16x16 sub-tile to HBM
            for el in range_constexpr(4):
                scalar = _vector.extract(o_acc, static_position=[el], dynamic_position=[])
                global_m_el = m_block * fx.Int32(M_TILE) + m_lo_o + fx.Int32(el)
                global_n_el = fx.Int32(out_n * 16) + n_o
                o_off = global_m_el * fx.Int32(D_V) + global_n_el
                o_.store(o_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q_t, k_t, v_t, o_t, num_m_wgs: fx.Int32):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(q_t, k_t, v_t, o_t).launch(
            grid=(num_m_wgs, 1, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    torch.manual_seed(99)
    M = 128
    BI = 64
    D = 448
    D_V = 448
    M_TILE = 16
    NUM_M_WGS = M // M_TILE

    print(f"[attn] building kernel M={M}, BI={BI}, D={D}, D_V={D_V}...", flush=True)
    launch = build_attention()
    print("[attn] launcher built", flush=True)

    q = torch.randn(M, D, dtype=torch.bfloat16, device="cuda") * 0.05
    k = torch.randn(BI, D, dtype=torch.bfloat16, device="cuda") * 0.05
    v = torch.randn(BI, D_V, dtype=torch.bfloat16, device="cuda") * 0.05
    o = torch.zeros(M, D_V, dtype=torch.float32, device="cuda")

    print(f"[attn] launching {NUM_M_WGS} WGs...", flush=True)
    launch(q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1), NUM_M_WGS)
    torch.cuda.synchronize()
    print("[attn] kernel ran OK", flush=True)

    # Reference: softmax(Q @ K.T) @ V
    # V has shape (BI, D_V) = (64, 16), standard (k, n) layout.
    s_ref = q.float() @ k.float().T
    s_ref_sm = torch.softmax(s_ref, dim=-1)
    o_ref = s_ref_sm @ v.float()

    diff = (o.cpu() - o_ref.cpu()).abs()
    print(f"[attn] max abs diff: {diff.max().item():.6e}", flush=True)
    print(f"[attn] mean abs diff: {diff.mean().item():.6e}", flush=True)
    print(f"[attn] o[0,:4]: {o.cpu()[0, :4].tolist()}", flush=True)
    print(f"[attn] ref[0,:4]: {o_ref.cpu()[0, :4].tolist()}", flush=True)

    tol = 5e-3
    n_within = (diff < tol).sum().item()
    total = M * D_V
    if n_within == total:
        print(f"\n[attn] VERDICT: PASS — {total}/{total} within abs tol {tol}", flush=True)
    else:
        print(f"\n[attn] VERDICT: {n_within}/{total} within tol", flush=True)
        sys.exit(2)

    # ---- Microbenchmark ----
    import time
    # Warmup
    for _ in range(10):
        launch(q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1), NUM_M_WGS)
    torch.cuda.synchronize()

    # Time
    N_ITERS = 200
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        launch(q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1), NUM_M_WGS)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_call_us = (t1 - t0) / N_ITERS * 1e6
    print(f"\n[attn] perf: {per_call_us:.2f} µs / call (M={M}, BI={BI}, D={D}, D_V={D_V})",
          flush=True)


if __name__ == "__main__":
    main()
