"""r051 Round 11: row-wise softmax via LDS cross-lane reduction.

Setup: 1 workgroup, 64 threads (1 wave). Each lane holds f32x4 as if
from an mfma_f32_16x16x32_bf16 result in CDNA3 col-fixed layout:
  lane = (m_lo // 4) * 16 + n  where m_lo = (lane / 16) * 4, n = lane % 16
  lane holds C[m_lo:m_lo+4, n] (4 row elements at column n)

Goal: compute torch.softmax(C, dim=-1) row-wise across n ∈ [0, 16).
Across-lane reduction implemented via LDS spill + cooperative read.

Algorithm:
  1. Each lane writes its 4 f32 to LDS at byte_off = (m_lo+el)*16*4 + n*4
     for el ∈ [0,4). LDS row stride = 16 f32 = 64 bytes.
  2. Barrier.
  3. Lanes 0..15: each handles one row. Reads 16 f32 from LDS row tid,
     computes max + (sum after subtracting max in exp), writes softmax
     back to same row.
  4. Barrier.
  5. (Optionally: each lane re-reads its 4 elements; for this test we
     dump LDS to HBM directly.)

Verification: compare to torch.softmax(C, dim=-1).
"""
import os
import sys
import math
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith, vector as _fvec, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def build_softmax_kernel():
    arch = get_hip_arch()
    print(f"[sm] gpu_arch = {arch}", flush=True)

    # LDS for 16 rows × 16 cols = 256 f32 = 1024 bytes
    ROW_F32 = 16
    COL_F32 = 16
    LDS_BYTES = ROW_F32 * COL_F32 * 4

    allocator = SmemAllocator(None, arch=arch, global_sym_name="softmax_smem_v1")
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_BYTES

    @flyc.kernel(name="softmax_lds_kernel")
    def softmax_kernel(
        in_tensor: fx.Tensor,   # f32 [16*16] - input "mfma result"
        out_tensor: fx.Tensor,  # f32 [16*16] - softmax(input, dim=-1)
    ):
        tid = fx.thread_idx.x

        in_ = GTensor(in_tensor, dtype=T.f32, shape=(-1,))
        out_ = GTensor(out_tensor, dtype=T.f32, shape=(-1,))

        # ---- Lane → C fragment mapping (CDNA3 col-fixed) ----
        # m_lo = (tid / 16) * 4, n = tid % 16
        m_lo = (tid // fx.Int32(16)) * fx.Int32(4)
        n_c = tid % fx.Int32(16)

        # Load my 4 elements from input
        f32x4_t = T.vec(4, T.f32)
        # in_tensor laid out row-major (m * 16 + n).
        # My lane holds C[m_lo:m_lo+4, n_c]. Load each element.
        my_frag_vals = []
        for el in range_constexpr(4):
            row_idx = (m_lo + fx.Int32(el)) * fx.Int32(COL_F32) + n_c
            v = in_.load(row_idx, vec_size=1)
            my_frag_vals.append(v)

        # ---- Step 1: write fragment to LDS at (row, col) byte offset ----
        lds_view = SmemPtr(
            allocator.get_base(), lds_offset, T.f32,
            shape=(ROW_F32 * COL_F32,),
        )
        lds_mr = lds_view.get()

        for el in range_constexpr(4):
            # LDS element index = (m_lo+el)*COL_F32 + n_c
            lds_idx = (m_lo + fx.Int32(el)) * fx.Int32(COL_F32) + n_c
            lds_idx_ix = arith.index_cast(T.index, lds_idx.value if hasattr(lds_idx, "value") else lds_idx)
            v_raw = my_frag_vals[el].value if hasattr(my_frag_vals[el], "value") else my_frag_vals[el]
            from flydsl._mlir.dialects import memref as _memref
            _memref.store(v_raw, lds_mr, [lds_idx_ix])

        # ---- Barrier ----
        rocdl.barrier()

        # ---- Step 2: lanes 0..15 each handle one row ----
        # Use scf.if so only 16 of 64 lanes do the reduction work.
        c_neg_inf = arith.constant(-1e30, type=T.f32)
        c_zero = arith.constant(0.0, type=T.f32)

        # Predicate: tid < 16
        is_worker = arith.cmpi(
            _mlir_arith.CmpIPredicate.ult,
            tid.value if hasattr(tid, "value") else tid,
            arith.constant(16, type=T.i32),
        )

        def _then_body():
            # I'm responsible for row `tid`.
            # Load 16 elements from LDS row tid.
            row_idx_base = tid * fx.Int32(COL_F32)
            row_base_raw = row_idx_base.value if hasattr(row_idx_base, "value") else row_idx_base

            # First pass: find max
            cur_max = c_neg_inf
            row_vals = []
            from flydsl._mlir.dialects import memref as _memref
            for j in range_constexpr(COL_F32):
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                v = _memref.load(lds_mr, [idx_ix])
                row_vals.append(v)
                cur_max = arith.maxnumf(cur_max, v)

            # Second pass: exp(v - max), sum
            cur_sum = c_zero
            exp_vals = []
            for j in range_constexpr(COL_F32):
                d = arith.subf(row_vals[j], cur_max)
                # exp via exp2: exp(x) = exp2(x * log2(e))
                log2e = arith.constant(1.4426950408889634, type=T.f32)
                d_log2e = arith.mulf(d, log2e)
                e = rocdl.exp2(T.f32, d_log2e)
                exp_vals.append(e)
                cur_sum = arith.addf(cur_sum, e)

            # Third pass: divide
            for j in range_constexpr(COL_F32):
                s = arith.divf(exp_vals[j], cur_sum)
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                _memref.store(s, lds_mr, [idx_ix])

            _scf.yield_([])

        def _else_body():
            _scf.yield_([])

        if_op = _scf.IfOp(is_worker, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            _then_body()

        # ---- Barrier ----
        rocdl.barrier()

        # ---- Step 3: every lane writes its (m_lo+el, n_c) back to HBM ----
        from flydsl._mlir.dialects import memref as _memref
        for el in range_constexpr(4):
            lds_idx = (m_lo + fx.Int32(el)) * fx.Int32(COL_F32) + n_c
            lds_idx_ix = arith.index_cast(T.index, lds_idx.value if hasattr(lds_idx, "value") else lds_idx)
            v = _memref.load(lds_mr, [lds_idx_ix])
            out_off = (m_lo + fx.Int32(el)) * fx.Int32(COL_F32) + n_c
            out_.store(out_off, v, vec_size=1)

    @flyc.jit
    def launch(in_t, out_t):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        softmax_kernel(in_t, out_t).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    M, N = 16, 16
    torch.manual_seed(123)

    print("[sm] building kernel...", flush=True)
    launch = build_softmax_kernel()
    print("[sm] launcher built", flush=True)

    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    print(f"[sm] launching...", flush=True)
    launch(x.reshape(-1), out.reshape(-1))
    torch.cuda.synchronize()
    print(f"[sm] kernel ran OK", flush=True)

    ref = torch.softmax(x, dim=-1).cpu()
    got = out.cpu()
    diff = (got - ref).abs()
    print(f"[sm] max abs diff: {diff.max().item():.6e}", flush=True)
    print(f"[sm] mean abs diff: {diff.mean().item():.6e}", flush=True)
    print(f"[sm] row sums (should be 1.0): {got.sum(dim=-1).tolist()[:4]}", flush=True)

    tol = 1e-4
    n_within = (diff < tol).sum().item()
    total = M * N
    if n_within == total:
        print(f"\n[sm] VERDICT: PASS — softmax {total}/{total} within abs tol {tol}", flush=True)
    else:
        print(f"\n[sm] VERDICT: FAIL — only {n_within}/{total}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
