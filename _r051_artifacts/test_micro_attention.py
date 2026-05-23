"""r051 Round 12: full micro-attention — Q@K → softmax → S@V, all FlyDSL.

Composes all building blocks from rounds 7-11 into one kernel:
  - mfma_f32_16x16x32_bf16   (round 7-9)
  - LDS cross-lane reduction (round 11)
  - chained mfma with bf16 cast + LDS transpose for fragment layout

Shape: M=N=K=16, with K-tile of 32 for mfma. So:
  Q: (16, 32) bf16
  K: (16, 32) bf16  → Q @ K^T = S (16, 16) f32
  S_norm = softmax(S, dim=-1) → bf16
  V: (16, 32) bf16  → S_norm @ V_T = O (16, 16) f32  (where V_T is V transposed)

Note: second mfma input shape is 16x16 for A (S_norm), 16x16 for B (V_T).
Use mfma_f32_16x16x16_bf16: A=16x16 bf16, B=16x16 bf16, C=16x16 f32.

Validation: compare O to torch.softmax(Q @ K.T) @ V (with V reshaped appropriately).
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


def build_micro_attention():
    arch = get_hip_arch()

    # LDS layout:
    #   [0 .. 256*4): S f32 = 16x16 = 256 f32
    #   [256*4 .. 256*4 + 256*2): S_bf16 = 16x16 = 256 bf16
    LDS_S_F32_BYTES = 16 * 16 * 4   # 1024
    LDS_S_BF16_BYTES = 16 * 16 * 2  # 512

    allocator = SmemAllocator(None, arch=arch, global_sym_name="micro_attn_smem_v1")
    lds_s_f32_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_f32_off + LDS_S_F32_BYTES
    lds_s_bf16_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_s_bf16_off + LDS_S_BF16_BYTES

    @flyc.kernel(name="micro_attention_kernel")
    def kernel(
        q_tensor: fx.Tensor,    # bf16 [16*32]
        k_tensor: fx.Tensor,    # bf16 [16*32]
        v_tensor: fx.Tensor,    # bf16 [16*16]  (think V_T: rows = K dim, cols = n dim)
        o_tensor: fx.Tensor,    # f32  [16*16]
    ):
        tid = fx.thread_idx.x  # 0..63

        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        o_ = GTensor(o_tensor, dtype=T.f32, shape=(-1,))

        # ============================================================
        # Step 1: Q @ K^T via mfma_f32_16x16x32_bf16
        # ============================================================
        m_in = tid % fx.Int32(16)
        n_in = tid % fx.Int32(16)
        k_lo_32 = (tid // fx.Int32(16)) * fx.Int32(8)

        a_off = m_in * fx.Int32(32) + k_lo_32
        b_off = n_in * fx.Int32(32) + k_lo_32
        q_frag = q_.load(a_off, vec_size=8)
        k_frag = k_.load(b_off, vec_size=8)

        f32x4_t = T.vec(4, T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)
        s_acc = _vector.broadcast(f32x4_t, zero_f32)
        s_acc = rocdl.mfma_f32_16x16x32_bf16(
            f32x4_t, [q_frag, k_frag, s_acc, 0, 0, 0]
        )

        # ============================================================
        # Step 2: spill S to LDS (f32 layout), then softmax (lanes 0..15)
        # ============================================================
        m_lo_c = (tid // fx.Int32(16)) * fx.Int32(4)
        n_c = tid % fx.Int32(16)

        s_lds = SmemPtr(allocator.get_base(), lds_s_f32_off, T.f32, shape=(256,))
        s_lds_mr = s_lds.get()

        for el in range_constexpr(4):
            scalar = _vector.extract(s_acc, static_position=[el], dynamic_position=[])
            ix = (m_lo_c + fx.Int32(el)) * fx.Int32(16) + n_c
            ix_raw = ix.value if hasattr(ix, "value") else ix
            ix_ix = arith.index_cast(T.index, ix_raw)
            _memref.store(scalar, s_lds_mr, [ix_ix])

        rocdl.barrier()

        # Softmax: lanes 0..15 each handle one row
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
            row_base = tid * fx.Int32(16)
            row_base_raw = row_base.value if hasattr(row_base, "value") else row_base

            row_vals = []
            cur_max = c_neg_inf
            for j in range_constexpr(16):
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                v = _memref.load(s_lds_mr, [idx_ix])
                row_vals.append(v)
                cur_max = arith.maxnumf(cur_max, v)

            exp_vals = []
            cur_sum = c_zero
            for j in range_constexpr(16):
                d = arith.subf(row_vals[j], cur_max)
                d_l = arith.mulf(d, log2e)
                e = rocdl.exp2(T.f32, d_l)
                exp_vals.append(e)
                cur_sum = arith.addf(cur_sum, e)

            for j in range_constexpr(16):
                s = arith.divf(exp_vals[j], cur_sum)
                idx = arith.addi(row_base_raw, arith.constant(j, type=T.i32))
                idx_ix = arith.index_cast(T.index, idx)
                _memref.store(s, s_lds_mr, [idx_ix])

            _scf.yield_([])

        rocdl.barrier()

        # ============================================================
        # Step 3: cast S f32 → bf16, store to second LDS region
        # ============================================================
        s_bf16_lds = SmemPtr(
            allocator.get_base(), lds_s_bf16_off, T.bf16, shape=(256,)
        )
        s_bf16_mr = s_bf16_lds.get()

        for el in range_constexpr(4):
            ix = (m_lo_c + fx.Int32(el)) * fx.Int32(16) + n_c
            ix_raw = ix.value if hasattr(ix, "value") else ix
            ix_ix = arith.index_cast(T.index, ix_raw)
            f32_v = _memref.load(s_lds_mr, [ix_ix])
            bf16_v = arith.truncf(T.bf16, f32_v)
            _memref.store(bf16_v, s_bf16_mr, [ix_ix])

        rocdl.barrier()

        # ============================================================
        # Step 4: S_bf16 @ V → O via mfma_f32_16x16x16_bf16
        # ============================================================
        # A (S_bf16) layout: lane = (k/4)*16 + m, lane holds A[m, k_lo:k_lo+4] bf16x4
        # B (V)      layout: lane = (k/4)*16 + n, lane holds B[n, k_lo:k_lo+4] bf16x4
        m_in_a = tid % fx.Int32(16)
        n_in_b = tid % fx.Int32(16)
        k_lo_16 = (tid // fx.Int32(16)) * fx.Int32(4)

        # Read A from LDS: 4 consecutive bf16 at S_bf16[m, k_lo:k_lo+4]
        bf16x4_t = T.vec(4, T.bf16)
        # Construct A frag by 4 scalar loads (could vectorize, but keeps simple).
        a_elems = []
        for el in range_constexpr(4):
            a_lds_ix = m_in_a * fx.Int32(16) + k_lo_16 + fx.Int32(el)
            a_lds_raw = a_lds_ix.value if hasattr(a_lds_ix, "value") else a_lds_ix
            a_lds_ix_ix = arith.index_cast(T.index, a_lds_raw)
            a_v = _memref.load(s_bf16_mr, [a_lds_ix_ix])
            a_elems.append(a_v)
        a2_frag = _vector.from_elements(bf16x4_t, a_elems)

        # Read B from V HBM: V[n_in_b, k_lo:k_lo+4]
        v_off = n_in_b * fx.Int32(16) + k_lo_16
        v_frag = v_.load(v_off, vec_size=4)

        # mfma_f32_16x16x16bf16_1k expects vector<4xi16> (legacy gfx9 ABI),
        # so bitcast bf16x4 → i16x4.
        i16x4_t = T.vec(4, T.i16)
        a2_i16 = _vector.bitcast(i16x4_t, a2_frag)
        v_i16 = _vector.bitcast(i16x4_t, v_frag)

        o_acc = _vector.broadcast(f32x4_t, zero_f32)
        o_acc = rocdl.mfma_f32_16x16x16bf16_1k(
            f32x4_t, [a2_i16, v_i16, o_acc, 0, 0, 0]
        )

        # ============================================================
        # Step 5: store O (col-fixed CDNA3 layout)
        # ============================================================
        m_lo_o = (tid // fx.Int32(16)) * fx.Int32(4)
        n_o = tid % fx.Int32(16)
        for el in range_constexpr(4):
            scalar = _vector.extract(o_acc, static_position=[el], dynamic_position=[])
            o_off = (m_lo_o + fx.Int32(el)) * fx.Int32(16) + n_o
            o_.store(o_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q_t, k_t, v_t, o_t):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(q_t, k_t, v_t, o_t).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    torch.manual_seed(7)
    M, K, N = 16, 32, 16   # Q@K^T: 16x32 @ 32x16 → 16x16
    print("[ma] building micro-attention kernel...", flush=True)
    launch = build_micro_attention()
    print("[ma] launcher built", flush=True)

    q = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(M, M, dtype=torch.bfloat16, device="cuda") * 0.1  # 16x16
    o = torch.zeros(M, M, dtype=torch.float32, device="cuda")

    print("[ma] launching...", flush=True)
    launch(q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1))
    torch.cuda.synchronize()
    print("[ma] kernel ran OK", flush=True)

    # Reference: softmax(Q @ K^T) @ V^T
    # (kernel loads B = V[n, k], which corresponds to V^T in standard layout)
    s_ref = q.float() @ k.float().T
    s_ref_sm = torch.softmax(s_ref, dim=-1)
    o_ref = s_ref_sm @ v.float().T

    got = o.cpu()
    ref = o_ref.cpu()
    diff = (got - ref).abs()
    print(f"[ma] max abs diff: {diff.max().item():.6e}", flush=True)
    print(f"[ma] mean abs diff: {diff.mean().item():.6e}", flush=True)
    print(f"[ma] got[0]: {got[0, :4].tolist()}", flush=True)
    print(f"[ma] ref[0]: {ref[0, :4].tolist()}", flush=True)

    # bf16 precision in two mfmas + bf16 cast on S → tolerance ~ 5e-3
    tol = 5e-3
    n_within = (diff < tol).sum().item()
    total = M * M
    if n_within == total:
        print(f"\n[ma] VERDICT: PASS — micro-attention {total}/{total} within abs tol {tol}",
              flush=True)
    else:
        print(f"\n[ma] VERDICT: FAIL — only {n_within}/{total}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
