"""r051 Round 8: real QK mfma with HBM data + lane layout correctness.

Tile: M=16, N=16, K=32 (one mfma_f32_16x16x32_bf16 call).
  A = Q tile (16 × 32) bf16
  B = K tile (16 × 32) bf16
  C = Q @ K^T (16 × 16) f32

CDNA3 lane layout for v_mfma_f32_16x16x32_bf16 (64 lanes/wave):
  A[m][k]:  m = lane % 16,  k_lo = (lane // 16) * 8,  el ∈ [0,8)
            => lane holds A[m, k_lo : k_lo + 8] as bf16x8
  B[n][k]:  n = lane % 16,  k_lo = (lane // 16) * 8,  el ∈ [0,8)
            => lane holds B[n, k_lo : k_lo + 8] as bf16x8
  C[m][n]:  m = lane % 16,  n_lo = (lane // 16) * 4,  el ∈ [0,4)
            => lane holds C[m, n_lo : n_lo + 4] as f32x4

Test: Q, K filled with random data; compute Q @ K^T via mfma; compare to
torch.matmul reference.
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
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def build_qk_mfma_kernel():
    arch = get_hip_arch()
    print(f"[qkmfma] gpu_arch = {arch}", flush=True)

    M, N, K = 16, 16, 32

    @flyc.kernel(name="qk_mfma_kernel")
    def qk_kernel(
        q_tensor: fx.Tensor,   # bf16 [M, K] = [16, 32]
        k_tensor: fx.Tensor,   # bf16 [N, K] = [16, 32]
        c_tensor: fx.Tensor,   # f32  [M, N] = [16, 16]
    ):
        tid = fx.thread_idx.x   # 0..63 (one wave)
        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        c_ = GTensor(c_tensor, dtype=T.f32, shape=(-1,))

        # ---- A (Q) fragment: lane holds A[m, k_lo : k_lo + 8] ----
        # m = lane % 16
        # k_lo = (lane // 16) * 8
        m = tid % fx.Int32(16)
        k_lo = (tid // fx.Int32(16)) * fx.Int32(8)
        # Element offset in Q[16, 32] = m * 32 + k_lo
        a_elem_off = m * fx.Int32(K) + k_lo
        # buffer_load 8 bf16 = 16 bytes
        a_frag = q_.load(a_elem_off, vec_size=8)

        # ---- B (K) fragment ----
        # n = lane % 16 (= same as m)
        n = tid % fx.Int32(16)
        b_elem_off = n * fx.Int32(K) + k_lo
        b_frag = k_.load(b_elem_off, vec_size=8)

        # ---- C init to zero ----
        f32x4_t = T.vec(4, T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)
        c_init = _vector.broadcast(f32x4_t, zero_f32)

        # ---- mfma ----
        c_out = rocdl.mfma_f32_16x16x32_bf16(
            f32x4_t,
            [a_frag, b_frag, c_init, 0, 0, 0],
        )

        # ---- Store C: CDNA3 mfma_f32_16x16x32 output layout ----
        # lane = m_block * 16 + n, where m_block = lane // 16, n = lane % 16
        # Per lane: 4 f32 values at C[m_block*4 + el, n] for el ∈ [0, 4)
        # (column fixed, 4 consecutive rows -- stride N=16 between elements).
        n_c = tid % fx.Int32(16)
        m_lo = (tid // fx.Int32(16)) * fx.Int32(4)
        # Extract individual scalars and store one by one (non-contiguous).
        for el in range_constexpr(4):
            scalar = _vector.extract(c_out, static_position=[el], dynamic_position=[])
            c_off = (m_lo + fx.Int32(el)) * fx.Int32(N) + n_c
            c_.store(c_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q_tensor, k_tensor, c_tensor):
        qk_kernel(q_tensor, k_tensor, c_tensor).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    M, N, K = 16, 16, 32
    torch.manual_seed(42)

    print("[qkmfma] building kernel...", flush=True)
    launch = build_qk_mfma_kernel()
    print("[qkmfma] launcher built", flush=True)

    # Random data
    q = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    q_flat = q.reshape(-1).contiguous()
    k_flat = k.reshape(-1).contiguous()
    c_flat = c.reshape(-1).contiguous()

    try:
        print("[qkmfma] launching...", flush=True)
        launch(q_flat, k_flat, c_flat)
        torch.cuda.synchronize()
        print("[qkmfma] kernel ran OK", flush=True)
    except Exception as e:
        print(f"[qkmfma] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    got = c_flat.view(M, N).cpu()
    # Reference: Q @ K^T  (both bf16, accumulate in f32)
    ref = (q.float() @ k.float().T).cpu()

    diff = (got - ref).abs()
    rel = diff / (ref.abs() + 1e-6)
    print(f"[qkmfma] got[0]: {got[0, :4].tolist()}", flush=True)
    print(f"[qkmfma] ref[0]: {ref[0, :4].tolist()}", flush=True)
    print(f"[qkmfma] max abs diff: {diff.max().item():.6f}", flush=True)
    print(f"[qkmfma] max rel diff: {rel.max().item():.6f}", flush=True)
    print(f"[qkmfma] mean abs diff: {diff.mean().item():.6f}", flush=True)

    # bf16 acc → f32 should match within ~0.01 absolute (32 ops of bf16 mul-add)
    tol = 0.05
    n_within = (diff < tol).sum().item()
    print(f"[qkmfma] {n_within}/256 elements within abs tol {tol}", flush=True)

    if n_within == 256:
        print(f"\n[qkmfma] VERDICT: PASS — Q@K^T mfma matches torch reference", flush=True)
    else:
        print(f"\n[qkmfma] VERDICT: FAIL — lane layout likely wrong", flush=True)
        # Sample bad rows
        bad = (diff >= tol).nonzero()[:5].tolist()
        for m_i, n_i in bad:
            print(f"   C[{m_i},{n_i}] got={got[m_i,n_i].item():.4f} ref={ref[m_i,n_i].item():.4f}",
                  flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
