"""r051 Round 9: full DSv4-sized QK gemm via FlyDSL mfma.

Shape matches dpsk_v4_fp8_partial_kernel inner gemm (excluding tail/D_tail):
  M = H_per_block = 128   (one workgroup processes 16 m rows)
  N = BI = 64             (one workgroup processes 16 n cols)
  K = D = 448             (14 K-tiles of size 32 each, accumulated)

Grid: (M/16, N/16) = (8, 4) = 32 workgroups
Threads per wg: 64 (= 1 wave)

Per workgroup: accumulate Q @ K^T over 14 K-tiles, write 16x16 acc to
output[m_block, n_block, :, :].

Validation: compare full 128x64 output to torch.matmul(Q, K^T).
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


def build_qk_d448_kernel():
    arch = get_hip_arch()
    print(f"[qkd448] gpu_arch = {arch}", flush=True)
    M, N, K = 128, 64, 448
    MFMA_M, MFMA_N, MFMA_K = 16, 16, 32
    NK_TILES = K // MFMA_K  # 14

    @flyc.kernel(name="qk_d448_kernel")
    def qk_kernel(
        q_tensor: fx.Tensor,   # bf16 flat [M*K]
        k_tensor: fx.Tensor,   # bf16 flat [N*K]
        c_tensor: fx.Tensor,   # f32  flat [M*N]
    ):
        tid = fx.thread_idx.x   # 0..63
        m_block = fx.block_idx.x   # 0..M/16 = 7
        n_block = fx.block_idx.y   # 0..N/16 = 3

        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        c_ = GTensor(c_tensor, dtype=T.f32, shape=(-1,))

        # Lane mapping (CDNA3 mfma_f32_16x16x32_bf16):
        #   A[m][k]: lane = (k/8)*16 + m   → m = lane%16, k_lo_in_tile = (lane/16)*8
        #   B[n][k]: lane = (k/8)*16 + n   → n = lane%16
        m_in_tile = tid % fx.Int32(16)
        n_in_tile = tid % fx.Int32(16)
        k_lo_in_tile = (tid // fx.Int32(16)) * fx.Int32(8)

        # Global row/col in Q / K
        global_m = m_block * fx.Int32(MFMA_M) + m_in_tile
        global_n = n_block * fx.Int32(MFMA_N) + n_in_tile

        # ---- Accumulate over K-tiles ----
        f32x4_t = T.vec(4, T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)
        acc = _vector.broadcast(f32x4_t, zero_f32)

        for k_tile in range_constexpr(NK_TILES):
            # K-tile k start = k_tile * MFMA_K
            k_tile_off = fx.Int32(k_tile * MFMA_K) + k_lo_in_tile

            # Load A fragment: Q[global_m, k_tile_off : k_tile_off + 8]
            a_elem_off = global_m * fx.Int32(K) + k_tile_off
            a_frag = q_.load(a_elem_off, vec_size=8)

            # Load B fragment: K[global_n, k_tile_off : k_tile_off + 8]
            b_elem_off = global_n * fx.Int32(K) + k_tile_off
            b_frag = k_.load(b_elem_off, vec_size=8)

            # mfma accumulate
            acc = rocdl.mfma_f32_16x16x32_bf16(
                f32x4_t,
                [a_frag, b_frag, acc, 0, 0, 0],
            )

        # ---- Store C: col-fixed, 4 rows per lane (stride N in M dir) ----
        # lane = (m/4)*16 + n, lane holds C[m_lo:m_lo+4, n]
        n_c_in_tile = tid % fx.Int32(16)
        m_lo_in_tile = (tid // fx.Int32(16)) * fx.Int32(4)
        global_n_c = n_block * fx.Int32(MFMA_N) + n_c_in_tile

        for el in range_constexpr(4):
            scalar = _vector.extract(acc, static_position=[el], dynamic_position=[])
            global_m_el = m_block * fx.Int32(MFMA_M) + m_lo_in_tile + fx.Int32(el)
            c_off = global_m_el * fx.Int32(N) + global_n_c
            c_.store(c_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q_t, k_t, c_t):
        qk_kernel(q_t, k_t, c_t).launch(
            grid=(M // MFMA_M, N // MFMA_N, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    M, N, K = 128, 64, 448
    torch.manual_seed(42)

    print("[qkd448] building kernel...", flush=True)
    launch = build_qk_d448_kernel()
    print("[qkd448] launcher built", flush=True)

    q = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    print(f"[qkd448] launching... grid=(8, 4), threads=64", flush=True)
    try:
        launch(q.reshape(-1), k.reshape(-1), c.reshape(-1))
        torch.cuda.synchronize()
        print("[qkd448] kernel ran OK", flush=True)
    except Exception as e:
        print(f"[qkd448] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    got = c.cpu()
    ref = (q.float() @ k.float().T).cpu()
    diff = (got - ref).abs()
    rel = diff / (ref.abs() + 1e-6)
    print(f"[qkd448] max abs diff: {diff.max().item():.6f}", flush=True)
    print(f"[qkd448] max rel diff: {rel.max().item():.6f}", flush=True)
    print(f"[qkd448] mean abs diff: {diff.mean().item():.6f}", flush=True)

    # bf16 accumulating 14 K-tiles, expect tiny residual
    tol = 0.05
    n_within = (diff < tol).sum().item()
    total = M * N
    print(f"[qkd448] {n_within}/{total} within abs tol {tol}", flush=True)

    if n_within == total:
        print(f"\n[qkd448] VERDICT: PASS — full d=448 Q@K^T matches torch", flush=True)
    else:
        print(f"\n[qkd448] VERDICT: FAIL", flush=True)
        bad = (diff >= tol).nonzero()[:5].tolist()
        for r, c_i in bad:
            print(f"   C[{r},{c_i}] got={got[r,c_i]:.4f} ref={ref[r,c_i]:.4f}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
