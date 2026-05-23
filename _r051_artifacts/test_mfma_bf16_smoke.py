"""r051 Round 7: smoke-test mfma_f32_16x16x32_bf16 on gfx950.

Minimal kernel:
  - 1 workgroup, 64 threads (= 1 wave)
  - Each thread holds A fragment (bf16x8), B fragment (bf16x8), acc (f32x4=0)
  - Single mfma_f32_16x16x32_bf16 call
  - Write C to HBM
  - Verify kernel runs + ISA shows `v_mfma_f32_16x16x32_bf16`

This proves the mfma op compiles and lowers on MI355X. Correctness
(lane layout) deferred to a subsequent round once compile path is proven.
"""
import os
import sys
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith, vector as _fvec
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def build_mfma_smoke():
    arch = get_hip_arch()
    print(f"[mfma] gpu_arch = {arch}", flush=True)

    @flyc.kernel(name="mfma_bf16_smoke")
    def smoke_kernel(out_tensor: fx.Tensor):
        tid = fx.thread_idx.x
        out_ = GTensor(out_tensor, dtype=T.f32, shape=(-1,))

        # A bf16x8 = constant 1.0 (all elements)
        bf16x8_t = T.vec(8, T.bf16)
        f32x4_t = T.vec(4, T.f32)
        # zero acc
        zero_f32 = arith.constant(0.0, type=T.f32)
        acc = _vector.broadcast(f32x4_t, zero_f32)
        # one bf16 constant
        one_bf16 = arith.constant(1.0, type=T.bf16)
        a_frag = _vector.broadcast(bf16x8_t, one_bf16)
        b_frag = _vector.broadcast(bf16x8_t, one_bf16)

        # cbsz/abid/blgp are Python int constants (not ArithValue).
        # mfma_f32_16x16x32_bf16(result_type, [a, b, c, cbsz, abid, blgp])
        result = rocdl.mfma_f32_16x16x32_bf16(
            f32x4_t,
            [a_frag, b_frag, acc, 0, 0, 0],
        )
        # result is a v4f32 - write to out_tensor[tid*4 .. tid*4+4]
        out_off = tid * fx.Int32(4)
        out_.store(out_off, result, vec_size=4)

    @flyc.jit
    def launch(out_tensor: fx.Tensor):
        smoke_kernel(out_tensor).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    print("[mfma] building smoke kernel...", flush=True)
    launch = build_mfma_smoke()
    print("[mfma] launcher built", flush=True)

    # output: 64 lanes × 4 f32 = 256 f32
    out_t = torch.zeros(256, dtype=torch.float32, device="cuda")

    try:
        print("[mfma] launching...", flush=True)
        launch(out_t)
        torch.cuda.synchronize()
        print("[mfma] kernel ran OK", flush=True)
    except Exception as e:
        print(f"[mfma] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # All-ones A * all-ones B should give: each C element = sum over K (32)
    # of 1*1 = 32.0
    print(f"[mfma] out[:8] = {out_t[:8].tolist()}", flush=True)
    # Expected all = 32.0
    if torch.all(out_t == 32.0):
        print("[mfma] CORRECTNESS PASS — all 256 output elements = 32.0 (= K = 32)", flush=True)
    else:
        n_ok = (out_t == 32.0).sum().item()
        print(f"[mfma] partial: {n_ok}/256 elements == 32.0", flush=True)
        print(f"[mfma]   unique values: {torch.unique(out_t).tolist()}", flush=True)

    print("\n[mfma] VERDICT: mfma_f32_16x16x32_bf16 lowering works on gfx950", flush=True)


if __name__ == "__main__":
    main()
