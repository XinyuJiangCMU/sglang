"""r051 stage 1 round 2: verify weapon 1 emission with minimal FlyDSL kernel.

Goal: prove FlyDSL can emit `global_load_lds_b128` / `buffer_load_dwordx4 ... lds`
(AMD weapon 1: direct HBM → LDS bypassing VGPRs).

This is the FIRST aiter kernel to use `rocdl.buffer_load_to_lds` — searched
/sgl-workspace/aiter/aiter/ops/flydsl/ and found ZERO existing usages.
"""
import os
import sys
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, arith
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_test_kernel():
    gpu_arch = get_hip_arch()
    print(f"[test_weapon1] gpu_arch = {gpu_arch}", flush=True)
    NUM_THREADS = 256
    BYTES_PER_THREAD = 16
    TOTAL_BYTES = NUM_THREADS * BYTES_PER_THREAD

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="weapon1_test_smem_v1",
    )
    lds_buf_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_buf_offset + TOTAL_BYTES

    @flyc.kernel(name="weapon1_test_kernel")
    def test_kernel(in_tensor: fx.Tensor, out_tensor: fx.Tensor):
        tid = fx.thread_idx.x

        # Per-thread LDS byte offset: lds_buf_offset + tid * BYTES_PER_THREAD
        # built via FlyDSL Int32 arithmetic (matches chunk_gated_delta_h style).
        thread_lds_byte_i32 = tid * fx.Int32(BYTES_PER_THREAD) + fx.Int32(
            lds_buf_offset
        )
        # cast to i64 then IntToPtr to !llvm.ptr<3>.
        i32_raw = thread_lds_byte_i32.value if hasattr(
            thread_lds_byte_i32, "value"
        ) else thread_lds_byte_i32
        i64_v = arith.extui(T.i64, i32_raw)
        lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), i64_v).result

        # buffer resource for input
        rsrc = buffer_ops.create_buffer_resource(in_tensor, max_size=True)
        voffset = tid * fx.Int32(BYTES_PER_THREAD)
        voffset_raw = voffset.value if hasattr(voffset, "value") else voffset

        rocdl.buffer_load_to_lds(
            rsrc=rsrc,
            lds_ptr=lds_ptr,
            voffset=voffset_raw,
            size_bytes=16,
            soffset=0,
            offset=0,
        )
        rocdl.barrier()

    @flyc.jit
    def launch_test(in_tensor: fx.Tensor, out_tensor: fx.Tensor):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        test_kernel(in_tensor, out_tensor).launch(
            grid=(1, 1, 1),
            block=(NUM_THREADS, 1, 1),
        )

    return launch_test


def main():
    print("[test_weapon1] starting kernel build...", flush=True)
    try:
        launch_test = build_test_kernel()
        print(f"[test_weapon1] launcher built", flush=True)
    except Exception as e:
        print(f"[test_weapon1] BUILD FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    NUM_THREADS = 256
    total_bytes = NUM_THREADS * 16
    in_t = torch.zeros(total_bytes, dtype=torch.int8, device="cuda")
    out_t = torch.zeros(total_bytes, dtype=torch.int8, device="cuda")

    try:
        print("[test_weapon1] launching kernel...", flush=True)
        launch_test(in_t, out_t)
        torch.cuda.synchronize()
        print("[test_weapon1] kernel ran OK", flush=True)
    except Exception as e:
        print(f"[test_weapon1] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Find newly created cache entry (by mtime, last 5 min)
    print("\n[test_weapon1] finding our cache entry...", flush=True)
    import time
    now = time.time()
    candidates = []
    for base in ("/root/.flydsl/cache", "/root/.cache/flydsl"):
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if not os.path.isdir(p):
                continue
            mtime = os.path.getmtime(p)
            if now - mtime < 300:
                candidates.append((mtime, p))
    candidates.sort(reverse=True)
    print(f"[test_weapon1] {len(candidates)} cache dirs touched in last 5 min", flush=True)
    target = None
    for mt, p in candidates[:10]:
        print(f"  age={int(now-mt)}s  {p}", flush=True)
        if "launch_test" in p or "weapon1" in p:
            target = p
            break

    if target:
        print(f"\n[test_weapon1] inspecting {target}", flush=True)
        for f in os.listdir(target):
            print(f"  - {f} ({os.path.getsize(os.path.join(target, f))} bytes)", flush=True)


if __name__ == "__main__":
    main()
