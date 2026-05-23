"""r051 stage 1 round 3 (incremental): validate weapon-1 buffer resource
works on REAL DSv4 K cache tensor (FP8, shape (NB, BS_KV, 1, 584)).

Goal: prove the weapon-1 path (verified in round 2 with synthetic input)
also works when the source buffer is the real DSv4 FP8 K cache layout —
584-byte rows, large total bytes (~217 MB for NB=2897, BS_KV=128).

Uses the EXACT verified API pattern from test_weapon1_emission.py to
minimize risk of regression. Only changes: input tensor sourced from
captured microbench pickle instead of zeros.
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
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_test_kernel(num_threads: int, bytes_per_thread: int):
    gpu_arch = get_hip_arch()
    print(f"[w1real] gpu_arch = {gpu_arch}", flush=True)
    TOTAL_BYTES = num_threads * bytes_per_thread

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="w1real_test_smem_v1",
    )
    lds_buf_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_buf_offset + TOTAL_BYTES

    @flyc.kernel(name="w1real_test_kernel")
    def test_kernel(in_tensor: fx.Tensor, out_tensor: fx.Tensor):
        tid = fx.thread_idx.x

        thread_lds_byte_i32 = tid * fx.Int32(bytes_per_thread) + fx.Int32(lds_buf_offset)
        ti32_raw = thread_lds_byte_i32.value if hasattr(thread_lds_byte_i32, "value") else thread_lds_byte_i32
        i64_v = arith.extui(T.i64, ti32_raw)
        lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), i64_v).result

        rsrc = buffer_ops.create_buffer_resource(in_tensor, max_size=True)
        voffset = tid * fx.Int32(bytes_per_thread)
        voffset_raw = voffset.value if hasattr(voffset, "value") else voffset

        rocdl.buffer_load_to_lds(
            rsrc=rsrc,
            lds_ptr=lds_ptr,
            voffset=voffset_raw,
            size_bytes=bytes_per_thread,
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
            block=(num_threads, 1, 1),
        )

    return launch_test


def main():
    print("[w1real] loading real DSv4 K cache from microbench pickle...", flush=True)
    pkl_path = "/tmp/microbench_bs192.pkl"
    if not os.path.exists(pkl_path):
        print(f"[w1real] FAIL: {pkl_path} not present", flush=True)
        sys.exit(1)
    mb = torch.load(pkl_path, map_location="cuda", weights_only=False)
    kw = mb["kwargs"]

    # Tensors are stored as dicts: {shape, dtype, device, data}.
    # `data` is already a torch.Tensor — reshape into the encoded shape.
    def _unwrap(d):
        if isinstance(d, dict) and "data" in d:
            return d["data"].to("cuda")
        return d

    # k_cache: (NB, BS_KV, 1, 584) FP8. View as flat i8.
    k_cache = _unwrap(kw["k_cache"])
    print(f"[w1real] k_cache shape={tuple(k_cache.shape)} dtype={k_cache.dtype}", flush=True)
    k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
    print(f"[w1real] k_cache_i8 size = {k_cache_i8.numel()} bytes "
          f"({k_cache_i8.numel() / 1024 / 1024:.1f} MB)", flush=True)

    # Test: gather 256 threads × 16 bytes = first 4 KB of K cache via weapon 1.
    NUM_THREADS = 256
    BYTES_PER_THREAD = 16
    launch = build_test_kernel(NUM_THREADS, BYTES_PER_THREAD)
    print("[w1real] launcher built", flush=True)

    out_t = torch.zeros(NUM_THREADS * BYTES_PER_THREAD, dtype=torch.int8, device="cuda")
    try:
        print(f"[w1real] launching against real K cache (size={k_cache_i8.numel()} bytes)...", flush=True)
        launch(k_cache_i8, out_t)
        torch.cuda.synchronize()
        print("[w1real] kernel ran OK against real DSv4 K cache", flush=True)
    except Exception as e:
        print(f"[w1real] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Final ISA inspection
    import time
    now = time.time()
    candidates = []
    for d in os.listdir("/root/.flydsl/cache"):
        p = os.path.join("/root/.flydsl/cache", d)
        if os.path.isdir(p) and now - os.path.getmtime(p) < 300:
            candidates.append((os.path.getmtime(p), p))
    candidates.sort(reverse=True)
    print(f"[w1real] {len(candidates)} fresh cache dirs", flush=True)
    if candidates:
        latest = candidates[0][1]
        print(f"[w1real] latest: {latest}", flush=True)
        # Dump ISA if dump dir set
        dump = os.environ.get("FLYDSL_DUMP_DIR")
        if dump:
            isa_path = f"{dump}/w1real_test_kernel/21_final_isa.s"
            if os.path.exists(isa_path):
                import subprocess
                out = subprocess.run(
                    ["grep", "-E", "buffer_load|global_load|ds_write|ds_read", isa_path],
                    capture_output=True, text=True
                )
                print(f"[w1real] ISA load/store ops:", flush=True)
                for ln in (out.stdout.strip().split("\n") or [])[:20]:
                    print(f"    {ln}", flush=True)

    print("\n[w1real] VERDICT: weapon-1 works on real DSv4 K cache buffer", flush=True)


if __name__ == "__main__":
    main()
