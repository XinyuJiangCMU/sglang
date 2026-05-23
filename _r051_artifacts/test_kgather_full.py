"""r051 Milestone 3A: real FlyDSL K-gather kernel with byte-exact validation.

Builds on weapon-1 verification (rounds 2+3) to deliver an actual usable
primitive: a FlyDSL kernel that gathers BS*TOPK K vectors from the DSv4
K cache via weapon 1, writes them to an HBM scratch buffer, and is
byte-exact vs torch.gather reference.

Workgroup layout:
  grid_x = BS * TOPK  (one workgroup per (batch, k_idx))
  threads_per_wg = PACKED_W_BYTES // 16  (each thread does one 16-byte load)

Per workgroup:
  1. Read row_byte_offsets[wgid] (i32) to get this row's source byte offset
     in k_cache_flat.
  2. Each thread issues `buffer_load_dwordx4 ... lds` for its 16-byte slice.
  3. Barrier.
  4. Each thread reads its 16-byte LDS slice back as v16i8.
  5. Writes to k_scratch[wgid * PACKED_W_BYTES + tid*16 : +16].

Test:
  - Build with PACKED_W_BYTES = 576 (= 36 * 16, matches PACKED_W4=144 u32 = packed FP8 only).
  - Source = real DSv4 k_cache_i8 from pickle.
  - Reference = torch indexed gather of the same rows.
  - Pass = byte-exact match.
"""
import os
import sys
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, arith, vector as _fvector
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

# Use aiter's GTensor wrapper for typed HBM tensor access (load / store via
# buffer_ops). Available at /sgl-workspace/aiter/aiter/ops/flydsl/kernels/.
import sys as _sys
_sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_kgather_kernel(PACKED_W_BYTES: int):
    """Compile a FlyDSL K-gather kernel for the given row size."""
    BYTES_PER_LOAD = 16
    assert PACKED_W_BYTES % BYTES_PER_LOAD == 0
    NUM_THREADS = PACKED_W_BYTES // BYTES_PER_LOAD

    arch = get_hip_arch()
    allocator = SmemAllocator(
        None, arch=arch, global_sym_name=f"kgather_v1_p{PACKED_W_BYTES}"
    )
    lds_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_row_offset + PACKED_W_BYTES

    @flyc.kernel(name=f"flydsl_kgather_p{PACKED_W_BYTES}")
    def kgather_kernel(
        k_cache_i8: fx.Tensor,         # i8 flat, the K cache
        row_byte_offsets: fx.Tensor,   # i32 [grid_x], byte offset per row
        k_scratch_i8: fx.Tensor,       # i8 flat output [grid_x * PACKED_W_BYTES]
    ):
        tid = fx.thread_idx.x
        wgid = fx.block_idx.x

        # Wrap GTensors for typed HBM access. K cache + scratch are i32 views
        # to use dwordx4 stores cleanly (i8 stores hit LLVM split bug).
        ro_ = GTensor(row_byte_offsets, dtype=T.i32, shape=(-1,))
        kc_ = GTensor(k_cache_i8, dtype=T.i8, shape=(-1,))
        ks_ = GTensor(k_scratch_i8, dtype=T.i32, shape=(-1,))

        # Per-thread LDS byte offset within the row LDS buffer.
        thread_lds_byte = tid * fx.Int32(BYTES_PER_LOAD) + fx.Int32(lds_row_offset)
        tlb_raw = thread_lds_byte.value if hasattr(thread_lds_byte, "value") else thread_lds_byte
        lds_byte_i64 = arith.extui(T.i64, tlb_raw)
        lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_byte_i64).result

        # row_byte_offsets[wgid] via GTensor.load (returns scalar i32).
        row_base_i32 = ro_.load(wgid, vec_size=1)
        row_base_raw = row_base_i32.value if hasattr(row_base_i32, "value") else row_base_i32

        thread_in_row = tid * fx.Int32(BYTES_PER_LOAD)
        tir_raw = thread_in_row.value if hasattr(thread_in_row, "value") else thread_in_row
        voffset = arith.addi(row_base_raw, tir_raw)

        # WEAPON 1: HBM → LDS direct via kc_'s rsrc.
        rocdl.buffer_load_to_lds(
            rsrc=kc_.rsrc,
            lds_ptr=lds_ptr,
            voffset=voffset,
            size_bytes=BYTES_PER_LOAD,
            soffset=0,
            offset=0,
        )
        rocdl.barrier()

        # LDS → register → HBM scratch.
        # Read LDS as v4i32 (= 16 bytes = dwordx4) for clean store lowering.
        lds_view_i32 = SmemPtr(
            allocator.get_base(), lds_row_offset, T.i32, shape=(PACKED_W_BYTES // 4,)
        )
        lds_memref_i32 = lds_view_i32.get()
        v4i32_ty = T.vec(4, T.i32)
        lds_idx_words = (tid * fx.Int32(BYTES_PER_LOAD // 4)).value
        lds_idx_ix = arith.index_cast(T.index, lds_idx_words)
        v = _vector.load(v4i32_ty, lds_memref_i32, [lds_idx_ix])

        # Store to k_scratch (as i32 view): element index = byte_off / 4.
        out_word_off = (
            wgid * fx.Int32(PACKED_W_BYTES // 4) + tid * fx.Int32(BYTES_PER_LOAD // 4)
        )
        ks_.store(out_word_off, v, vec_size=4)

    @flyc.jit
    def launch(k_cache_i8, row_byte_offsets, k_scratch_i8, grid_x: fx.Int32):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kgather_kernel(k_cache_i8, row_byte_offsets, k_scratch_i8).launch(
            grid=(grid_x, 1, 1),
            block=(NUM_THREADS, 1, 1),
        )

    return launch


def main():
    PACKED_W_BYTES = 576  # 36 * 16, packed FP8 region only (scale region handled separately)

    print(f"[kgather] building kernel for PACKED_W_BYTES={PACKED_W_BYTES}...", flush=True)
    launch = build_kgather_kernel(PACKED_W_BYTES)
    print(f"[kgather] launcher built", flush=True)

    print("[kgather] loading real DSv4 K cache + indices...", flush=True)
    mb = torch.load("/tmp/microbench_bs192.pkl", map_location="cuda", weights_only=False)
    kw = mb["kwargs"]

    def _unwrap(d):
        if isinstance(d, dict) and "data" in d:
            return d["data"].to("cuda")
        return d

    k_cache = _unwrap(kw["k_cache"])   # (NB, BS_KV, 1, 584) fp8
    indices = _unwrap(kw["indices"])   # (bs, 1, topk) i32
    NB, BS_KV, H_KV, PACKED_W_FULL = k_cache.shape
    BS, S_Q, TOPK = indices.shape
    assert S_Q == 1
    print(f"[kgather] k_cache shape={tuple(k_cache.shape)}", flush=True)
    print(f"[kgather] indices shape={tuple(indices.shape)} TOPK={TOPK}", flush=True)

    # Use first 32 batches × all topk for the test (keep test fast).
    BS_TEST = min(32, BS)
    GRID_X = BS_TEST * TOPK
    print(f"[kgather] GRID_X={GRID_X} workgroups, {PACKED_W_BYTES // 16} threads/wg", flush=True)

    idx_flat = indices[:BS_TEST, 0, :].reshape(-1).to(torch.int32)
    # Clamp invalid indices to 0 (they would otherwise blow up the gather)
    idx_clamped = torch.clamp(idx_flat, min=0)
    block_id = idx_clamped // BS_KV
    in_block = idx_clamped % BS_KV
    # Row byte offset in k_cache_flat:
    #   block_id * (BS_KV * PACKED_W_FULL) + in_block * PACKED_W_FULL
    row_byte_offsets = (
        block_id.long() * (BS_KV * PACKED_W_FULL) + in_block.long() * PACKED_W_FULL
    ).to(torch.int32).contiguous()

    # K cache as flat i8
    k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
    # Output scratch
    k_scratch_i8 = torch.zeros(GRID_X * PACKED_W_BYTES, dtype=torch.int8, device="cuda")

    print(f"[kgather] k_cache_i8 size={k_cache_i8.numel():,} bytes "
          f"({k_cache_i8.numel()/1024/1024:.1f} MB)", flush=True)
    print(f"[kgather] k_scratch_i8 size={k_scratch_i8.numel():,} bytes", flush=True)

    try:
        print(f"[kgather] launching kernel...", flush=True)
        launch(k_cache_i8, row_byte_offsets, k_scratch_i8, GRID_X)
        torch.cuda.synchronize()
        print(f"[kgather] kernel ran OK", flush=True)
    except Exception as e:
        print(f"[kgather] LAUNCH FAIL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ---- Build torch reference ----
    print(f"[kgather] computing torch reference gather...", flush=True)
    # For each row (b, k), reference = k_cache[block_id, in_block, 0, :PACKED_W_BYTES]
    # as int8.
    kc_2d_i8 = k_cache.view(torch.int8).reshape(NB * BS_KV, PACKED_W_FULL)
    row_idx_2d = (block_id.long() * BS_KV + in_block.long()).to(torch.int64)
    ref_gather = kc_2d_i8[row_idx_2d, :PACKED_W_BYTES].contiguous()  # (GRID_X, PACKED_W_BYTES)
    got_gather = k_scratch_i8.view(GRID_X, PACKED_W_BYTES)

    # Compare
    diff = (got_gather.cpu() != ref_gather.cpu()).any(dim=1)
    n_mismatch_rows = diff.sum().item()
    total_rows = GRID_X
    print(f"[kgather] correctness: {total_rows - n_mismatch_rows} / {total_rows} rows byte-exact "
          f"({100*(total_rows-n_mismatch_rows)/total_rows:.2f}%)", flush=True)

    if n_mismatch_rows > 0:
        first_mm = torch.where(diff)[0][:3]
        for r in first_mm.tolist():
            got = got_gather[r].cpu()
            ref = ref_gather[r].cpu()
            n_diff = (got != ref).sum().item()
            print(f"    row {r}: {n_diff}/{PACKED_W_BYTES} byte diffs; "
                  f"got first 16 = {got[:16].tolist()}; "
                  f"ref first 16 = {ref[:16].tolist()}", flush=True)
        print(f"[kgather] VERDICT: MISMATCH — gather kernel logic wrong", flush=True)
        sys.exit(2)

    print(f"\n[kgather] VERDICT: PASS — FlyDSL K-gather byte-exact vs torch reference", flush=True)


if __name__ == "__main__":
    main()
