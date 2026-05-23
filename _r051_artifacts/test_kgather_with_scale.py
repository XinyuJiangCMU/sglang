"""r051 Round 5: extend K-gather to ALSO gather the per-row scale region.

DSv4 FP8 K cache layout per block (BS_KV tokens):
  [packed 0..PACKED_W bytes per token (FP8 data + bf16 tail)]
  [scale  0..SCALE_W  bytes per token (per-tile scale bytes)]
Block stride = BS_KV * (PACKED_W + SCALE_W) = BS_KV * 584 bytes.

Per token (1 row):
  - packed bytes:  row_base + 0      .. row_base + PACKED_W       (576 bytes)
  - scale  bytes:  row_base + PACKED_W .. row_base + PACKED_W+SCALE_W (8 bytes)

This round: validates we can also gather the SCALE region byte-exact.
Once both work, we have all the input data for FP8 dequant in subsequent
rounds.

Strategy: two kgather kernels.
  1. packed: NUM_THREADS=36, BYTES_PER_LOAD=16, ROW_BYTES=576
  2. scale:  NUM_THREADS=1,  BYTES_PER_LOAD=8,  ROW_BYTES=8

Both byte-exact vs torch reference.
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
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_kgather_kernel(ROW_BYTES: int, BYTES_PER_LOAD: int, label: str):
    """Generic per-row weapon-1 gather kernel.

    Supports BYTES_PER_LOAD ∈ {4, 8, 12, 16} (rocdl.buffer_load_to_lds widths).
    """
    assert BYTES_PER_LOAD in (4, 8, 12, 16), BYTES_PER_LOAD
    assert ROW_BYTES % BYTES_PER_LOAD == 0
    NUM_THREADS = ROW_BYTES // BYTES_PER_LOAD

    arch = get_hip_arch()
    allocator = SmemAllocator(
        None, arch=arch, global_sym_name=f"kgather_{label}_r{ROW_BYTES}_b{BYTES_PER_LOAD}_v1"
    )
    lds_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_row_offset + max(ROW_BYTES, 16)  # align LDS up

    @flyc.kernel(name=f"flydsl_kgather_{label}_r{ROW_BYTES}")
    def kgather_kernel(
        src_i8: fx.Tensor,
        row_byte_offsets: fx.Tensor,
        scratch_i8: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        wgid = fx.block_idx.x

        ro_ = GTensor(row_byte_offsets, dtype=T.i32, shape=(-1,))
        src_ = GTensor(src_i8, dtype=T.i8, shape=(-1,))
        # Scratch indexed in 4-byte words (i32) to dodge i8 store split bug.
        scr_i32 = GTensor(scratch_i8, dtype=T.i32, shape=(-1,))

        # LDS byte offset for this thread.
        thread_lds_byte = tid * fx.Int32(BYTES_PER_LOAD) + fx.Int32(lds_row_offset)
        tlb = thread_lds_byte.value if hasattr(thread_lds_byte, "value") else thread_lds_byte
        lds_byte_i64 = arith.extui(T.i64, tlb)
        lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_byte_i64).result

        # voffset = row_base[wgid] + tid * BYTES_PER_LOAD
        row_base = ro_.load(wgid, vec_size=1)
        rb = row_base.value if hasattr(row_base, "value") else row_base
        tir = (tid * fx.Int32(BYTES_PER_LOAD))
        tir_raw = tir.value if hasattr(tir, "value") else tir
        voffset = arith.addi(rb, tir_raw)

        # WEAPON 1
        rocdl.buffer_load_to_lds(
            rsrc=src_.rsrc, lds_ptr=lds_ptr, voffset=voffset,
            size_bytes=BYTES_PER_LOAD, soffset=0, offset=0,
        )
        rocdl.barrier()

        # LDS → reg → HBM (via i32 vector store).
        WORDS_PER_LOAD = BYTES_PER_LOAD // 4
        if WORDS_PER_LOAD >= 1:
            lds_view_i32 = SmemPtr(
                allocator.get_base(), lds_row_offset, T.i32,
                shape=(max(ROW_BYTES // 4, 4),),
            )
            lds_memref_i32 = lds_view_i32.get()
            vt = T.vec(WORDS_PER_LOAD, T.i32)
            lds_idx = (tid * fx.Int32(WORDS_PER_LOAD)).value
            lds_idx_ix = arith.index_cast(T.index, lds_idx)
            v = _vector.load(vt, lds_memref_i32, [lds_idx_ix])

            out_word_off = (
                wgid * fx.Int32(ROW_BYTES // 4) + tid * fx.Int32(WORDS_PER_LOAD)
            )
            scr_i32.store(out_word_off, v, vec_size=WORDS_PER_LOAD)

    @flyc.jit
    def launch(src_i8, row_byte_offsets, scratch_i8, grid_x: fx.Int32):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kgather_kernel(src_i8, row_byte_offsets, scratch_i8).launch(
            grid=(grid_x, 1, 1),
            block=(NUM_THREADS, 1, 1),
        )

    return launch


def main():
    PACKED_W_BYTES = 576
    SCALE_W_BYTES = 8
    PACKED_W_FULL = PACKED_W_BYTES + SCALE_W_BYTES  # 584

    # Scale is 8 bytes per row, but LLVM AMDGPU backend fails to lower
    # buffer_load_to_lds with size_bytes=8 (LLVM "expand operand" bug).
    # Workaround: load 16 bytes per scale row (8 valid + 8 overrun that we
    # ignore in torch-ref comparison). Scratch row stride = 16 bytes.
    SCALE_ROW_BYTES_PADDED = 16
    print(f"[kgs] building packed kernel (ROW={PACKED_W_BYTES}, LOAD=16)...", flush=True)
    launch_packed = build_kgather_kernel(PACKED_W_BYTES, 16, "packed")
    print(f"[kgs] building scale kernel  (ROW={SCALE_ROW_BYTES_PADDED}, LOAD=16)...", flush=True)
    launch_scale = build_kgather_kernel(SCALE_ROW_BYTES_PADDED, 16, "scale")
    print("[kgs] launchers built", flush=True)

    mb = torch.load("/tmp/microbench_bs192.pkl", map_location="cuda", weights_only=False)
    kw = mb["kwargs"]
    def _u(d): return d["data"].to("cuda") if isinstance(d, dict) else d
    k_cache = _u(kw["k_cache"])
    indices = _u(kw["indices"])
    NB, BS_KV, H_KV, PW = k_cache.shape
    assert PW == PACKED_W_FULL
    BS, S_Q, TOPK = indices.shape

    BS_TEST = min(32, BS)
    GRID_X = BS_TEST * TOPK
    print(f"[kgs] GRID_X={GRID_X}, BS_KV={BS_KV}", flush=True)

    idx_flat = indices[:BS_TEST, 0, :].reshape(-1).to(torch.int32)
    idx_c = torch.clamp(idx_flat, min=0)
    block_id = idx_c // BS_KV
    in_block = idx_c % BS_KV
    block_stride = BS_KV * PACKED_W_FULL  # bytes per block

    # Packed row byte offset
    packed_row_off = (
        block_id.long() * block_stride + in_block.long() * PACKED_W_BYTES
    ).to(torch.int32).contiguous()
    # Scale row byte offset:
    #   block_base + BS_KV * PACKED_W_BYTES + in_block * SCALE_W_BYTES
    scale_row_off = (
        block_id.long() * block_stride
        + BS_KV * PACKED_W_BYTES
        + in_block.long() * SCALE_W_BYTES
    ).to(torch.int32).contiguous()

    k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
    packed_scratch = torch.zeros(GRID_X * PACKED_W_BYTES, dtype=torch.int8, device="cuda")
    scale_scratch  = torch.zeros(GRID_X * SCALE_ROW_BYTES_PADDED, dtype=torch.int8, device="cuda")

    print(f"[kgs] launching packed...", flush=True)
    launch_packed(k_cache_i8, packed_row_off, packed_scratch, GRID_X)
    torch.cuda.synchronize()
    print(f"[kgs] launching scale...", flush=True)
    launch_scale(k_cache_i8, scale_row_off, scale_scratch, GRID_X)
    torch.cuda.synchronize()
    print(f"[kgs] both kernels ran OK", flush=True)

    # Torch reference for both regions
    kc_blocks = k_cache.view(torch.int8).reshape(NB, BS_KV * PACKED_W_FULL)
    # For each row r = (b, k): block=block_id[r], in_block=in_block[r]
    ref_packed = torch.empty(GRID_X, PACKED_W_BYTES, dtype=torch.int8, device="cuda")
    ref_scale  = torch.empty(GRID_X, SCALE_W_BYTES,  dtype=torch.int8, device="cuda")
    for r in range(GRID_X):
        b_id = block_id[r].item()
        ib = in_block[r].item()
        packed_off = ib * PACKED_W_BYTES
        scale_off = BS_KV * PACKED_W_BYTES + ib * SCALE_W_BYTES
        ref_packed[r] = kc_blocks[b_id, packed_off : packed_off + PACKED_W_BYTES]
        ref_scale[r]  = kc_blocks[b_id, scale_off  : scale_off  + SCALE_W_BYTES]

    got_packed = packed_scratch.view(GRID_X, PACKED_W_BYTES)
    # Scale: scratch has 16 bytes per row, but only the first 8 are valid scale bytes.
    got_scale  = scale_scratch.view(GRID_X, SCALE_ROW_BYTES_PADDED)[:, :SCALE_W_BYTES]

    p_diff = (got_packed != ref_packed).any(dim=1).sum().item()
    s_diff = (got_scale  != ref_scale).any(dim=1).sum().item()

    print(f"[kgs] packed: {GRID_X - p_diff}/{GRID_X} rows byte-exact "
          f"({100*(GRID_X-p_diff)/GRID_X:.2f}%)", flush=True)
    print(f"[kgs] scale:  {GRID_X - s_diff}/{GRID_X} rows byte-exact "
          f"({100*(GRID_X-s_diff)/GRID_X:.2f}%)", flush=True)

    if p_diff == 0 and s_diff == 0:
        print(f"\n[kgs] VERDICT: PASS — packed + scale gather both byte-exact", flush=True)
    else:
        print(f"\n[kgs] VERDICT: FAIL", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
