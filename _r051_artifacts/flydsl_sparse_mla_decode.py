"""r051 stage 1 round 1: FlyDSL backend SCAFFOLD for dpsk_v4 sparse MLA decode.

Module path target (docker cp): /sgl-workspace/aiter/aiter/ops/flydsl/kernels/sparse_mla_decode.py

Status: SCAFFOLD ONLY. Correctness via delegation to tilelang baseline.
Round 1 deliverable: dispatch path exists, sglang `backend="flydsl"` works,
output bit-equal to backend="tilelang" (because we're just calling it).

Round 2 deliverable: real FlyDSL kernel implementation replacing the delegation,
correctness max-diff < 1e-2 vs tilelang baseline.

Same signature as tilelang's `dpsk_v4_fp8_attention_fwd`:
    (q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata,
     num_splits, softmax_scale, causal, is_fp8_kvcache, indices, attn_sink,
     extra_k_cache, extra_indices_in_kvcache, topk_length, extra_topk_length)
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch

# Flag to switch between delegation (r051) and real FlyDSL kernel (r052+).
# When env SGLANG_FLYDSL_REAL=1, use the (future) FlyDSL kernel.
# Otherwise, delegate to tilelang reference for correctness baseline.
_USE_REAL_FLYDSL = os.environ.get("SGLANG_FLYDSL_REAL", "0") == "1"


def dpsk_v4_fp8_attention_fwd_flydsl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL backend entry point for DSv4 sparse MLA decode attention.

    For r051 stage 1 round 1: delegates to tilelang baseline for correctness.
    For r051 stage 1 round 2+: real FlyDSL kernel.

    Returns (output, lse) tuple, same as tilelang baseline.
    """
    if _USE_REAL_FLYDSL:
        # r052+: actual FlyDSL kernel call
        return _dpsk_v4_fp8_attention_fwd_flydsl_real(
            q=q, k_cache=k_cache, block_table=block_table,
            cache_seqlens=cache_seqlens, head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata, num_splits=num_splits,
            softmax_scale=softmax_scale, causal=causal,
            is_fp8_kvcache=is_fp8_kvcache, indices=indices, attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length, extra_topk_length=extra_topk_length,
        )
    # r051: delegate to tilelang (identity correctness)
    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        dpsk_v4_fp8_attention_fwd,
    )
    return dpsk_v4_fp8_attention_fwd(
        q=q, k_cache=k_cache, block_table=block_table,
        cache_seqlens=cache_seqlens, head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata, num_splits=num_splits,
        softmax_scale=softmax_scale, causal=causal,
        is_fp8_kvcache=is_fp8_kvcache, indices=indices, attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length, extra_topk_length=extra_topk_length,
    )


# =============================================================================
# r051 Round 6: FlyDSL K-gather kernel integrated into dispatch path
# =============================================================================
# - Runs validated FlyDSL weapon-1 kgather kernels (rounds 4+5) on the real
#   K cache + extra K cache. Output discarded — this is an EXERCISE to prove
#   the kernel runs under real server load (cuda graph, concurrent requests,
#   etc.) without crashing.
# - Math then delegates to tilelang (proven correct).
# - Subsequent rounds (7+) replace the tilelang math piece-by-piece with
#   FlyDSL mfma/softmax/dequant.

_KGATHER_KERNEL_CACHE = {}
_EXERCISE_LOGGED = {"once": False}


def _build_kgather_kernel(ROW_BYTES: int, BYTES_PER_LOAD: int, label: str):
    """Build a FlyDSL weapon-1 K-gather kernel. Implementation matches
    _r051_artifacts/test_kgather_with_scale.py (proven byte-exact)."""
    import sys as _sys
    _sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import buffer_ops, rocdl, arith
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.runtime.device import get_rocm_arch
    from tensor_shim import GTensor

    assert BYTES_PER_LOAD in (4, 16), BYTES_PER_LOAD
    assert ROW_BYTES % BYTES_PER_LOAD == 0
    NUM_THREADS = ROW_BYTES // BYTES_PER_LOAD

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"flydsl_kgs_{label}_r{ROW_BYTES}_b{BYTES_PER_LOAD}",
    )
    lds_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_row_offset + max(ROW_BYTES, 16)

    def _llvm_lds_ptr_ty():
        return ir.Type.parse("!llvm.ptr<3>")

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
        scr_i32 = GTensor(scratch_i8, dtype=T.i32, shape=(-1,))

        thread_lds_byte = tid * fx.Int32(BYTES_PER_LOAD) + fx.Int32(lds_row_offset)
        tlb = thread_lds_byte.value if hasattr(thread_lds_byte, "value") else thread_lds_byte
        lds_byte_i64 = arith.extui(T.i64, tlb)
        lds_ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), lds_byte_i64).result

        row_base = ro_.load(wgid, vec_size=1)
        rb = row_base.value if hasattr(row_base, "value") else row_base
        tir = (tid * fx.Int32(BYTES_PER_LOAD))
        tir_raw = tir.value if hasattr(tir, "value") else tir
        voffset = arith.addi(rb, tir_raw)

        rocdl.buffer_load_to_lds(
            rsrc=src_.rsrc, lds_ptr=lds_ptr, voffset=voffset,
            size_bytes=BYTES_PER_LOAD, soffset=0, offset=0,
        )
        rocdl.barrier()

        WORDS_PER_LOAD = BYTES_PER_LOAD // 4
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


def _exercise_kgather_on_real(k_cache: torch.Tensor, indices: torch.Tensor):
    """Run FlyDSL weapon-1 kgather on (k_cache, indices). Discards output.

    Validates that the FlyDSL kernel actually runs under real server load:
    real cuda graph, concurrent requests, real GPU memory pressure.
    """
    if k_cache is None or indices is None:
        return
    NB, BS_KV, H_KV, PACKED_W_FULL = k_cache.shape
    if PACKED_W_FULL != 584:
        return  # unfamiliar layout — skip
    PACKED_W_BYTES = 576
    SCALE_PADDED = 16
    SCALE_W_BYTES = 8

    BS, S_Q, TOPK = indices.shape
    if S_Q != 1:
        return

    # Build per-row byte offsets (packed region only — scale similar pattern,
    # exercised by separate kernel).
    idx_flat = indices.reshape(-1).to(torch.int32)
    idx_c = torch.clamp(idx_flat, min=0)
    block_id = idx_c // BS_KV
    in_block = idx_c % BS_KV
    block_stride = BS_KV * PACKED_W_FULL
    packed_row_off = (
        block_id.long() * block_stride + in_block.long() * PACKED_W_BYTES
    ).to(torch.int32).contiguous()

    GRID_X = packed_row_off.numel()
    k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
    packed_scratch = torch.zeros(GRID_X * PACKED_W_BYTES, dtype=torch.int8, device=k_cache.device)

    cache_key = (PACKED_W_BYTES, 16, "packed")
    if cache_key not in _KGATHER_KERNEL_CACHE:
        _KGATHER_KERNEL_CACHE[cache_key] = _build_kgather_kernel(
            PACKED_W_BYTES, 16, "packed"
        )
    launch = _KGATHER_KERNEL_CACHE[cache_key]
    launch(k_cache_i8, packed_row_off, packed_scratch, GRID_X)
    # Sync — we want this to be a real kernel exercise, not async noise.
    torch.cuda.synchronize()


def _dpsk_v4_fp8_attention_fwd_flydsl_real(**kwargs):
    """r051 Round 6 entry point.

    Strategy:
      1. EXERCISE the FlyDSL kgather kernel on the real k_cache + extra_k_cache
         (validates kernel runs under real server load).
      2. DELEGATE the actual attention math to tilelang (proven correct).

    Subsequent rounds replace step 2 piece-by-piece:
      r052+: dequant kernel in FlyDSL, output validated vs torch
      r053+: QK gemm in FlyDSL via mfma_f32_16x16x32_bf16
      r054+: online softmax + S@V in FlyDSL
      r055+: full kernel — replace tilelang call entirely

    Env vars:
      SGLANG_FLYDSL_EXERCISE=0  → skip exercise (just delegate; default)
      SGLANG_FLYDSL_EXERCISE=1  → run kgather exercise then delegate
      SGLANG_FLYDSL_DEBUG=1     → log first exercise failure
    """
    do_exercise = os.environ.get("SGLANG_FLYDSL_EXERCISE", "0") == "1"
    debug = os.environ.get("SGLANG_FLYDSL_DEBUG", "0") == "1"

    if do_exercise:
        try:
            _exercise_kgather_on_real(kwargs.get("k_cache"), kwargs.get("indices"))
            extra_kc = kwargs.get("extra_k_cache")
            extra_idx = kwargs.get("extra_indices_in_kvcache")
            if extra_kc is not None and extra_idx is not None:
                _exercise_kgather_on_real(extra_kc, extra_idx)
        except Exception as e:
            if debug and not _EXERCISE_LOGGED["once"]:
                import traceback
                print(f"[flydsl] kgather exercise failed: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                _EXERCISE_LOGGED["once"] = True

    # Delegate math to tilelang (proven correct under server load).
    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        dpsk_v4_fp8_attention_fwd,
    )
    return dpsk_v4_fp8_attention_fwd(**kwargs)
