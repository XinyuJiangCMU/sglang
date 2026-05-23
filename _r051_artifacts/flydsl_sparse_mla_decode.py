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


def _dpsk_v4_fp8_attention_fwd_flydsl_real(**kwargs):
    """r052+ entry point. NOT IMPLEMENTED in r051 stage 1 round 1.

    Sketch of what this will do (per r051 CONTEXT.md stage 1+2):

    Stage 1 (correctness via simplest FlyDSL primitives):
      1. Allocate Q_lds, K_packed_lds, K_scale_lds, KV_lds, K_tail_lds (SmemAllocator)
      2. Load Q (contiguous) via buffer_copy_gmem16_dwordx4 → Q_lds
      3. For each batch / span / inner_iter:
         a. Gather K_packed via per-row buffer_load (indirect addressing,
            page_idx_shared[bi_i] like tilelang does)
         b. Gather K_scale similarly
         c. Dequant FP8 → BF16 in registers (or use mfma_scale_f32_16x16x128_f8f6f4
            directly if weapon 5 maps cleanly)
         d. mfma_f32_16x16x32_bf16 for Q @ K^T → acc_s
         e. Online softmax with running m_i / sumexp
         f. mfma_f32_16x16x32_bf16 for S @ V → acc_o
      4. Write partial_O / partial_LSE to HBM
      5. Combine kernel separately

    Stage 2 (weapons 1+2+5):
      W1: replace buffer_load + lds_store_16b_xor16 with global_load_lds (direct
          HBM→LDS, 0 VGPR transit pressure)
      W2: double-buffer + software pipeline K_load (stage 0) → dequant (stage 1)
          → gemm (stage 2), using mfma_preshuffle_pipeline pattern
      W5: if FlyDSL's mfma_scale_f32_16x16x128_f8f6f4 maps cleanly, skip dequant
          and run FP8 mfma directly with scale.

    Stage 3 (optional): manual producer/consumer via LDS atomic flags +
      thread-id partitioning, if stage 2 doesn't close the gap to triton at bs≥256.
    """
    raise NotImplementedError(
        "r051 stage 1 round 1: real FlyDSL kernel not yet implemented; "
        "set SGLANG_FLYDSL_REAL=0 (default) to delegate to tilelang baseline."
    )
