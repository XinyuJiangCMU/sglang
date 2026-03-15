"""Tree decode utilities for async SSD.

Implements the core tree operations:
  - make_glue_decode_input_ids: combine recovery + spec tokens for glue decode
  - get_forked_recovery_tokens: fork K+1 logits into MQ_LEN branches
  - compute_step_positions_and_slot_maps: precompute all K steps' KV positions
"""

import torch
from typing import List


def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,
    rec_tokens: torch.Tensor,
) -> torch.Tensor:
    """Combine recovery token + K spec tokens for glue decode.

    The "glue decode" runs the draft model on K+1 tokens (1 recovery + K spec)
    to get updated logits at each position for forking.

    Args:
        draft_tokens: [B, K] — previous speculated tokens
        rec_tokens: [B] — recovery token IDs

    Returns:
        [B*(K+1)] — flattened input IDs for glue decode
    """
    # [B, K+1] = [rec_token, spec_0, spec_1, ..., spec_{K-1}]
    out = torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1)
    return out.view(-1)


def get_forked_recovery_tokens(
    logits: torch.Tensor,
    cache_hits: torch.Tensor,
    returned_tokens: torch.Tensor,
    fan_out_list: List[int],
    fan_out_list_miss: List[int],
) -> torch.Tensor:
    """Fork K+1 logits into MQ_LEN branches by sampling top-F per position.

    For each of the K+1 positions in the glue decode output, we take
    the top fan_out[k] tokens (excluding the already-returned token).
    This creates the branches of the speculation tree.

    Args:
        logits: [B, K+1, V] — glue decode output logits
        cache_hits: [B] — which sequences had cache hits (int64, 0 or 1)
        returned_tokens: [B, K+1] — rec + spec tokens (to exclude from fork)
        fan_out_list: [K+1] — branching factor per position (cache hit)
        fan_out_list_miss: [K+1] — branching factor per position (cache miss)

    Returns:
        [B, MQ_LEN] — forked recovery token IDs
        where MQ_LEN = sum(fan_out_list)
    """
    B, Kp1, V = logits.shape
    device = logits.device

    # Clone logits to avoid mutating original
    logits = logits.clone()

    # Mask out returned tokens at positions 0..K-1 to avoid resampling
    # Position K (last spec position) doesn't need masking since we sample new from it
    if Kp1 > 1:
        # returned_tokens[:, 1:] are the spec tokens at positions 0..K-1
        # We mask them out at positions 0..K-1 in the logits
        mask_positions = returned_tokens[:, 1:]  # [B, K]
        logits[:, :Kp1 - 1, :].scatter_(
            dim=2,
            index=mask_positions.unsqueeze(2).to(torch.int64),
            value=float("-inf"),
        )

    # Compute top-k at max fan-out
    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]

    # Build per-batch per-position fan-out counts
    fan_hit = torch.tensor(fan_out_list, device=device)  # [K+1]
    fan_miss = torch.tensor(fan_out_list_miss, device=device)  # [K+1]
    ch_bool = cache_hits.bool().view(B, 1)  # [B, 1]
    counts = torch.where(
        ch_bool.expand(B, Kp1),
        fan_hit.unsqueeze(0).expand(B, Kp1),
        fan_miss.unsqueeze(0).expand(B, Kp1),
    )  # [B, K+1]

    # Create selection mask and extract
    arange_k = torch.arange(k_max, device=device).view(1, 1, k_max)  # [1, 1, k_max]
    mask = arange_k < counts.unsqueeze(2)  # [B, K+1, k_max]

    # Select and reshape: each sequence gets MQ_LEN tokens
    mq_len = sum(fan_out_list)
    result = topk_idx.masked_select(mask).view(B, mq_len)

    return result


def compute_step_positions_and_slot_maps(
    initial_positions: torch.Tensor,
    K: int,
    B: int,
    N: int,
    mq_len: int,
    block_tables: torch.Tensor,
    block_size: int,
    device: torch.device,
):
    """Precompute positions and slot maps for all K tree decode steps.

    Each depth adds MQ_LEN to the position offset, ensuring branches
    at different depths write to non-overlapping KV cache slots.

    Args:
        initial_positions: [N] — starting positions for each branch
        K: speculation depth
        B: batch size
        N: total branches (B * MQ_LEN)
        mq_len: branches per sequence
        block_tables: [B, max_blocks] — KV cache block allocations
        block_size: KV cache page size
        device: torch device

    Returns:
        step_positions: [K, N] — positions at each depth
        step_context_lens: [K, N] — context lengths at each depth
        step_slot_maps: [K, N] — KV cache slot mappings at each depth
    """
    # Position offsets: [K, 1] = [[0], [MQ_LEN], [2*MQ_LEN], ...]
    step_pos_offsets = (
        torch.arange(K, device=device, dtype=torch.int64).unsqueeze(1) * mq_len
    )

    # Step positions: [K, N]
    step_positions = initial_positions.unsqueeze(0) + step_pos_offsets

    # Context lengths: position + 1
    step_context_lens = step_positions + 1

    # Expand block tables from [B, M] to [N, M]
    b_flat = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(B, mq_len)
        .flatten()
    )  # [N]
    dbt_expanded = block_tables[b_flat]  # [N, M]

    # Compute slot maps for each step: [K, N]
    block_idx = step_positions // block_size  # [K, N]
    pos_in_block = step_positions % block_size  # [K, N]

    # Gather block IDs: dbt_expanded[branch_i, block_idx[k, branch_i]]
    arange_n = torch.arange(N, device=device)  # [N]
    step_slot_maps = torch.zeros_like(step_positions)
    for k in range(K):
        blk_ids = dbt_expanded[arange_n, block_idx[k].clamp(max=block_tables.shape[1] - 1)]
        step_slot_maps[k] = blk_ids * block_size + pos_in_block[k]

    return step_positions, step_context_lens, step_slot_maps


def build_fan_indices(
    fan_out_list: List[int],
    fan_out_list_miss: List[int],
    K: int,
    device: torch.device,
):
    """Build fan-out index tensors for cache population.

    These map each branch to its depth position in the tree.

    Args:
        fan_out_list: [K+1] branching factors (hit)
        fan_out_list_miss: [K+1] branching factors (miss)
        K: speculation depth
        device: torch device

    Returns:
        fan_idx_hit: [MQ_LEN] — depth index per branch (hit)
        fan_idx_miss: [MQ_LEN] — depth index per branch (miss)
    """
    fan_t = torch.tensor(fan_out_list, device=device, dtype=torch.int64)
    fan_t_miss = torch.tensor(fan_out_list_miss, device=device, dtype=torch.int64)

    fan_idx_hit = torch.arange(K + 1, device=device).repeat_interleave(fan_t)
    fan_idx_miss = torch.arange(K + 1, device=device).repeat_interleave(fan_t_miss)

    return fan_idx_hit, fan_idx_miss
