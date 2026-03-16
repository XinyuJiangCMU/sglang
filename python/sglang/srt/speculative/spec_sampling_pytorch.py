"""
Pure PyTorch implementations of speculative sampling functions.

These are portable fallbacks for platforms where the CUDA-specific sgl_kernel
ops (CUB + FlashInfer based) are not available, e.g. AMD HIP/ROCm.
"""

import torch


def top_k_renorm_prob(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Zero out probabilities below top-k and renormalize.

    Args:
        probs: (batch, vocab_size) probability distributions
        top_ks: (batch,) per-row top-k values
    Returns:
        Renormalized probabilities with only top-k entries kept.
    """
    vocab_size = probs.shape[-1]
    # Clamp top_ks to valid range; values >= vocab_size mean "keep all"
    top_ks_clamped = top_ks.clamp(min=1, max=vocab_size)
    max_k = top_ks_clamped.max().item()
    if max_k >= vocab_size:
        return probs
    # Get the k-th largest value per row as threshold
    topk_vals, _ = torch.topk(probs, k=int(max_k), dim=-1, sorted=True)
    # For each row, the threshold is the value at position top_ks[i]-1
    ks = top_ks_clamped.long() - 1  # (batch,)
    ks = ks.clamp(max=int(max_k) - 1)
    thresholds = topk_vals[torch.arange(probs.shape[0], device=probs.device), ks]
    thresholds = thresholds.unsqueeze(-1)  # (batch, 1)
    # Zero out entries below threshold
    filtered = probs.clone()
    filtered[probs < thresholds] = 0.0
    # Renormalize
    sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return filtered / sums


def top_p_renorm_prob(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Zero out probabilities below top-p (nucleus) threshold and renormalize.

    Args:
        probs: (batch, vocab_size) probability distributions
        top_ps: (batch,) per-row top-p values
    Returns:
        Renormalized probabilities with only nucleus entries kept.
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Create mask: keep tokens where cumsum <= top_p (plus the first token exceeding it)
    top_ps_expanded = top_ps.unsqueeze(-1)  # (batch, 1)
    # Shift cumsum right by 1 so that the token that pushes over the threshold is included
    cumsum_shifted = cumsum_probs - sorted_probs
    mask = cumsum_shifted < top_ps_expanded  # True = keep
    # Always keep at least the top token
    mask[:, 0] = True
    # Zero out filtered tokens
    sorted_probs[~mask] = 0.0
    # Scatter back to original order
    filtered = torch.zeros_like(probs)
    filtered.scatter_(1, sorted_indices, sorted_probs)
    # Renormalize
    sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return filtered / sums


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable, (total_draft_tokens,) int32
    accept_index: torch.Tensor,  # mutable, (bs, num_spec_steps) int32
    accept_token_num: torch.Tensor,  # mutable, (bs,) int32
    candidates: torch.Tensor,  # (bs, num_draft_tokens) int32
    retrive_index: torch.Tensor,  # (bs, num_draft_tokens) int32
    retrive_next_token: torch.Tensor,  # (bs, num_draft_tokens) int32
    retrive_next_sibling: torch.Tensor,  # (bs, num_draft_tokens) int32
    uniform_samples: torch.Tensor,  # (bs, num_draft_tokens) float32
    uniform_samples_for_final_sampling: torch.Tensor,  # (bs,) float32
    target_probs: torch.Tensor,  # (bs, num_draft_tokens, vocab) float32
    draft_probs: torch.Tensor,  # (bs, num_draft_tokens, vocab) float32
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """PyTorch implementation of tree speculative sampling (target-only).

    Mirrors the CUDA kernel TreeSpeculativeSamplingTargetOnly:
    1. Walk the draft tree, accepting/rejecting tokens via rejection sampling.
    2. Sample a bonus/correction token from relu(target_probs - draft_probs).

    All output tensors (predicts, accept_index, accept_token_num) are modified in-place.
    """
    bs = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_spec_tokens = accept_index.shape[1]
    d = target_probs.shape[-1]
    threshold_acc_capped = max(threshold_acc, 1e-9)
    dev = target_probs.device

    # Phase 1: Batch-transfer small tensors to CPU (single sync, zero-overhead indexing).
    candidates_l = candidates.tolist()
    retrive_index_l = retrive_index.tolist()
    retrive_next_token_l = retrive_next_token.tolist()
    retrive_next_sibling_l = retrive_next_sibling.tolist()
    uniform_samples_l = uniform_samples.tolist()
    coins_final_l = uniform_samples_for_final_sampling.tolist()

    # Pre-gather target probs for candidate tokens on GPU (avoids large tensor transfer).
    # gathered[bx, pos, idx] = target_probs[bx, pos, candidates[bx, idx]]
    cand_exp = candidates.unsqueeze(1).expand(-1, num_draft_tokens, -1).long()
    gathered_l = target_probs.gather(2, cand_exp).tolist()

    # Snapshot accept_index for CPU-side modification.
    accept_index_l = accept_index.tolist()

    # Phase 2: Tree walk entirely on CPU (no GPU syncs).
    predict_updates = []    # (flat_index, token_id) for accepted tokens
    draft_corrections = []  # (bx, pos, token_id) for rejection correction
    final_info = []         # (bx, cur_prob_idx, last_accepted_retrive_idx)
    accept_nums = []

    for bx in range(bs):
        prob_acc = 0.0
        cur_prob_idx = 0
        coin = uniform_samples_l[bx][0]
        last_accepted_retrive_idx = retrive_index_l[bx][0]
        accept_index_l[bx][0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for j in range(1, num_spec_tokens):
            cur_index = retrive_next_token_l[bx][cur_index]
            while cur_index != -1:
                draft_index = retrive_index_l[bx][cur_index]
                draft_token_id = candidates_l[bx][cur_index]
                target_prob_single = gathered_l[bx][cur_prob_idx][cur_index]
                prob_acc += target_prob_single

                if (
                    coin <= prob_acc / threshold_acc_capped
                    or target_prob_single >= threshold_single
                ):
                    # Accept token
                    prob_acc = 0.0
                    cur_prob_idx = cur_index
                    coin = uniform_samples_l[bx][cur_index]
                    predict_updates.append(
                        (last_accepted_retrive_idx, draft_token_id)
                    )
                    num_accepted += 1
                    accept_index_l[bx][num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    # Reject: track correction for final sampling
                    draft_corrections.append(
                        (bx, cur_prob_idx, draft_token_id)
                    )
                    cur_index = retrive_next_sibling_l[bx][cur_index]

            if cur_index == -1:
                break

        accept_nums.append(num_accepted)
        final_info.append((bx, cur_prob_idx, last_accepted_retrive_idx))

    # Phase 3: Batch GPU operations (minimal syncs).

    # Apply rejection corrections to draft_probs.
    if draft_corrections:
        cb, cp, ct = zip(*draft_corrections)
        draft_probs[cb, cp, ct] = target_probs[cb, cp, ct]

    # Write accepted predictions.
    if predict_updates:
        pi, pv = zip(*predict_updates)
        predicts[list(pi)] = torch.tensor(
            pv, dtype=predicts.dtype, device=dev
        )

    # Vectorized final sampling on GPU for all batch elements.
    bx_idx = torch.tensor([f[0] for f in final_info], device=dev, dtype=torch.long)
    pos_idx = torch.tensor([f[1] for f in final_info], device=dev, dtype=torch.long)
    last_idx = [f[2] for f in final_info]

    q_batch = target_probs[bx_idx, pos_idx]    # (bs, vocab)
    p_batch = draft_probs[bx_idx, pos_idx]     # (bs, vocab)
    relu_diff = torch.relu(q_batch - p_batch)  # (bs, vocab)
    sums = relu_diff.sum(dim=-1)               # (bs,)

    coins_t = torch.tensor(coins_final_l, dtype=torch.float32, device=dev)
    targets = coins_t * sums
    cumsums = torch.cumsum(relu_diff, dim=-1)
    sampled = torch.searchsorted(
        cumsums, targets.unsqueeze(-1), right=True
    ).squeeze(-1)
    sampled = sampled.clamp(max=d - 1)
    sampled[sums <= 0] = d - 1

    predicts[last_idx] = sampled.to(predicts.dtype)

    # Write back accept_index and accept_token_num.
    accept_index.copy_(
        torch.tensor(accept_index_l, dtype=accept_index.dtype, device=dev)
    )
    accept_token_num.copy_(
        torch.tensor(accept_nums, dtype=accept_token_num.dtype, device=dev)
    )
