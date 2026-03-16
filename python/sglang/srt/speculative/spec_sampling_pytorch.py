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

    for bx in range(bs):
        prob_acc = 0.0
        cur_prob_idx = 0  # index into the (num_draft_tokens,) dimension
        coin = uniform_samples[bx, 0].item()
        last_accepted_retrive_idx = retrive_index[bx, 0].item()
        accept_index[bx, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for j in range(1, num_spec_tokens):
            cur_index = retrive_next_token[bx, cur_index].item()
            while cur_index != -1:
                draft_index = retrive_index[bx, cur_index].item()
                draft_token_id = candidates[bx, cur_index].item()
                target_prob_single = target_probs[
                    bx, cur_prob_idx, draft_token_id
                ].item()
                prob_acc += target_prob_single

                if (
                    coin <= prob_acc / threshold_acc_capped
                    or target_prob_single >= threshold_single
                ):
                    # Accept token
                    prob_acc = 0.0
                    cur_prob_idx = cur_index
                    coin = uniform_samples[bx, cur_index].item()
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    # Reject: record draft_probs for final sampling correction
                    draft_probs[bx, cur_prob_idx, draft_token_id] = target_probs[
                        bx, cur_prob_idx, draft_token_id
                    ]
                    cur_index = retrive_next_sibling[bx, cur_index].item()

            if cur_index == -1:
                break

        accept_token_num[bx] = num_accepted

        # Final sampling from relu(target_probs - draft_probs)
        coin_final = uniform_samples_for_final_sampling[bx].item()
        q = target_probs[bx, cur_prob_idx]  # (vocab,)
        p = draft_probs[bx, cur_prob_idx]  # (vocab,)
        relu_q_minus_p = torch.relu(q - p)
        sum_val = relu_q_minus_p.sum().item()

        if sum_val > 0:
            # Sample proportional to relu(q - p)
            target = coin_final * sum_val
            cumsum = torch.cumsum(relu_q_minus_p, dim=0)
            # Find first index where cumsum > target (strict >), matching CUDA kernel
            sampled_id = torch.searchsorted(cumsum, target, right=True).clamp(max=d - 1).item()
        else:
            sampled_id = d - 1

        predicts[last_accepted_retrive_idx] = sampled_id
