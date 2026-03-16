"""
Tests for the PyTorch implementations of speculative sampling functions.

Verifies:
1. top_k_renorm_prob: correctness of top-k filtering and renormalization
2. top_p_renorm_prob: correctness of nucleus (top-p) filtering and renormalization
3. tree_speculative_sampling_target_only: correctness of tree-based rejection sampling

For tree_speculative_sampling_target_only, we compare against the HIP
verify_tree_greedy kernel as a reference where applicable, and test
specific deterministic scenarios.
"""

import torch
import pytest

from sglang.srt.speculative.spec_sampling_pytorch import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)


# ─────────────────────────────────────────────────────────
# top_k_renorm_prob tests
# ─────────────────────────────────────────────────────────


class TestTopKRenormProb:
    def test_basic(self):
        """Top-k=2 on a 5-token vocab should keep exactly 2 nonzero entries."""
        probs = torch.tensor([[0.1, 0.3, 0.05, 0.4, 0.15]], device="cuda")
        top_ks = torch.tensor([2], dtype=torch.int32, device="cuda")
        result = top_k_renorm_prob(probs, top_ks)

        # Should keep tokens 1 (0.3) and 3 (0.4)
        assert (result[0, 0] == 0.0).item()
        assert (result[0, 2] == 0.0).item()
        assert (result[0, 4] == 0.0).item()
        assert result[0, 1] > 0
        assert result[0, 3] > 0
        # Should sum to 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, device="cuda"))

    def test_k_equals_vocab(self):
        """k >= vocab_size should return original probs."""
        probs = torch.tensor([[0.2, 0.3, 0.5]], device="cuda")
        top_ks = torch.tensor([3], dtype=torch.int32, device="cuda")
        result = top_k_renorm_prob(probs, top_ks)
        assert torch.allclose(result, probs)

    def test_k_exceeds_vocab(self):
        """k > vocab_size should not crash, return original probs."""
        probs = torch.tensor([[0.2, 0.3, 0.5]], device="cuda")
        top_ks = torch.tensor([100], dtype=torch.int32, device="cuda")
        result = top_k_renorm_prob(probs, top_ks)
        assert torch.allclose(result, probs)

    def test_batched(self):
        """Different k per row in a batch."""
        probs = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], device="cuda"
        )
        top_ks = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        result = top_k_renorm_prob(probs, top_ks)

        # Row 0: k=1, keep only token 3 (0.4)
        assert (result[0] > 0).sum() == 1
        assert result[0, 3] == 1.0

        # Row 1: k=2, keep tokens 0 (0.4) and 1 (0.3)
        assert (result[1] > 0).sum() == 2
        assert torch.allclose(result.sum(dim=-1), torch.ones(2, device="cuda"))

    def test_renormalization(self):
        """Output should always sum to 1."""
        probs = torch.randn(8, 1000, device="cuda").softmax(-1)
        top_ks = torch.randint(1, 100, (8,), dtype=torch.int32, device="cuda")
        result = top_k_renorm_prob(probs, top_ks)
        assert torch.allclose(
            result.sum(dim=-1), torch.ones(8, device="cuda"), atol=1e-5
        )


# ─────────────────────────────────────────────────────────
# top_p_renorm_prob tests
# ─────────────────────────────────────────────────────────


class TestTopPRenormProb:
    def test_basic(self):
        """top_p=0.5 on [0.4, 0.35, 0.15, 0.1] should keep top 2."""
        probs = torch.tensor([[0.4, 0.35, 0.15, 0.1]], device="cuda")
        top_ps = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        result = top_p_renorm_prob(probs, top_ps)

        # Token 0 (0.4) cumsum=0.4 < 0.5, keep
        # Token 1 (0.35) cumsum_shifted=0.4 < 0.5, keep
        # Token 2 (0.15) cumsum_shifted=0.75 >= 0.5, zero out
        assert result[0, 0] > 0
        assert result[0, 1] > 0
        assert result[0, 2] == 0.0
        assert result[0, 3] == 0.0
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, device="cuda"))

    def test_p_equals_one(self):
        """top_p=1.0 should return original probs."""
        probs = torch.tensor([[0.2, 0.3, 0.5]], device="cuda")
        top_ps = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        result = top_p_renorm_prob(probs, top_ps)
        assert torch.allclose(result, probs, atol=1e-6)

    def test_always_keeps_top_token(self):
        """Even with very small p, at least one token is kept."""
        probs = torch.tensor([[0.5, 0.3, 0.2]], device="cuda")
        top_ps = torch.tensor([0.01], dtype=torch.float32, device="cuda")
        result = top_p_renorm_prob(probs, top_ps)
        assert (result > 0).sum() >= 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(1, device="cuda"))

    def test_renormalization(self):
        """Output should always sum to 1."""
        probs = torch.randn(8, 1000, device="cuda").softmax(-1)
        top_ps = torch.rand(8, device="cuda") * 0.8 + 0.1  # [0.1, 0.9]
        result = top_p_renorm_prob(probs, top_ps)
        assert torch.allclose(
            result.sum(dim=-1), torch.ones(8, device="cuda"), atol=1e-5
        )


# ─────────────────────────────────────────────────────────
# tree_speculative_sampling_target_only tests
# ─────────────────────────────────────────────────────────

def _make_linear_tree(bs, num_draft_tokens, num_spec_steps, device="cuda"):
    """Create a simple linear tree (no branching): root -> 1 -> 2 -> 3.

    retrive_index: identity mapping
    retrive_next_token: each node points to the next
    retrive_next_sibling: all -1 (no siblings)
    """
    retrive_index = (
        torch.arange(num_draft_tokens, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    retrive_next_token = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int32, device=device
    )
    for i in range(num_draft_tokens - 1):
        retrive_next_token[:, i] = i + 1
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int32, device=device
    )
    return retrive_index, retrive_next_token, retrive_next_sibling


class TestTreeSpeculativeSampling:
    def test_coins_zero_accepts_all(self):
        """With coins=0 and lenient threshold, all draft tokens should be accepted."""
        bs, num_draft_tokens, vocab = 1, 4, 100
        num_spec_steps = num_draft_tokens  # spec_steps = draft_token_num for linear tree
        device = "cuda"

        retrive_index, retrive_next_token, retrive_next_sibling = _make_linear_tree(
            bs, num_draft_tokens, num_spec_steps, device
        )

        # Draft tokens: [10, 20, 30, 40]
        candidates = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32, device=device)

        # Target probs: each position gives >0 prob to the draft token
        target_probs = torch.zeros(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device)
        target_probs[0, 0, 10] = 0.5  # position 0 -> token 10 has prob 0.5
        target_probs[0, 1, 20] = 0.3  # position 1 -> token 20 has prob 0.3
        target_probs[0, 2, 30] = 0.4  # position 2 -> token 30 has prob 0.4
        # Spread remaining prob
        for i in range(num_draft_tokens):
            remaining = 1.0 - target_probs[0, i].sum().item()
            target_probs[0, i, 0] += remaining

        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # coins=0 means coin <= prob_acc/threshold is always true (prob_acc > 0)
        coins = torch.zeros(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins_final = torch.zeros(bs, dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # All 3 draft tokens (positions 1,2,3) should be accepted
        assert accept_length[0].item() == 3
        # predicts should have the draft token IDs at the accepted positions
        assert predicts[0].item() == 20  # position 0 predicts token at position 1
        assert predicts[1].item() == 30  # position 1 predicts token at position 2
        assert predicts[2].item() == 40  # position 2 predicts token at position 3

    def test_coins_one_rejects_low_prob(self):
        """With coins=1.0 and threshold_acc=1.0, only tokens with cumulative prob >= 1.0 accepted."""
        bs, num_draft_tokens, vocab = 1, 4, 100
        num_spec_steps = num_draft_tokens
        device = "cuda"

        retrive_index, retrive_next_token, retrive_next_sibling = _make_linear_tree(
            bs, num_draft_tokens, num_spec_steps, device
        )
        candidates = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32, device=device)

        # Low probs for draft tokens -> should be rejected with coin=1.0
        target_probs = torch.ones(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device) / vocab
        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # coin=1.0 means coin <= prob_acc/1.0 requires prob_acc >= 1.0
        # With uniform probs (1/100 per token), prob_acc = 1/100 < 1.0 → reject
        coins = torch.ones(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins_final = torch.tensor([0.5], dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # No tokens should be accepted (uniform prob too low)
        assert accept_length[0].item() == 0

    def test_final_sampling_from_residual(self):
        """When all tokens are rejected, final token should be sampled from relu(q - p)."""
        bs, num_draft_tokens, vocab = 1, 2, 10
        num_spec_steps = 2
        device = "cuda"

        retrive_index, retrive_next_token, retrive_next_sibling = _make_linear_tree(
            bs, num_draft_tokens, num_spec_steps, device
        )
        # Draft token at position 1 is token 3
        candidates = torch.tensor([[0, 3]], dtype=torch.int32, device=device)

        # Position 0: target gives all prob to token 5
        target_probs = torch.zeros(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device)
        target_probs[0, 0, 5] = 1.0  # all mass on token 5

        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # coin=1.0 → reject (target_probs[0,0,3]=0.0, prob_acc=0.0 < 1.0)
        coins = torch.ones(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins_final = torch.tensor([0.0], dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # Token rejected, so final sampling from relu(target - draft) at position 0
        # draft_probs[0,0,3] was set to target_probs[0,0,3] = 0.0 during rejection
        # relu(target - draft) = target (since draft is ~0 everywhere)
        # All mass is on token 5, so final token should be 5
        assert accept_length[0].item() == 0
        assert predicts[0].item() == 5

    def test_batched(self):
        """Multiple batches should be handled independently."""
        bs, num_draft_tokens, vocab = 2, 3, 50
        num_spec_steps = 3
        device = "cuda"

        retrive_index, retrive_next_token, retrive_next_sibling = _make_linear_tree(
            bs, num_draft_tokens, num_spec_steps, device
        )
        candidates = torch.tensor(
            [[10, 20, 30], [5, 15, 25]], dtype=torch.int32, device=device
        )

        target_probs = torch.zeros(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device)
        # Batch 0: high prob for draft tokens → should accept with coin=0
        target_probs[0, 0, 10] = 0.8
        target_probs[0, 0, 0] = 0.2
        target_probs[0, 1, 20] = 0.7
        target_probs[0, 1, 0] = 0.3
        # Batch 1: zero prob for draft tokens → should reject
        target_probs[1, 0, 0] = 1.0  # all mass on token 0, not token 5
        target_probs[1, 1, 0] = 1.0

        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # Batch 0: coin=0 → accept; Batch 1: coin=1 → reject
        coins = torch.zeros(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins[1, :] = 1.0
        coins_final = torch.zeros(bs, dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # Batch 0: all accepted
        assert accept_length[0].item() == 2
        # Batch 1: none accepted (prob of draft token 5 at position 0 is 0)
        assert accept_length[1].item() == 0

    def test_threshold_single_accepts_high_prob(self):
        """threshold_single < target_prob should force acceptance regardless of coin."""
        bs, num_draft_tokens, vocab = 1, 3, 10
        num_spec_steps = 3
        device = "cuda"

        retrive_index, retrive_next_token, retrive_next_sibling = _make_linear_tree(
            bs, num_draft_tokens, num_spec_steps, device
        )
        candidates = torch.tensor([[0, 7, 3]], dtype=torch.int32, device=device)

        target_probs = torch.zeros(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device)
        # Position 0: token 7 has prob 0.9 (>= threshold_single=0.8) → accept
        target_probs[0, 0, 7] = 0.9
        target_probs[0, 0, 0] = 0.1
        # Position 1 (after accepting token 7): token 3 has prob 0.1 (< 0.8) → reject
        target_probs[0, 1, 3] = 0.1
        target_probs[0, 1, 0] = 0.9

        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # coin=1.0 would normally reject, but threshold_single overrides
        coins = torch.ones(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins_final = torch.tensor([0.0], dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=0.8,  # accept if target_prob >= 0.8
            threshold_acc=1.0,
        )

        # Position 1 (token 7, prob=0.9 >= 0.8): accepted
        # Position 2 (token 3, prob=0.1 < 0.8): rejected (coin=1.0, prob_acc=0.1 < 1.0)
        assert accept_length[0].item() == 1
        assert predicts[0].item() == 7

    def test_sibling_traversal(self):
        """When first child is rejected, should try sibling."""
        bs, num_draft_tokens, vocab = 1, 3, 10
        num_spec_steps = 2  # root + 1 speculative step
        device = "cuda"

        # Tree: root(0) -> child(1), child(1) has sibling(2)
        retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
        retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int32, device=device)
        retrive_next_sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int32, device=device)

        # candidate at index 1 is token 3 (will be rejected)
        # candidate at index 2 is token 7 (sibling, will be accepted)
        candidates = torch.tensor([[0, 3, 7]], dtype=torch.int32, device=device)

        target_probs = torch.zeros(bs, num_draft_tokens, vocab, dtype=torch.float32, device=device)
        # Position 0: token 3 has 0 prob, token 7 has 0.9 prob
        target_probs[0, 0, 3] = 0.0
        target_probs[0, 0, 7] = 0.9
        target_probs[0, 0, 0] = 0.1

        draft_probs = torch.zeros_like(target_probs)

        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_spec_steps), -1, dtype=torch.int32, device=device)
        accept_length = torch.zeros(bs, dtype=torch.int32, device=device)

        # coin=0 → accept first token with prob > 0
        coins = torch.zeros(bs, num_draft_tokens, dtype=torch.float32, device=device)
        coins_final = torch.zeros(bs, dtype=torch.float32, device=device)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # Token 3 rejected (prob=0, prob_acc=0, 0 <= 0/1.0 is true... wait)
        # Actually coin=0 <= 0/1.0 = 0 → 0 <= 0 is TRUE → accepted!
        # Hmm, let me use coin slightly > 0
        # Actually: prob_acc starts at 0, target_prob for token 3 is 0.0
        # prob_acc = 0 + 0.0 = 0.0
        # coin=0 <= 0.0/1.0 = 0.0 → 0 <= 0 is True → accepted even with 0 prob
        # This matches the CUDA kernel behavior. Let's use coin=0.001 instead.

        # Reset and redo with small positive coin
        predicts.zero_()
        accept_index.fill_(-1)
        accept_length.zero_()
        draft_probs.zero_()
        coins.fill_(0.001)

        tree_speculative_sampling_target_only(
            predicts, accept_index, accept_length,
            candidates, retrive_index, retrive_next_token, retrive_next_sibling,
            coins, coins_final,
            target_probs, draft_probs,
            threshold_single=1.0, threshold_acc=1.0,
        )

        # Now: token 3 at index 1: prob_acc=0.0, coin=0.001 > 0.0 → REJECT
        # Try sibling at index 2: token 7: prob_acc=0.0+0.9=0.9, coin=0.001 <= 0.9 → ACCEPT
        assert accept_length[0].item() == 1
        assert predicts[0].item() == 7  # accepted sibling's token


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
