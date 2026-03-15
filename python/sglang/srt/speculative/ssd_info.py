"""Data structures for SSD (Speculative Speculative Decoding).

This module defines the input/output types used by the SSD worker
for draft and verify phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class SSDDraftInput(SpecInput):
    """Input passed to the draft model for speculation.

    In async mode, this replaces EagleDraftInput between decode steps.
    It tracks the recovery token (verified_id) and accept lengths
    needed for the next async speculate call.
    """

    # Hidden states from target model (for potential EAGLE-style conditioning)
    hidden_states: Optional[torch.Tensor] = None
    # Last accepted/verified token IDs per sequence: [B]
    # Used in async mode to track recovery tokens between steps
    verified_id: Optional[torch.Tensor] = None
    # Last accepted lengths per sequence: [B]
    # Used for tree cache key depth in async mode
    last_accepted_lens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SSD_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1

    def filter_batch(self, new_indices, has_been_filtered: bool = True):
        """Filter batch by keeping only entries at new_indices."""
        if self.verified_id is not None:
            if has_been_filtered:
                self.verified_id = self.verified_id[: len(new_indices)]
            else:
                self.verified_id = self.verified_id[new_indices]
        if self.last_accepted_lens is not None:
            if has_been_filtered:
                self.last_accepted_lens = self.last_accepted_lens[: len(new_indices)]
            else:
                self.last_accepted_lens = self.last_accepted_lens[new_indices]

    def merge_batch(self, other: "SSDDraftInput"):
        """Merge another SSDDraftInput into this one."""
        if self.verified_id is None:
            self.verified_id = other.verified_id
            self.last_accepted_lens = other.last_accepted_lens
            return
        if other.verified_id is None:
            return
        self.verified_id = torch.cat([self.verified_id, other.verified_id])
        if self.last_accepted_lens is not None and other.last_accepted_lens is not None:
            self.last_accepted_lens = torch.cat(
                [self.last_accepted_lens, other.last_accepted_lens]
            )


@dataclass
class SSDVerifyInput(SpecInput):
    """Input for the target model verify phase.

    After the draft model produces K candidate tokens per sequence,
    we pack them and run one target forward to verify.
    """

    # Draft tokens to verify: [total_draft_tokens]
    draft_token: torch.Tensor = None
    # Draft logits (q distribution): [B, K, V]
    draft_logits: Optional[torch.Tensor] = None
    # Number of speculative steps (K)
    spec_steps: int = 0
    # Whether any drafts came from cache hits (async mode): [B]
    cache_hits: Optional[torch.Tensor] = None
    # Recovery token IDs for tree cache keying (async mode): [B]
    recovery_token_ids: Optional[torch.Tensor] = None
    # Accepted length from previous step for cache key depth: [B]
    last_accepted_lens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SSD_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # During verify, each request has K+1 tokens (K draft + 1 original)
        return self.spec_steps + 1, self.spec_steps + 1


@dataclass
class SSDVerifyOutput:
    """Output from the verify phase.

    Contains accepted token suffixes and recovery tokens for each sequence.
    """

    # Number of accepted tokens per sequence: [B]
    accept_length: torch.Tensor = None
    # Accepted token ids per sequence (variable length, padded): [B, max_accept]
    accepted_tokens: Optional[torch.Tensor] = None
    # Recovery token for each sequence (sampled from residual): [B]
    recovery_tokens: Optional[torch.Tensor] = None
