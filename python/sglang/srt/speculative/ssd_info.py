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
    """Input passed to the draft model for speculation."""

    # Hidden states from target model (for potential EAGLE-style conditioning)
    hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SSD_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1


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
    # Whether any drafts came from cache hits
    cache_hits: Optional[torch.Tensor] = None

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
