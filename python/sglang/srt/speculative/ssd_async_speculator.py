"""SSD Async Speculator — target-side interface for async SSD.

Sends NCCL commands to the draft process and receives speculation results.
Used by SSDWorker when speculative_ssd_async=True.

Protocol:
  prefill():    cmd=1 → meta → fused(input_ids, num_tokens, block_tables)
  speculate():  cmd=0 → meta → fused(cache_keys, num_tokens, block_tables, temps)
                         ← fused(cache_hits, tokens) ← logits
  shutdown():   cmd=2
"""

import logging
from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.speculative.ssd_nccl_utils import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPECULATE,
    _pg_recv,
    send_cmd,
    send_int64,
    send_meta,
    temps_to_int64,
)

logger = logging.getLogger(__name__)


class SSDAsyncSpeculator:
    """Target-side async speculator for SSD.

    Communicates with SSDDraftRunner via NCCL process group.
    Pre-allocates send/recv buffers for zero-copy handshakes.
    """

    def __init__(
        self,
        device: torch.device,
        async_pg: dist.ProcessGroup,
        draft_rank: int,
        spec_k: int,
        fan_out: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        max_blocks: int,
    ):
        self.device = device
        self.async_pg = async_pg
        self.draft_rank = draft_rank
        self.K = spec_k
        self.fan_out = fan_out
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.max_blocks = max_blocks

        # Pre-allocate handshake buffers (resized on batch size change)
        self._hs_B = 0
        self._alloc_handshake_bufs(1)

    def _alloc_handshake_bufs(self, B: int):
        """Pre-allocate send/recv buffers for batch size B."""
        self._hs_B = B
        d = self.device
        K = self.K

        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self._block_tables_buf = torch.full(
            (B, self.max_blocks), -1, dtype=torch.int32, device=d
        )
        self._fused_response = torch.empty(
            B + B * K, dtype=torch.int64, device=d
        )
        self._logits_q = torch.empty(
            B, K, self.vocab_size, dtype=self.draft_dtype, device=d
        )

    def prefill(
        self,
        input_ids_list: List[torch.Tensor],
        num_tokens_list: List[int],
        block_tables_list: List[torch.Tensor],
    ):
        """Send prefill command to draft process.

        Args:
            input_ids_list: list of [seq_len] token ID tensors per sequence
            num_tokens_list: list of sequence lengths
            block_tables_list: list of block table tensors per sequence
        """
        B = len(input_ids_list)
        total_new_tokens = sum(t.shape[0] for t in input_ids_list)

        # Flatten input_ids
        input_ids = torch.cat(input_ids_list).to(torch.int64).to(self.device)
        num_tokens = torch.tensor(
            num_tokens_list, dtype=torch.int64, device=self.device
        )

        # Pad block tables to max_blocks
        max_blocks = self.max_blocks
        draft_block_table = torch.full(
            (B, max_blocks), -1, dtype=torch.int64, device=self.device
        )
        for i, bt in enumerate(block_tables_list):
            bt_len = min(len(bt), max_blocks)
            if bt_len > 0:
                draft_block_table[i, :bt_len] = bt[:bt_len].to(torch.int64)

        # Send: cmd → meta → fused payload
        send_cmd(self.async_pg, self.draft_rank, CMD_PREFILL, self.device)
        send_meta(
            self.async_pg,
            self.draft_rank,
            total_new_tokens,
            B,
            max_blocks,
            device=self.device,
        )
        send_int64(
            self.async_pg,
            self.draft_rank,
            input_ids,
            num_tokens,
            draft_block_table,
        )

        logger.debug(f"Sent prefill: B={B}, total_tokens={total_new_tokens}")

    def speculate(
        self,
        seq_ids: torch.Tensor,
        cache_key_depths: torch.Tensor,
        recovery_token_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        block_tables: torch.Tensor,
    ):
        """Send speculation request to draft, receive results.

        Args:
            seq_ids: [B] sequence IDs
            cache_key_depths: [B] depth indices for cache lookup
            recovery_token_ids: [B] recovery token IDs
            num_tokens: [B] current sequence lengths
            temperatures: [B] sampling temperatures
            block_tables: [B, max_blocks] KV cache block tables

        Returns:
            spec_tokens: [B, K] speculated token sequences
            logits_q: [B, K, V] draft model logits
            cache_hits: [B] which sequences had cache hits
        """
        B = seq_ids.shape[0]
        K = self.K

        # Resize buffers if batch size changed
        if B != self._hs_B:
            self._alloc_handshake_bufs(B)

        # Build cache keys: [B, 3] = (seq_id, depth, recovery_token_id)
        self._cache_keys[:B, 0] = seq_ids
        self._cache_keys[:B, 1] = cache_key_depths
        self._cache_keys[:B, 2] = recovery_token_ids

        self._num_tokens_buf[:B] = num_tokens
        self._temps_buf[:B] = temperatures

        # Copy block tables
        bt_cols = min(block_tables.shape[1], self.max_blocks)
        self._block_tables_buf[:B, :bt_cols] = block_tables[:, :bt_cols]
        self._block_tables_buf[:B, bt_cols:] = -1

        # Send: cmd → meta → fused payload
        send_cmd(self.async_pg, self.draft_rank, CMD_SPECULATE, self.device)
        send_meta(
            self.async_pg,
            self.draft_rank,
            B,
            K,
            self.fan_out,
            device=self.device,
        )
        temps_i64 = temps_to_int64(self._temps_buf[:B])
        send_int64(
            self.async_pg,
            self.draft_rank,
            self._cache_keys[:B],
            self._num_tokens_buf[:B],
            self._block_tables_buf[:B].to(torch.int64),
            temps_i64,
        )

        # Receive response: fused [cache_hits(B), tokens(B*K)] + logits
        fused_response = self._fused_response[: B + B * K]
        _pg_recv(self.async_pg, fused_response, self.draft_rank)
        cache_hits = fused_response[:B]
        spec_tokens = fused_response[B:].view(B, K)

        logits_q = self._logits_q[:B]
        _pg_recv(self.async_pg, logits_q, self.draft_rank)

        logger.debug(
            f"Spec response: B={B}, hits={cache_hits.sum().item()}/{B}"
        )

        return spec_tokens, logits_q, cache_hits

    def shutdown(self):
        """Send exit command to draft process."""
        send_cmd(self.async_pg, self.draft_rank, CMD_EXIT, self.device)
        logger.info("Sent shutdown to draft process")
