"""SSD Draft Runner — runs on a separate GPU as an independent process.

Handles three NCCL commands from the target:
  CMD_PREFILL (1):  Run draft model prefill for new sequences
  CMD_SPECULATE (0): Check tree cache → hit: return cached / miss: JIT speculate
  CMD_EXIT (2):     Shutdown

Tree Cache:
  Keys:   [N, 3] — (seq_id, k_idx, recovery_token_id)
  Tokens: [N, K] — speculated token sequences
  Logits: [N, K, V] — draft logits for each step
"""

import logging
import os
import time
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.speculative.ssd_nccl_utils import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPECULATE,
    concat_int64,
    int64_to_temps,
    recv_cmd,
    recv_int64,
    recv_meta,
    send_int64,
)

logger = logging.getLogger(__name__)


class SSDDraftRunner:
    """Draft model runner for async SSD.

    Runs in a separate process on its own GPU. Loads the draft model,
    maintains a tree cache, and responds to NCCL commands from target.
    """

    def __init__(
        self,
        draft_model_path: str,
        device: torch.device,
        gpu_id: int,
        async_pg: dist.ProcessGroup,
        target_rank: int,
        spec_k: int,
        fan_out: int,
        fan_out_list: list,
        fan_out_list_miss: list,
        vocab_size: int,
        dtype: torch.dtype,
        max_blocks: int,
        block_size: int,
        jit_speculate: bool = True,
    ):
        self.device = device
        self.gpu_id = gpu_id
        self.async_pg = async_pg
        self.target_rank = target_rank
        self.K = spec_k
        self.fan_out = fan_out
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.jit_speculate = jit_speculate

        self.mq_len = sum(fan_out_list)

        # Tree cache tensors
        self._reset_tree_cache()

        # Timing
        self._draft_step_times = []

        logger.info(
            f"SSDDraftRunner initialized: K={spec_k}, fan_out={fan_out}, "
            f"vocab={vocab_size}, jit={jit_speculate}, device={device}"
        )

    def _reset_tree_cache(self):
        """Reset tensor-backed tree cache to empty."""
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    # ------------------------------------------------------------------
    # Tree cache lookup
    # ------------------------------------------------------------------

    def _hit_cache(self, request_keys: torch.Tensor, B: int, K: int):
        """Check tree cache for hits. Returns (out_tokens, out_logits, cache_hits).

        Args:
            request_keys: [B, 3] — (seq_id, k_idx, recovery_token_id)
            B: batch size
            K: speculation depth

        Returns:
            out_tokens: [B, K] — speculated tokens (filled for hits)
            out_logits: [B, K, V] — draft logits (filled for hits)
            cache_hits: [B] — 1 if hit, 0 if miss
        """
        V = self.vocab_size
        # Init with random valid tokens for miss slots
        out_logits = torch.zeros(
            (B, K, V), dtype=self.dtype, device=self.device
        )
        out_tokens = torch.zeros((B, K), dtype=torch.int64, device=self.device)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership: [B, T, 3] → [B, T] → [B]
            eq = request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)
            match = torch.all(eq, dim=2)  # [B, T]
            cache_hits = match.any(dim=1).to(torch.int64)  # [B]

            if cache_hits.any():
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits.bool()
                out_tokens[sel] = self.tree_cache_tokens[idx[sel]]
                out_logits[sel] = self.tree_cache_logits[idx[sel]]

        return out_tokens, out_logits, cache_hits

    def _populate_tree_cache(
        self,
        seq_ids: torch.Tensor,
        recovery_tokens: torch.Tensor,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        cache_hits: torch.Tensor,
    ):
        """Populate tree cache with new speculation results.

        For each sequence, creates entries keyed by (seq_id, k_idx, recovery_token)
        where k_idx indexes depth in the speculation tree.

        Args:
            seq_ids: [B] sequence IDs
            recovery_tokens: [B] recovery token IDs
            tokens: [B, K] speculated tokens
            logits: [B, K, V] draft logits
            cache_hits: [B] which were cache hits
        """
        B = seq_ids.shape[0]
        K = self.K

        # For each sequence, create fan_out entries at each depth
        # Simple version: one entry per (seq_id, depth, recovery_token)
        entries_per_seq = K + 1  # one for each depth level
        N = B * entries_per_seq

        keys = torch.zeros((N, 3), dtype=torch.int64, device=self.device)
        cached_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
        cached_logits = torch.zeros(
            (N, K, self.vocab_size), dtype=self.dtype, device=self.device
        )

        for b in range(B):
            for k in range(entries_per_seq):
                idx = b * entries_per_seq + k
                keys[idx, 0] = seq_ids[b]
                keys[idx, 1] = k
                keys[idx, 2] = recovery_tokens[b]
                cached_tokens[idx] = tokens[b]
                cached_logits[idx] = logits[b]

        self.tree_cache_keys = keys
        self.tree_cache_tokens = cached_tokens
        self.tree_cache_logits = cached_logits

    # ------------------------------------------------------------------
    # JIT speculation (sequential K-step draft)
    # ------------------------------------------------------------------

    def jit_speculate(
        self,
        model,
        request_keys: torch.Tensor,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logits: torch.Tensor,
    ):
        """Run draft model forward K times for cache-miss sequences.

        This is the JIT (just-in-time) fallback when tree cache misses.
        Runs the draft model sequentially for K steps.

        Args:
            model: draft model runner
            request_keys: [B, 3] — cache keys (seq_id, k_idx, recovery_token_id)
            num_tokens: [B] — current sequence lengths
            temperatures: [B] — sampling temperatures
            draft_block_tables: [B, max_blocks] — KV cache block tables
            out_tokens: [B, K] — output buffer for tokens (modified in-place)
            out_logits: [B, K, V] — output buffer for logits (modified in-place)
        """
        input_ids = request_keys[:, -1]  # recovery_token_id
        B = input_ids.shape[0]
        positions = num_tokens - 1  # position of recovery token

        for i in range(self.K):
            # Run draft model forward
            logits = model.forward_draft_step(
                input_ids=input_ids,
                positions=positions,
                block_tables=draft_block_tables,
            )

            out_logits[:, i, :] = logits

            # Sample next tokens
            if (temperatures == 0).all():
                next_tokens = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperatures.unsqueeze(-1).clamp(min=1e-6), dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            out_tokens[:, i] = next_tokens

            # Update for next step
            input_ids = next_tokens
            positions = positions + 1

    # ------------------------------------------------------------------
    # NCCL command handlers
    # ------------------------------------------------------------------

    def _handle_prefill(self):
        """Handle CMD_PREFILL: receive prefill data from target."""
        # Receive metadata: [total_new_tokens, batch_size, max_blocks]
        meta = recv_meta(
            self.async_pg, src=self.target_rank, n=3, device=self.device
        )
        total_new_tokens, batch_size, max_blocks = meta

        # Receive fused payload: input_ids + num_tokens + block_tables
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = recv_int64(
            self.async_pg,
            src=self.target_rank,
            total_length=fused_total,
            device=self.device,
        )

        off = 0
        input_ids = fused[off : off + total_new_tokens]
        off += total_new_tokens
        num_tokens = fused[off : off + batch_size]
        off += batch_size
        draft_block_tables = (
            fused[off : off + batch_size * max_blocks]
            .view(batch_size, max_blocks)
            .to(torch.int32)
        )

        logger.debug(
            f"Draft prefill: B={batch_size}, total_tokens={total_new_tokens}"
        )

        # TODO: Actually run draft model prefill here
        # For now, just acknowledge receipt (draft KV cache populated by target)

    def _handle_speculate(self, model=None):
        """Handle CMD_SPECULATE: check cache, JIT if miss, respond.

        Args:
            model: draft model runner (for JIT speculation). None = cache-only mode.
        """
        # Receive meta: [B, K, fan_out]
        meta = recv_meta(
            self.async_pg, src=self.target_rank, n=3, device=self.device
        )
        B, K, F = meta

        # Receive fused payload: cache_keys + num_tokens + block_tables + temps
        fused_total = 3 * B + B + B * self.max_blocks + B
        fused = recv_int64(
            self.async_pg,
            src=self.target_rank,
            total_length=fused_total,
            device=self.device,
        )

        off = 0
        cache_keys = fused[off : off + 3 * B].view(B, 3)
        off += 3 * B
        num_tokens = fused[off : off + B]
        off += B
        draft_block_tables = (
            fused[off : off + B * self.max_blocks]
            .view(B, self.max_blocks)
            .to(torch.int32)
        )
        off += B * self.max_blocks
        temps_as_int64 = fused[off : off + B]
        off += B
        assert off == fused_total

        temperatures = int64_to_temps(temps_as_int64)

        # Check tree cache
        out_tokens, out_logits, cache_hits = self._hit_cache(cache_keys, B, K)

        # JIT speculate for misses
        if model is not None and self.jit_speculate:
            miss_mask = cache_hits == 0
            if miss_mask.any():
                self.jit_speculate_fn(
                    model,
                    cache_keys,
                    num_tokens,
                    temperatures,
                    draft_block_tables,
                    out_tokens,
                    out_logits,
                )

        # Send response: fused [cache_hits(B), tokens(B*K)] + logits
        fused_response = torch.cat(
            [cache_hits.reshape(-1), out_tokens.reshape(-1).to(torch.int64)]
        )
        dist.send(fused_response, dst=self.target_rank, group=self.async_pg)
        dist.send(
            out_logits[:, :K, :].contiguous(),
            dst=self.target_rank,
            group=self.async_pg,
        )

        # Reset and populate tree cache for next cycle
        seq_ids = cache_keys[:, 0]
        recovery_tokens = cache_keys[:, 2]
        self._reset_tree_cache()
        self._populate_tree_cache(
            seq_ids, recovery_tokens, out_tokens, out_logits, cache_hits
        )

        logger.debug(
            f"Spec response: B={B}, hits={cache_hits.sum().item()}/{B}"
        )

    # Alias for external use
    jit_speculate_fn = jit_speculate

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def draft_loop(self, model=None):
        """Main async draft loop. Blocks until CMD_EXIT.

        Args:
            model: draft model runner for JIT speculation. None = cache-only.
        """
        logger.info("Draft loop started, waiting for commands...")

        while True:
            cmd = recv_cmd(
                self.async_pg, src=self.target_rank, device=self.device
            )

            if cmd == CMD_PREFILL:
                self._handle_prefill()
                continue

            elif cmd == CMD_SPECULATE:
                t0 = time.perf_counter()
                self._handle_speculate(model)
                self._draft_step_times.append(time.perf_counter() - t0)
                continue

            elif cmd == CMD_EXIT:
                if self._draft_step_times:
                    avg_ms = (
                        sum(self._draft_step_times)
                        * 1000
                        / len(self._draft_step_times)
                    )
                    logger.info(f"Avg draft step time: {avg_ms:.2f}ms")
                logger.info("Draft loop exiting")
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")


def run_draft_process(
    draft_model_path: str,
    gpu_id: int,
    nccl_init_method: str,
    world_size: int,
    rank: int,
    spec_k: int,
    fan_out: int,
    fan_out_list: list,
    fan_out_list_miss: list,
    vocab_size: int,
    dtype_str: str,
    max_blocks: int,
    block_size: int,
    jit_speculate: bool = True,
):
    """Entry point for the draft process (mp.Process target).

    Sets up NCCL, loads the draft model, and enters draft_loop().
    """
    import torch.multiprocessing as mp

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Initialize NCCL process group
    dist.init_process_group(
        backend="nccl",
        init_method=nccl_init_method,
        world_size=world_size,
        rank=rank,
    )
    async_pg = dist.new_group(ranks=[0, rank])

    dtype = getattr(torch, dtype_str)

    runner = SSDDraftRunner(
        draft_model_path=draft_model_path,
        device=device,
        gpu_id=gpu_id,
        async_pg=async_pg,
        target_rank=0,
        spec_k=spec_k,
        fan_out=fan_out,
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
        vocab_size=vocab_size,
        dtype=dtype,
        max_blocks=max_blocks,
        block_size=block_size,
        jit_speculate=jit_speculate,
    )

    # TODO: Load draft model here and pass to draft_loop
    # For now, run in cache-only mode
    runner.draft_loop(model=None)

    dist.destroy_process_group()
