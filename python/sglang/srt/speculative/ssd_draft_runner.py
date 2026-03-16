"""SSD Draft Runner — runs on a separate GPU as an independent process.

Uses a full SGLang TpModelWorker (with ModelRunner, paged attention, KV cache)
instead of a bare HuggingFace wrapper, giving proper KV cache management.

Handles three NCCL commands from the target:
  CMD_PREFILL (1):  Run draft model prefill for new sequences
  CMD_SPECULATE (0): Check tree cache → hit: return cached / miss: JIT speculate
                     Then: glue decode → fork → tree decode → populate cache
  CMD_EXIT (2):     Shutdown

Tree Cache:
  Keys:   [N, 3] — (seq_id, k_idx, recovery_token_id)
  Tokens: [N, K] — speculated token sequences per branch
  Logits: [N, K, V] — draft logits for each step per branch
"""

import copy
import logging
import os
import time
from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.speculative.ssd_nccl_utils import (
    CMD_EXIT,
    CMD_PREFILL,
    CMD_SPECULATE,
    _pg_send,
    int64_to_temps,
    recv_cmd,
    recv_int64,
    recv_meta,
)
from sglang.srt.speculative.ssd_tree_utils import (
    build_fan_indices,
    compute_step_positions_and_slot_maps,
    get_forked_recovery_tokens,
    make_glue_decode_input_ids,
)

logger = logging.getLogger(__name__)


class SSDDraftRunner:
    """Draft model runner for async SSD.

    Runs in a separate process on its own GPU. Uses a full SGLang ModelRunner
    with paged attention and proper KV cache management.
    """

    def __init__(
        self,
        device: torch.device,
        gpu_id: int,
        async_pg: dist.ProcessGroup,
        target_rank: int,
        spec_k: int,
        fan_out: int,
        fan_out_list: List[int],
        fan_out_list_miss: List[int],
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
        self.mq_len_miss = sum(fan_out_list_miss)

        # Pre-build fan index tensors for cache population
        self._fan_idx_hit, self._fan_idx_miss = build_fan_indices(
            fan_out_list, fan_out_list_miss, spec_k, device
        )

        # Tree cache tensors
        self._reset_tree_cache()

        # ModelRunner and memory pool references (set by set_model_runner)
        self.model_runner = None
        self.req_to_token_pool = None
        self.allocator = None

        # Sequence tracking: seq_id → req_pool_idx, seq_id → kv_len
        self._seq_to_req_pool = {}
        self._seq_lens = {}

        # Timing
        self._draft_step_times = []

        logger.info(
            f"SSDDraftRunner initialized: K={spec_k}, fan_out={fan_out}, "
            f"mq_len={self.mq_len}, vocab={vocab_size}, jit={jit_speculate}"
        )

    def set_model_runner(self, model_runner):
        """Attach the ModelRunner from TpModelWorker."""
        self.model_runner = model_runner
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.allocator = model_runner.token_to_kv_pool_allocator

    # ------------------------------------------------------------------
    # req_to_token_pool allocation helpers (bypass Req objects)
    # ------------------------------------------------------------------

    def _alloc_req_pool(self, n: int) -> List[int]:
        """Allocate n req_pool_indices directly from the pool."""
        pool = self.req_to_token_pool
        if n > len(pool.free_slots):
            raise RuntimeError(
                f"No free req_pool slots: need {n}, have {len(pool.free_slots)}"
            )
        indices = pool.free_slots[:n]
        pool.free_slots = pool.free_slots[n:]
        return indices

    def _free_req_pool(self, idx: int):
        """Return a req_pool_index to the pool."""
        self.req_to_token_pool.free_slots.append(idx)

    # ------------------------------------------------------------------
    # Tree cache management
    # ------------------------------------------------------------------

    def _reset_tree_cache(self):
        """Reset tensor-backed tree cache to empty."""
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device
        )
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    def _hit_cache(self, request_keys: torch.Tensor, B: int, K: int):
        """Check tree cache for hits."""
        V = self.vocab_size
        out_logits = torch.zeros((B, K, V), dtype=self.dtype, device=self.device)
        out_tokens = torch.zeros((B, K), dtype=torch.int64, device=self.device)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.tree_cache_keys.numel() > 0:
            eq = request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0)
            match = torch.all(eq, dim=2)
            cache_hits = match.any(dim=1).to(torch.int64)

            if cache_hits.any():
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits.bool()
                out_tokens[sel] = self.tree_cache_tokens[idx[sel]]
                out_logits[sel] = self.tree_cache_logits[idx[sel]]

        return out_tokens, out_logits, cache_hits

    def _populate_tree_cache_from_tree(
        self,
        seq_ids_expanded: torch.Tensor,
        rec_flat: torch.Tensor,
        k_flat: torch.Tensor,
        tokens: torch.Tensor,
        logits: torch.Tensor,
    ):
        """Populate tree cache from tree decode results."""
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1)
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits

    # ------------------------------------------------------------------
    # KV cache rollback
    # ------------------------------------------------------------------

    def _rollback_kv(self, req_pool_idx: int, current_len: int, target_len: int):
        """Free KV cache entries from target_len to current_len."""
        if current_len <= target_len:
            return
        excess_locs = self.req_to_token_pool.req_to_token[
            req_pool_idx, target_len:current_len
        ].to(torch.int64)
        self.allocator.free(excess_locs)

    # ------------------------------------------------------------------
    # JIT speculation via ModelRunner
    # ------------------------------------------------------------------

    def _jit_speculate(
        self,
        request_keys: torch.Tensor,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logits: torch.Tensor,
        miss_mask: torch.Tensor,
    ):
        """Run draft model forward K times for cache-miss sequences.

        Uses ModelRunner with paged attention for batched forward passes.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
            clamp_position,
        )

        miss_indices = miss_mask.nonzero(as_tuple=True)[0]
        B_miss = len(miss_indices)
        if B_miss == 0:
            return

        # Gather miss sequence info
        miss_seq_ids = []
        miss_req_pool_indices = []
        for b_idx in miss_indices:
            b = int(b_idx.item())
            seq_id = int(request_keys[b, 0].item())
            miss_seq_ids.append(seq_id)
            req_pool_idx = self._seq_to_req_pool.get(seq_id)
            if req_pool_idx is None:
                logger.warning(f"Seq {seq_id} not in req_pool, skipping")
                return
            miss_req_pool_indices.append(req_pool_idx)

        miss_req_pool_tensor = torch.tensor(
            miss_req_pool_indices, dtype=torch.int64, device=self.device
        )
        miss_recovery_tokens = request_keys[miss_indices, 2]
        miss_verified_lens = num_tokens[miss_indices].to(torch.int64)
        miss_temps = temperatures[miss_indices]

        # Rollback KV to verified_len. verified_len = number of committed
        # tokens (excluding recovery). Recovery goes at position verified_len.
        # Positions 0..verified_len-1 are already correct in KV cache.
        for i, seq_id in enumerate(miss_seq_ids):
            current_len = self._seq_lens.get(seq_id, 0)
            verified_len = int(miss_verified_lens[i].item())
            if current_len > verified_len:
                self._rollback_kv(miss_req_pool_indices[i], current_len, verified_len)
            self._seq_lens[seq_id] = verified_len

        # Run recovery token as step 0 to build KV and get logits
        recovery_tokens = miss_recovery_tokens.to(torch.int64)
        base_seq_lens = torch.tensor(
            [self._seq_lens[int(miss_seq_ids[i])] for i in range(B_miss)],
            dtype=torch.int64, device=self.device,
        )

        # Allocate KV slots for recovery token
        recovery_cache_locs = self.allocator.alloc(B_miss)
        if recovery_cache_locs is None:
            logger.warning("KV cache full during recovery step, skipping speculation")
            return

        for i in range(B_miss):
            pos = int(base_seq_lens[i].item())
            self.req_to_token_pool.req_to_token[
                miss_req_pool_indices[i], pos
            ] = recovery_cache_locs[i].to(torch.int32)

        recovery_seq_lens = base_seq_lens + 1
        seq_lens_cpu = recovery_seq_lens.to(torch.int32).cpu()
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=B_miss,
            input_ids=recovery_tokens.to(torch.int32),
            req_pool_indices=miss_req_pool_tensor,
            seq_lens=recovery_seq_lens.to(torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=recovery_cache_locs,
            seq_lens_sum=int(recovery_seq_lens.sum().item()),
            positions=clamp_position(recovery_seq_lens),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
        )
        recovery_output = self.model_runner.forward(forward_batch)
        recovery_logits = recovery_output.logits_output.next_token_logits

        # Always use greedy (argmax) in draft to maximize accept rate.
        # SGLang converts temperature=0 to temperature=1.0 + top_k=1,
        # so we cannot rely on temperature to detect greedy mode.
        first_spec = recovery_logits.argmax(dim=-1)

        # Update seq_lens and store step 0 results
        for i, seq_id in enumerate(miss_seq_ids):
            self._seq_lens[seq_id] = int(recovery_seq_lens[i].item())
        for i, b_idx in enumerate(miss_indices):
            b = int(b_idx.item())
            out_logits[b, 0, :] = recovery_logits[i]
            out_tokens[b, 0] = first_spec[i]

        # Run remaining K-1 decode steps
        current_tokens = first_spec
        current_seq_lens = recovery_seq_lens.clone()

        for step in range(1, self.K):
            # Allocate new KV cache locations
            new_cache_locs = self.allocator.alloc(B_miss)
            if new_cache_locs is None:
                logger.warning(f"KV cache full at step {step}, stopping speculation")
                break

            # Write cache locations to req_to_token_pool
            for i in range(B_miss):
                pos = int(current_seq_lens[i].item())
                self.req_to_token_pool.req_to_token[
                    miss_req_pool_indices[i], pos
                ] = new_cache_locs[i].to(torch.int32)

            # Advance seq_lens (new token being added)
            current_seq_lens += 1

            # Build ForwardBatch for DECODE
            seq_lens_cpu = current_seq_lens.to(torch.int32).cpu()
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=B_miss,
                input_ids=current_tokens.to(torch.int32),
                req_pool_indices=miss_req_pool_tensor,
                seq_lens=current_seq_lens.to(torch.int32),
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=new_cache_locs,
                seq_lens_sum=int(current_seq_lens.sum().item()),
                positions=clamp_position(current_seq_lens),
                capture_hidden_mode=CaptureHiddenMode.NULL,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
            )

            # Run forward (init attention backend each step since seq_lens change)
            output = self.model_runner.forward(forward_batch)
            logits = output.logits_output.next_token_logits  # [B_miss, V]

            # Store logits
            for i, b_idx in enumerate(miss_indices):
                b = int(b_idx.item())
                out_logits[b, step, :] = logits[i]

            # Always greedy in draft to maximize accept rate
            next_tokens = logits.argmax(dim=-1)

            # Store tokens
            for i, b_idx in enumerate(miss_indices):
                b = int(b_idx.item())
                out_tokens[b, step] = next_tokens[i]

            current_tokens = next_tokens

        # Process the last predicted token to create its KV entry.
        # Without this, when the target accepts all K tokens, there's a gap:
        # draft has KV for K positions but the last predicted token has no KV.
        tail_cache_locs = self.allocator.alloc(B_miss)
        if tail_cache_locs is not None:
            for i in range(B_miss):
                pos = int(current_seq_lens[i].item())
                self.req_to_token_pool.req_to_token[
                    miss_req_pool_indices[i], pos
                ] = tail_cache_locs[i].to(torch.int32)
            current_seq_lens += 1
            sl_cpu = current_seq_lens.to(torch.int32).cpu()
            fb = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=B_miss,
                input_ids=current_tokens.to(torch.int32),
                req_pool_indices=miss_req_pool_tensor,
                seq_lens=current_seq_lens.to(torch.int32),
                seq_lens_cpu=sl_cpu,
                out_cache_loc=tail_cache_locs,
                seq_lens_sum=int(current_seq_lens.sum().item()),
                positions=clamp_position(current_seq_lens),
                capture_hidden_mode=CaptureHiddenMode.NULL,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
            )
            self.model_runner.forward(fb)

        # Update tracked seq_lens
        for i, seq_id in enumerate(miss_seq_ids):
            self._seq_lens[seq_id] = int(current_seq_lens[i].item())

    # ------------------------------------------------------------------
    # Tree decode pipeline (glue → fork → tree decode → populate)
    # ------------------------------------------------------------------

    def _build_tree_batch(
        self,
        out_tokens: torch.Tensor,
        out_logits: torch.Tensor,
        cache_hits: torch.Tensor,
        cache_keys: torch.Tensor,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
    ):
        """Build the tree batch for pre-computing the next speculation tree.

        Currently uses approximate logits (no model forward) since
        tree decode via ModelRunner is deferred to a future phase.
        """
        B = cache_keys.shape[0]
        K = self.K
        rec_tokens = cache_keys[:, 2]
        seq_ids = cache_keys[:, 0]

        # Use returned logits as approximate glue logits (no model forward)
        rec_logits = torch.zeros(
            (B, 1, self.vocab_size), dtype=self.dtype, device=self.device
        )
        glue_logits = torch.cat([rec_logits, out_logits], dim=1)

        returned_tokens = torch.cat(
            [rec_tokens.unsqueeze(1), out_tokens], dim=1
        )

        forked_tokens = get_forked_recovery_tokens(
            logits=glue_logits,
            cache_hits=cache_hits,
            returned_tokens=returned_tokens,
            fan_out_list=self.fan_out_list,
            fan_out_list_miss=self.fan_out_list_miss,
        )

        MQ_LEN = self.mq_len
        N = B * MQ_LEN

        b_flat = (
            torch.arange(B, device=self.device)
            .unsqueeze(1)
            .expand(B, MQ_LEN)
            .flatten()
        )
        seq_ids_expanded = seq_ids[b_flat]
        temperatures_expanded = temperatures[b_flat]

        initial_positions = num_tokens[b_flat] + K
        fkp1_flat = torch.arange(MQ_LEN, device=self.device).repeat(B)
        initial_positions = initial_positions + fkp1_flat

        k_flat = torch.cat(
            [
                self._fan_idx_hit if cache_hits[b] else self._fan_idx_miss
                for b in range(B)
            ]
        )

        step_positions, step_context_lens, step_slot_maps = (
            compute_step_positions_and_slot_maps(
                initial_positions=initial_positions,
                K=K,
                B=B,
                N=N,
                mq_len=MQ_LEN,
                block_tables=draft_block_tables,
                block_size=self.block_size,
                device=self.device,
            )
        )

        return {
            "B": B,
            "K": K,
            "N": N,
            "MQ_LEN": MQ_LEN,
            "input_ids": forked_tokens.flatten(),
            "seq_ids_expanded": seq_ids_expanded,
            "rec_flat": forked_tokens.flatten(),
            "k_flat": k_flat,
            "temperatures": temperatures_expanded,
            "block_tables": draft_block_tables,
            "step_positions": step_positions,
            "step_context_lens": step_context_lens,
            "step_slot_maps": step_slot_maps,
            "cache_hits": cache_hits,
        }

    def _decode_tree(self, args: dict):
        """Decode all N branches for K steps using random logits.

        Tree decode via ModelRunner is deferred to a future phase.
        """
        B, K, N = args["B"], args["K"], args["N"]
        V = self.vocab_size

        spec_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
        spec_logits = torch.zeros((N, K, V), dtype=self.dtype, device=self.device)

        current_input_ids = args["input_ids"]
        temperatures = args["temperatures"]
        all_greedy = bool((temperatures == 0).all())

        for depth in range(K):
            logits = torch.randn(
                (N, V), dtype=self.dtype, device=self.device
            )

            spec_logits[:, depth, :] = logits

            if all_greedy:
                next_tokens = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(
                    logits / temperatures.unsqueeze(-1).clamp(min=1e-6),
                    dim=-1,
                )
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            spec_tokens[:, depth] = next_tokens
            current_input_ids = next_tokens

        return spec_tokens, spec_logits

    # ------------------------------------------------------------------
    # NCCL command handlers
    # ------------------------------------------------------------------

    def _handle_prefill(self):
        """Handle CMD_PREFILL: receive data and run draft model prefill."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
            compute_position,
        )

        meta = recv_meta(
            self.async_pg, src=self.target_rank, n=3, device=self.device
        )
        total_new_tokens, batch_size, max_blocks = meta

        # Fused payload: input_ids + num_tokens + seq_ids + block_tables
        fused_total = total_new_tokens + batch_size + batch_size + batch_size * max_blocks
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
        seq_ids = fused[off : off + batch_size]
        off += batch_size
        draft_block_tables = (
            fused[off : off + batch_size * max_blocks]
            .view(batch_size, max_blocks)
            .to(torch.int32)
        )

        logger.debug(f"Draft prefill: B={batch_size}, tokens={total_new_tokens}")

        if self.model_runner is None:
            logger.warning("No model_runner, skipping prefill forward")
            return

        # Allocate req_pool_indices for new sequences
        req_pool_indices = self._alloc_req_pool(batch_size)

        # Allocate KV cache locations for all tokens
        all_cache_locs = self.allocator.alloc(total_new_tokens)
        if all_cache_locs is None:
            logger.error("KV cache full during prefill, cannot allocate")
            # Return the req_pool_indices
            for idx in req_pool_indices:
                self._free_req_pool(idx)
            return

        # Set up req_to_token_pool mapping and track sequences
        token_offset = 0
        cache_offset = 0
        seq_lens_list = []
        for i in range(batch_size):
            seq_id = int(seq_ids[i].item())
            seq_len = int(num_tokens[i].item())
            req_pool_idx = req_pool_indices[i]

            # Write cache locations for this sequence
            self.req_to_token_pool.req_to_token[
                req_pool_idx, :seq_len
            ] = all_cache_locs[cache_offset : cache_offset + seq_len].to(torch.int32)

            # Track sequence
            self._seq_to_req_pool[seq_id] = req_pool_idx
            self._seq_lens[seq_id] = seq_len

            seq_lens_list.append(seq_len)
            token_offset += seq_len
            cache_offset += seq_len

        # Build ForwardBatch for EXTEND (prefill)
        req_pool_indices_tensor = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        seq_lens_tensor = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device
        )
        extend_seq_lens = seq_lens_tensor.clone()
        extend_prefix_lens = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )

        positions, extend_start_loc = compute_position(
            self.model_runner.server_args.attention_backend,
            extend_prefix_lens,
            extend_seq_lens,
            total_new_tokens,
        )

        seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=input_ids.to(torch.int32),
            req_pool_indices=req_pool_indices_tensor,
            seq_lens=seq_lens_tensor,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=all_cache_locs,
            seq_lens_sum=total_new_tokens,
            positions=positions,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_num_tokens=total_new_tokens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=seq_lens_list,
            extend_prefix_lens_cpu=[0] * batch_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
        )

        # Run prefill forward
        self.model_runner.forward(forward_batch)

        logger.debug(
            f"Draft prefill done: {batch_size} seqs, "
            f"tracked seqs: {len(self._seq_to_req_pool)}"
        )

    def _handle_speculate(self):
        """Handle CMD_SPECULATE: check cache → respond → build tree for next."""
        # ---- Receive request ----
        meta = recv_meta(
            self.async_pg, src=self.target_rank, n=3, device=self.device
        )
        B, K, F = meta

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

        # ---- Step 1: JIT speculate (always, tree cache disabled with ModelRunner) ----
        V = self.vocab_size
        out_logits = torch.zeros((B, K, V), dtype=self.dtype, device=self.device)
        out_tokens = torch.zeros((B, K), dtype=torch.int64, device=self.device)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.model_runner is not None and self.jit_speculate:
            miss_mask = torch.ones(B, dtype=torch.bool, device=self.device)
            self._jit_speculate(
                cache_keys, num_tokens, temperatures,
                draft_block_tables, out_tokens, out_logits,
                miss_mask=miss_mask,
            )

        # ---- Step 2: Send response IMMEDIATELY ----
        fused_response = torch.cat(
            [cache_hits.reshape(-1), out_tokens.reshape(-1).to(torch.int64)]
        )
        _pg_send(self.async_pg, fused_response, self.target_rank)
        _pg_send(self.async_pg, out_logits[:, :K, :].contiguous(), self.target_rank)

        hit_count = cache_hits.sum().item()
        logger.debug(f"Spec response: B={B}, hits={hit_count}/{B}")

        # ---- Step 3: Build tree for NEXT request (async) ----
        self._reset_tree_cache()

        tree_args = self._build_tree_batch(
            out_tokens=out_tokens,
            out_logits=out_logits,
            cache_hits=cache_hits,
            cache_keys=cache_keys,
            num_tokens=num_tokens,
            temperatures=temperatures,
            draft_block_tables=draft_block_tables,
        )

        if tree_args is not None:
            spec_tokens, spec_logits = self._decode_tree(tree_args)

            self._populate_tree_cache_from_tree(
                seq_ids_expanded=tree_args["seq_ids_expanded"],
                rec_flat=tree_args["rec_flat"],
                k_flat=tree_args["k_flat"],
                tokens=spec_tokens,
                logits=spec_logits,
            )

            logger.debug(
                f"Tree cache populated: {self.tree_cache_keys.shape[0]} entries"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def draft_loop(self):
        """Main async draft loop. Blocks until CMD_EXIT."""
        logger.info("Draft loop started, waiting for commands...")

        while True:
            cmd = recv_cmd(
                self.async_pg, src=self.target_rank, device=self.device
            )

            if cmd == CMD_PREFILL:
                self._handle_prefill()

            elif cmd == CMD_SPECULATE:
                t0 = time.perf_counter()
                self._handle_speculate()
                self._draft_step_times.append(time.perf_counter() - t0)

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
    server_args,
    gpu_id: int,
    nccl_init_method: str,
    nccl_port_dist: int,
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

    Creates a full TpModelWorker with ModelRunner on the draft GPU,
    sets up async NCCL, and enters draft_loop().
    """
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.managers.tp_worker import TpModelWorker

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    dtype = getattr(torch, dtype_str)

    # --- 1. Initialize NCCL process group for async communication ---
    nccl_host, nccl_port_str = nccl_init_method.replace("tcp://", "").split(":")
    store = dist.TCPStore(
        host_name=nccl_host,
        port=int(nccl_port_str),
        world_size=2,
        is_master=False,
        wait_for_workers=False,
    )
    async_pg = dist.ProcessGroupNCCL(store, 1, 2)

    # --- 2. Set up server_args for draft-as-standalone model ---
    # Deep copy to avoid mutating the original
    draft_server_args = copy.deepcopy(server_args)
    # Swap model_path to draft model so TpModelWorker loads the draft model
    draft_server_args.model_path = draft_server_args.speculative_draft_model_path
    # Clear speculative config so it doesn't try to act as a spec worker
    draft_server_args.speculative_algorithm = None
    draft_server_args.speculative_draft_model_path = None
    # Single GPU, no TP
    draft_server_args.tp_size = 1
    draft_server_args.dp_size = 1
    draft_server_args.pp_size = 1
    draft_server_args.ep_size = 1
    # Disable cuda graph for now (simpler init)
    draft_server_args.disable_cuda_graph = True
    # Skip tokenizer (not needed for draft)
    draft_server_args.skip_tokenizer_init = True
    # Use the separate port for distributed init
    draft_server_args.dist_init_addr = None  # will use nccl_port
    # Disable features that may conflict
    draft_server_args.enable_dp_attention = False
    draft_server_args.disable_overlap_schedule = True

    # --- 3. Initialize distributed environment for the draft process ---
    # TpModelWorker with is_draft_worker=False will call init_distributed_environment
    # inside init_torch_distributed. We provide a dedicated nccl_port for this.
    logger.info(f"Creating TpModelWorker on GPU {gpu_id}, dist port {nccl_port_dist}")
    try:
        tp_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=0,
            pp_rank=0,
            dp_rank=None,
            moe_ep_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            nccl_port=nccl_port_dist,
            is_draft_worker=False,  # standalone model, creates own pools
        )
        model_runner = tp_worker.model_runner
        logger.info(
            f"TpModelWorker created: "
            f"max_tokens={model_runner.max_total_num_tokens}, "
            f"max_reqs={model_runner.max_running_requests}"
        )
    except Exception as e:
        logger.error(f"Failed to create TpModelWorker: {e}", exc_info=True)
        return

    # --- 4. Create SSDDraftRunner ---
    runner = SSDDraftRunner(
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
    runner.set_model_runner(model_runner)

    # --- 5. Enter draft loop ---
    runner.draft_loop()
