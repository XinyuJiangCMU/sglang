"""SSD Draft Runner — runs on a separate GPU as an independent process.

Handles three NCCL commands from the target:
  CMD_PREFILL (1):  Run draft model prefill for new sequences
  CMD_SPECULATE (0): Check tree cache → hit: return cached / miss: JIT speculate
                     Then: glue decode → fork → tree decode → populate cache
  CMD_EXIT (2):     Shutdown

Tree Cache:
  Keys:   [N, 3] — (seq_id, k_idx, recovery_token_id)
  Tokens: [N, K] — speculated token sequences per branch
  Logits: [N, K, V] — draft logits for each step per branch

Tree Decode Pipeline (runs AFTER responding to target):
  1. Glue Decode: run draft on [rec_token, spec_0, ..., spec_{K-1}] → K+1 logits
  2. Fork: from K+1 logits, sample top-F per position → N=B*MQ_LEN branches
  3. Tree Decode: run K steps on all N branches in parallel → [N, K] tokens
  4. Populate Cache: store results keyed by (seq_id, k_idx, forked_token)
"""

import logging
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

    Runs in a separate process on its own GPU. Loads the draft model,
    maintains a tree cache, and responds to NCCL commands from target.
    After responding, builds a speculation tree for the NEXT request.
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

        # Timing
        self._draft_step_times = []

        logger.info(
            f"SSDDraftRunner initialized: K={spec_k}, fan_out={fan_out}, "
            f"mq_len={self.mq_len}, vocab={vocab_size}, jit={jit_speculate}"
        )

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
        """Check tree cache for hits.

        Args:
            request_keys: [B, 3] — (seq_id, k_idx, recovery_token_id)

        Returns:
            out_tokens: [B, K], out_logits: [B, K, V], cache_hits: [B]
        """
        V = self.vocab_size
        out_logits = torch.zeros((B, K, V), dtype=self.dtype, device=self.device)
        out_tokens = torch.zeros((B, K), dtype=torch.int64, device=self.device)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership: [B, T, 3] → [B, T] → [B]
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
        """Populate tree cache from tree decode results.

        Each branch in the tree becomes a cache entry keyed by
        (seq_id, fork_depth, forked_recovery_token).

        Args:
            seq_ids_expanded: [N] — sequence ID per branch
            rec_flat: [N] — forked recovery token per branch
            k_flat: [N] — fork depth index per branch
            tokens: [N, K] — speculated tokens per branch
            logits: [N, K, V] — logits per branch
        """
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1)
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits

    # ------------------------------------------------------------------
    # JIT speculation (sequential K-step, fallback for cache miss)
    # ------------------------------------------------------------------

    def _jit_speculate(
        self,
        model,
        request_keys: torch.Tensor,
        num_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        draft_block_tables: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logits: torch.Tensor,
    ):
        """Run draft model forward K times for cache-miss sequences."""
        input_ids = request_keys[:, -1]  # recovery_token_id
        positions = num_tokens - 1

        for i in range(self.K):
            logits = model.forward_draft_step(
                input_ids=input_ids,
                positions=positions,
                block_tables=draft_block_tables,
            )
            out_logits[:, i, :] = logits

            if (temperatures == 0).all():
                next_tokens = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(
                    logits / temperatures.unsqueeze(-1).clamp(min=1e-6), dim=-1
                )
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            out_tokens[:, i] = next_tokens
            input_ids = next_tokens
            positions = positions + 1

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
        model=None,
    ):
        """Build the tree batch: glue decode → fork → prepare tree decode args.

        After the immediate response to target, this runs asynchronously
        to pre-compute the speculation tree for the NEXT request.

        Args:
            out_tokens: [B, K] — tokens returned to target
            out_logits: [B, K, V] — logits returned to target
            cache_hits: [B] — cache hit status
            cache_keys: [B, 3] — request keys
            num_tokens: [B] — sequence lengths
            temperatures: [B] — sampling temperatures
            draft_block_tables: [B, max_blocks] — block tables
            model: draft model (None = skip glue decode, use returned logits)

        Returns:
            tree_decode_args dict or None if no model available
        """
        B = cache_keys.shape[0]
        K = self.K
        rec_tokens = cache_keys[:, 2]  # recovery token IDs
        seq_ids = cache_keys[:, 0]

        # ---- Step 1: Glue Decode ----
        # Combine recovery + spec tokens and run draft model forward
        # to get K+1 logits for forking
        if model is not None:
            glue_input_ids = make_glue_decode_input_ids(out_tokens, rec_tokens)
            # [B*(K+1)] → run draft model → [B, K+1, V] logits
            glue_logits = model.forward_glue_decode(
                input_ids=glue_input_ids,
                num_tokens=num_tokens,
                block_tables=draft_block_tables,
                K=K,
            )  # [B, K+1, V]
        else:
            # No model: use returned logits as approximate glue logits
            # Pad with zeros for the recovery position
            rec_logits = torch.zeros(
                (B, 1, self.vocab_size), dtype=self.dtype, device=self.device
            )
            glue_logits = torch.cat([rec_logits, out_logits], dim=1)  # [B, K+1, V]

        # ---- Step 2: Fork ----
        # Build returned_tokens = [rec, spec_0, ..., spec_{K-1}]
        returned_tokens = torch.cat(
            [rec_tokens.unsqueeze(1), out_tokens], dim=1
        )  # [B, K+1]

        forked_tokens = get_forked_recovery_tokens(
            logits=glue_logits,
            cache_hits=cache_hits,
            returned_tokens=returned_tokens,
            fan_out_list=self.fan_out_list,
            fan_out_list_miss=self.fan_out_list_miss,
        )  # [B, MQ_LEN]

        # ---- Step 3: Prepare tree decode args ----
        MQ_LEN = self.mq_len
        N = B * MQ_LEN

        # Expand sequence info to per-branch
        b_flat = (
            torch.arange(B, device=self.device)
            .unsqueeze(1)
            .expand(B, MQ_LEN)
            .flatten()
        )  # [N]
        seq_ids_expanded = seq_ids[b_flat]  # [N]
        temperatures_expanded = temperatures[b_flat]  # [N]

        # Initial positions for tree decode: after glue decode output
        initial_positions = num_tokens[b_flat] + K  # [N]
        fkp1_flat = torch.arange(MQ_LEN, device=self.device).repeat(B)  # [N]
        initial_positions = initial_positions + fkp1_flat

        # Build k_flat (fork depth index per branch) for cache population
        k_flat = torch.cat(
            [
                self._fan_idx_hit if cache_hits[b] else self._fan_idx_miss
                for b in range(B)
            ]
        )  # [N]

        # Precompute positions and slot maps for all K steps
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
            "input_ids": forked_tokens.flatten(),  # [N]
            "seq_ids_expanded": seq_ids_expanded,  # [N]
            "rec_flat": forked_tokens.flatten(),  # [N] — the forked recovery tokens
            "k_flat": k_flat,  # [N]
            "temperatures": temperatures_expanded,  # [N]
            "block_tables": draft_block_tables,  # [B, M]
            "step_positions": step_positions,  # [K, N]
            "step_context_lens": step_context_lens,  # [K, N]
            "step_slot_maps": step_slot_maps,  # [K, N]
            "cache_hits": cache_hits,  # [B]
        }

    def _decode_tree(self, args: dict, model=None):
        """Decode all N branches for K steps.

        All N branches are processed in a single batched forward pass
        per depth, maximizing GPU utilization.

        Args:
            args: tree decode args from _build_tree_batch()
            model: draft model for forward passes

        Returns:
            spec_tokens: [N, K] — generated tokens per branch
            spec_logits: [N, K, V] — logits per branch
        """
        B, K, N = args["B"], args["K"], args["N"]
        V = self.vocab_size

        spec_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
        spec_logits = torch.zeros((N, K, V), dtype=self.dtype, device=self.device)

        current_input_ids = args["input_ids"]  # [N]
        temperatures = args["temperatures"]  # [N]
        all_greedy = bool((temperatures == 0).all())

        for depth in range(K):
            if model is not None:
                # Run all N branches in one forward pass
                logits = model.forward_tree_step(
                    input_ids=current_input_ids,
                    positions=args["step_positions"][depth],
                    slot_maps=args["step_slot_maps"][depth],
                    context_lens=args["step_context_lens"][depth],
                    block_tables=args["block_tables"],
                    b_flat=torch.arange(B, device=self.device)
                    .unsqueeze(1)
                    .expand(B, args["MQ_LEN"])
                    .flatten(),
                )  # [N, V]
            else:
                # No model: generate random logits (for testing pipeline)
                logits = torch.randn(
                    (N, V), dtype=self.dtype, device=self.device
                )

            spec_logits[:, depth, :] = logits

            # Sample next tokens
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
        """Handle CMD_PREFILL: receive prefill data from target."""
        meta = recv_meta(
            self.async_pg, src=self.target_rank, n=3, device=self.device
        )
        total_new_tokens, batch_size, max_blocks = meta

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

        logger.debug(f"Draft prefill: B={batch_size}, tokens={total_new_tokens}")
        # TODO: Run draft model prefill with these inputs

    def _handle_speculate(self, model=None):
        """Handle CMD_SPECULATE: check cache → respond → build tree for next.

        Flow:
          1. Receive request from target
          2. Check tree cache for hits (+ JIT for misses)
          3. Send response immediately (low latency path)
          4. Build speculation tree for NEXT request (async, overlaps with target verify)
        """
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

        # ---- Step 1: Check cache + JIT ----
        out_tokens, out_logits, cache_hits = self._hit_cache(cache_keys, B, K)

        if model is not None and self.jit_speculate:
            miss_mask = cache_hits == 0
            if miss_mask.any():
                self._jit_speculate(
                    model, cache_keys, num_tokens, temperatures,
                    draft_block_tables, out_tokens, out_logits,
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
            model=model,
        )

        if tree_args is not None:
            # ---- Step 4: Tree decode ----
            spec_tokens, spec_logits = self._decode_tree(tree_args, model)

            # ---- Step 5: Populate cache ----
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

    def draft_loop(self, model=None):
        """Main async draft loop. Blocks until CMD_EXIT.

        Args:
            model: draft model runner for JIT/glue/tree decode. None = cache-only.
        """
        logger.info("Draft loop started, waiting for commands...")

        while True:
            cmd = recv_cmd(
                self.async_pg, src=self.target_rank, device=self.device
            )

            if cmd == CMD_PREFILL:
                self._handle_prefill()

            elif cmd == CMD_SPECULATE:
                t0 = time.perf_counter()
                self._handle_speculate(model)
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


class HFDraftModel:
    """Lightweight HuggingFace-based draft model wrapper.

    Provides simple forward interfaces for the SSDDraftRunner without
    requiring full SGLang ModelRunner infrastructure. Uses HuggingFace's
    built-in KV cache for simplicity.

    This is a minimal implementation for getting the async pipeline working.
    For production, consider using SGLang's ModelRunner with paged attention.
    """

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading draft model from {model_path} on {device}")
        self.device = device
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        # Per-sequence KV cache: {seq_idx: past_key_values}
        self._kv_caches = {}
        logger.info(f"Draft model loaded: {type(self.model).__name__}")

    def clear_kv_cache(self):
        """Clear all KV caches."""
        self._kv_caches.clear()

    def forward_draft_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decode step on the draft model.

        Args:
            input_ids: [B] token IDs
            positions: [B] position indices (unused, implicit in KV cache)
            block_tables: [B, M] block tables (unused with HF cache)

        Returns:
            logits: [B, V]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(1) if input_ids.dim() == 1 else input_ids,
                use_cache=False,  # Simplified: no KV cache tracking
            )
            # Last token logits
            return outputs.logits[:, -1, :]

    def forward_glue_decode(
        self,
        input_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        block_tables: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Run glue decode: process K+1 tokens per sequence.

        Args:
            input_ids: [B*(K+1)] flattened input IDs
            num_tokens: [B] sequence lengths
            block_tables: [B, M] block tables
            K: speculation depth

        Returns:
            logits: [B, K+1, V]
        """
        B = num_tokens.shape[0]
        Kp1 = K + 1
        # Reshape to [B, K+1]
        tokens_2d = input_ids.view(B, Kp1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens_2d,
                use_cache=False,
            )
            return outputs.logits  # [B, K+1, V]

    def forward_tree_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_maps: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        b_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Run one tree decode step on all N branches.

        Args:
            input_ids: [N] token IDs
            positions: [N] position indices
            slot_maps: [N] KV slot mappings (unused with HF)
            context_lens: [N] context lengths (unused with HF)
            block_tables: [B, M] block tables (unused with HF)
            b_flat: [N] batch index per branch

        Returns:
            logits: [N, V]
        """
        N = input_ids.shape[0]
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(1),
                use_cache=False,
            )
            return outputs.logits[:, -1, :]


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
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Initialize NCCL process group for async communication.
    # Use TCPStore + ProcessGroupNCCL directly (matches target side).
    nccl_host, nccl_port_str = nccl_init_method.replace("tcp://", "").split(":")
    store = dist.TCPStore(
        host_name=nccl_host,
        port=int(nccl_port_str),
        world_size=world_size,
        is_master=False,  # draft is rank 1 = worker
        wait_for_workers=False,
    )
    async_pg = dist.ProcessGroupNCCL(
        store, rank, world_size
    )

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

    # Load draft model
    model = None
    try:
        model = HFDraftModel(
            model_path=draft_model_path,
            device=device,
            dtype=dtype,
        )
    except Exception as e:
        logger.warning(
            f"Failed to load draft model: {e}. "
            "Running in cache-only mode (tree decode uses random logits)."
        )

    runner.draft_loop(model=model)

    # No dist.destroy_process_group() — we used ProcessGroupNCCL directly
