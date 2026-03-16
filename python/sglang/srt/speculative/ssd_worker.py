"""SSD (Speculative Speculative Decoding) Worker.

Phase 1 (sync): Sequential draft + verify on same GPU.
  - Inherits EAGLEWorker's draft/verify pipeline with topk=1 (linear speculation).
  - Loads a full standalone draft model on the same GPU as target.

Phase 3 (async): Draft on separate GPU with NCCL + tree cache.
  - Spawns SSDDraftRunner as a separate process on its own GPU.
  - Target communicates with draft via SSDAsyncSpeculator (NCCL process group).
  - Tree cache enables speculative pre-computation across decode steps.

Phase 5 (integration): Full async pipeline with model loading.
  - SSDWorker.forward_batch_generation() routes to async path when ssd_async=True.
  - Async extend: target forward + CMD_PREFILL to draft process.
  - Async decode: speculate via NCCL → build EagleVerifyInput → reuse verify pipeline.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.speculative.ssd_info import SSDDraftInput
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class SSDWorker(EAGLEWorker):
    """Speculative Speculative Decoding worker.

    Sync mode (default): Reuses EAGLEWorker's draft/verify pipeline with topk=1.
    Async mode (--speculative-ssd-async): Spawns draft on separate GPU,
      communicates via NCCL, uses tree cache for speculation reuse.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse SSD-specific config
        self.spec_k = server_args.speculative_num_steps or 6
        self.ssd_async = server_args.speculative_ssd_async
        self.ssd_fan_out = server_args.speculative_ssd_fan_out

        # Parse fan-out lists (for async tree cache)
        if server_args.speculative_ssd_fan_out_list:
            self.fan_out_list = [
                int(x) for x in server_args.speculative_ssd_fan_out_list.split(",")
            ]
        else:
            self.fan_out_list = [self.ssd_fan_out] * (self.spec_k + 1)

        if server_args.speculative_ssd_fan_out_list_miss:
            self.fan_out_list_miss = [
                int(x)
                for x in server_args.speculative_ssd_fan_out_list_miss.split(",")
            ]
        else:
            self.fan_out_list_miss = list(self.fan_out_list)

        self.mq_len = sum(self.fan_out_list)

        # Set defaults for sync SD: topk=1 (linear speculation), num_draft_tokens=K
        if server_args.speculative_num_steps is None:
            server_args.speculative_num_steps = self.spec_k
        if server_args.speculative_eagle_topk is None:
            server_args.speculative_eagle_topk = 1
        if server_args.speculative_num_draft_tokens is None:
            server_args.speculative_num_draft_tokens = self.spec_k

        logger.info(
            f"SSD config: k={self.spec_k}, async={self.ssd_async}, "
            f"topk={server_args.speculative_eagle_topk}, "
            f"num_draft_tokens={server_args.speculative_num_draft_tokens}, "
            f"fan_out={self.ssd_fan_out}, mq_len={self.mq_len}"
        )

        if self.ssd_async:
            self._init_async(
                server_args, gpu_id, tp_rank, dp_rank,
                moe_ep_rank, attn_cp_rank, moe_dp_rank,
                nccl_port, target_worker,
            )
        else:
            self._init_sync(
                server_args, gpu_id, tp_rank, dp_rank,
                moe_ep_rank, attn_cp_rank, moe_dp_rank,
                nccl_port, target_worker,
            )

    def _init_sync(
        self, server_args, gpu_id, tp_rank, dp_rank,
        moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port, target_worker,
    ):
        """Sync mode: load draft model on same GPU (StandaloneWorker pattern)."""
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override context length to match target
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Disable cuda graph during init, capture later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share allocator with target worker
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker (TpModelWorker.__init__)
        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Dummy tensors (needed by parent class)
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.async_speculator = None
        self.draft_process = None

        logger.info("SSD worker initialized (sync mode)")

    def _init_async(
        self, server_args, gpu_id, tp_rank, dp_rank,
        moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port, target_worker,
    ):
        """Async mode: spawn draft on separate GPU, init NCCL process group.

        The draft model runs as a separate process on its own GPU.
        Target communicates via NCCL send/recv through SSDAsyncSpeculator.
        Only tp_rank=0 spawns the draft process and communicates with it.
        """
        from sglang.srt.speculative.ssd_async_speculator import SSDAsyncSpeculator
        from sglang.srt.speculative.ssd_draft_runner import run_draft_process

        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Only tp_rank 0 spawns draft process and manages async speculation.
        # Other TP ranks only participate in the target verify forward.
        if tp_rank != 0:
            self.async_speculator = None
            self.draft_process = None

            # Still need model_runner/model_config proxied through target
            self._model_runner = target_worker.model_runner
            self.model_config = target_worker.model_config
            self.req_to_token_pool, self.token_to_kv_pool_allocator = (
                target_worker.get_memory_pool()
            )
            self.num_new_pages_per_topk = torch.empty(
                (), dtype=torch.int64, device=self.device
            )
            self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

            logger.info(f"SSD async: tp_rank={tp_rank}, skipping draft spawn (non-master)")
            return

        # Determine draft GPU (last visible GPU)
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_gpus:
            gpu_list = [int(g) for g in visible_gpus.split(",")]
            self.draft_gpu_id = len(gpu_list) - 1  # last GPU in visible set
        else:
            self.draft_gpu_id = server_args.tp_size  # GPU after TP group

        # NCCL init for async process group — find a free port
        import socket
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.bind(("127.0.0.1", 0))
        nccl_port_async = _sock.getsockname()[1]
        _sock.close()
        nccl_init_method = f"tcp://127.0.0.1:{nccl_port_async}"
        world_size = 2  # target (rank 0) + draft (rank 1)
        draft_rank = 1

        # Get model config for draft
        draft_model_path = server_args.speculative_draft_model_path
        target_config = target_worker.model_runner.model_config
        vocab_size = target_config.vocab_size
        dtype_str = str(target_config.dtype).replace("torch.", "")

        # Compute max_blocks from context length and page size
        context_len = target_config.context_len
        block_size = server_args.page_size
        max_blocks = (context_len + block_size - 1) // block_size

        # Dedicated port for the draft process's own distributed init (tp=1)
        _sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock2.bind(("127.0.0.1", 0))
        nccl_port_dist = _sock2.getsockname()[1]
        _sock2.close()

        # Spawn draft process with full TpModelWorker
        logger.info(
            f"Spawning async draft process on GPU {self.draft_gpu_id}, "
            f"async NCCL port {nccl_port_async}, dist port {nccl_port_dist}"
        )
        self.draft_process = mp.Process(
            target=run_draft_process,
            kwargs=dict(
                server_args=server_args,
                gpu_id=self.draft_gpu_id,
                nccl_init_method=nccl_init_method,
                nccl_port_dist=nccl_port_dist,
                spec_k=self.spec_k,
                fan_out=self.ssd_fan_out,
                fan_out_list=self.fan_out_list,
                fan_out_list_miss=self.fan_out_list_miss,
                vocab_size=vocab_size,
                dtype_str=dtype_str,
                max_blocks=max_blocks,
                block_size=block_size,
            ),
            daemon=True,
        )
        self.draft_process.start()

        # Init target-side NCCL process group for async communication.
        # Cannot use dist.init_process_group because SGLang already
        # initialized the default group. Use TCPStore + ProcessGroupNCCL
        # to create an independent group.
        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=nccl_port_async,
            world_size=world_size,
            is_master=True,
            wait_for_workers=False,
        )
        async_pg = dist.ProcessGroupNCCL(
            store, 0, world_size
        )

        # Create async speculator
        self.async_speculator = SSDAsyncSpeculator(
            device=torch.device(f"cuda:{gpu_id}"),
            async_pg=async_pg,
            draft_rank=draft_rank,
            spec_k=self.spec_k,
            fan_out=self.ssd_fan_out,
            vocab_size=vocab_size,
            draft_dtype=target_config.dtype,
            max_blocks=max_blocks,
        )

        # Expose target's model_runner and model_config so scheduler can access them.
        # In async mode, the draft runs in a separate process, so we proxy
        # through the target worker for metadata queries.
        self._model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config

        # For sync mode compatibility, set these but they won't be used
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        logger.info("SSD worker initialized (async mode)")

    # ------------------------------------------------------------------
    # forward_batch_generation: dispatch sync vs async
    # ------------------------------------------------------------------

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        Sync mode: delegates to EAGLEWorker (parent).
        Async mode: custom pipeline using NCCL speculator.
        """
        if not self.ssd_async:
            return super().forward_batch_generation(batch)

        # Async mode
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._async_forward_extend(batch)
        else:
            return self._async_forward_decode(batch)

    # ------------------------------------------------------------------
    # Async extend: target forward + send prefill to draft
    # ------------------------------------------------------------------

    def _async_forward_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Async extend: run target model forward, send prefill to draft process."""
        # Run target model extend (no hidden state capture needed for async draft)
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        next_token_ids = batch_result.next_token_ids

        # Send prefill to draft process
        self._send_prefill_to_draft(batch)

        # Set up spec_info for the next decode step
        # Store verified_id so the decode path knows the recovery token
        batch.spec_info = SSDDraftInput(
            verified_id=next_token_ids,
            last_accepted_lens=torch.ones(
                batch.batch_size(), dtype=torch.int64, device=self.device
            ),
        )

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=False,
        )

    def _send_prefill_to_draft(self, batch: ScheduleBatch):
        """Send CMD_PREFILL to the draft process with sequence data."""
        if self.async_speculator is None:
            return

        input_ids_list = []
        num_tokens_list = []
        block_tables_list = []

        for i, req in enumerate(batch.reqs):
            # Get input token IDs for this request
            seq_len = batch.seq_lens[i].item()
            req_pool_idx = batch.req_pool_indices[i]
            # Get tokens from the req_to_token_pool
            token_ids = batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
            input_ids_list.append(
                torch.tensor(req.fill_ids, dtype=torch.int64, device=self.device)
            )
            num_tokens_list.append(len(req.fill_ids))

            # Get block table for this request
            bt = batch.req_to_token_pool.req_to_token[req_pool_idx]
            block_tables_list.append(bt)

        # Pass req_pool_indices as seq_ids so draft can key KV caches correctly
        seq_ids_list = [
            int(batch.req_pool_indices[i].item()) for i in range(len(batch.reqs))
        ]

        self.async_speculator.prefill(
            input_ids_list=input_ids_list,
            num_tokens_list=num_tokens_list,
            block_tables_list=block_tables_list,
            seq_ids_list=seq_ids_list,
        )

    # ------------------------------------------------------------------
    # Async decode: speculate via NCCL → verify via target
    # ------------------------------------------------------------------

    def _async_forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Async decode: get draft tokens from speculator, verify with target."""
        B = batch.batch_size()
        K = self.spec_k
        device = self.device

        if batch.forward_mode.is_idle():
            # Idle batch: no work to do, return empty result
            return self._async_forward_idle(batch)

        # Extract verified_id (recovery token) from previous step
        verified_id = self._get_verified_id(batch)

        # Ensure verified_id matches batch size (may be stale after filtering)
        if verified_id.shape[0] != B:
            verified_id = verified_id[:B]

        tp_size = self.server_args.tp_size
        if self.async_speculator is not None:
            # tp_rank=0: speculate via NCCL with draft process
            seq_ids = batch.req_pool_indices.to(torch.int64)
            recovery_token_ids = verified_id.to(torch.int64)
            num_tokens = batch.seq_lens.to(torch.int64)
            temperatures = batch.sampling_info.temperatures.squeeze(-1)

            last_accepted_lens = self._get_last_accepted_lens(batch)
            if last_accepted_lens.shape[0] != B:
                last_accepted_lens = last_accepted_lens[:B]

            block_tables = batch.req_to_token_pool.req_to_token[batch.req_pool_indices]

            spec_tokens, logits_q, cache_hits = self.async_speculator.speculate(
                seq_ids=seq_ids,
                cache_key_depths=last_accepted_lens,
                recovery_token_ids=recovery_token_ids,
                num_tokens=num_tokens,
                temperatures=temperatures,
                block_tables=block_tables,
            )

            # Broadcast spec_tokens to other TP ranks so they build identical verify input
            if tp_size > 1:
                tp_group = self.target_worker.model_runner.tp_group
                dist.broadcast(spec_tokens, src=tp_group.ranks[0], group=tp_group.device_group)
                dist.broadcast(cache_hits, src=tp_group.ranks[0], group=tp_group.device_group)

            logger.debug(
                f"Async spec: B={B}, K={K}, "
                f"hits={cache_hits.sum().item()}/{B}"
            )
        else:
            # Non-master TP rank: receive broadcast spec_tokens from rank 0
            spec_tokens = torch.empty(B, K, dtype=torch.int64, device=device)
            cache_hits = torch.empty(B, dtype=torch.int64, device=device)
            tp_group = self.target_worker.model_runner.tp_group
            dist.broadcast(spec_tokens, src=tp_group.ranks[0], group=tp_group.device_group)
            dist.broadcast(cache_hits, src=tp_group.ranks[0], group=tp_group.device_group)

        # Build EagleVerifyInput for topk=1 linear chain
        spec_info = self._build_verify_input(
            verified_id=verified_id,
            spec_tokens=spec_tokens,
            batch=batch,
        )

        # Run verify (reuse parent method)
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self.verify(batch, spec_info)
        )

        # In async mode, skip forward_draft_extend_after_decode
        # (no local draft model to update)
        # batch.spec_info is already set to res.draft_input by verify()

        # Wrap batch.spec_info to preserve SSD-specific state
        self._update_spec_info_after_verify(batch, verify_output, cache_hits)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _async_forward_idle(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Handle idle batch in async mode."""
        # Run target forward on idle batch
        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def _get_verified_id(self, batch: ScheduleBatch) -> torch.Tensor:
        """Extract verified_id (recovery token) from batch spec_info."""
        spec_info = batch.spec_info
        if isinstance(spec_info, SSDDraftInput) and spec_info.verified_id is not None:
            return spec_info.verified_id
        elif isinstance(spec_info, EagleDraftInput) and spec_info.verified_id is not None:
            return spec_info.verified_id
        else:
            # Fallback: use output_ids from batch
            return batch.output_ids

    def _get_last_accepted_lens(self, batch: ScheduleBatch) -> torch.Tensor:
        """Get last accepted lengths for tree cache key depth."""
        spec_info = batch.spec_info
        if isinstance(spec_info, SSDDraftInput) and spec_info.last_accepted_lens is not None:
            return spec_info.last_accepted_lens
        # Default: 1 (first decode after extend)
        return torch.ones(
            batch.batch_size(), dtype=torch.int64, device=self.device
        )

    def _build_verify_input(
        self,
        verified_id: torch.Tensor,
        spec_tokens: torch.Tensor,
        batch: ScheduleBatch,
    ) -> EagleVerifyInput:
        """Build EagleVerifyInput for topk=1 linear speculation.

        Constructs the tree structures expected by build_tree_kernel_efficient
        for a linear chain (no branching).

        Args:
            verified_id: [B] — last accepted tokens (recovery tokens)
            spec_tokens: [B, K] — speculated tokens from async speculator
            batch: the schedule batch

        Returns:
            EagleVerifyInput ready for self.verify()
        """
        B = verified_id.shape[0]
        K = self.spec_k
        device = self.device
        # draft_token_num = K for topk=1 (1 verified + K-1 draft)
        draft_token_num = self.speculative_num_draft_tokens  # = K

        # Use K-1 spec tokens as draft (matching sync flow where
        # 1 token comes from draft extend + K-1 from draft forwards)
        draft_tokens_for_tree = spec_tokens[:, : draft_token_num - 1]

        # Build synthetic parent_list for linear chain
        # Format matches organize_draft_results output for topk=1:
        # [-1, 0, 1, 2, ..., K-2] → shape [B, K]
        parent_values = torch.arange(
            -1, draft_token_num - 1, device=device, dtype=torch.int64
        )
        parent_list = parent_values.unsqueeze(0).expand(B, -1).contiguous()

        # Build top_scores_index: [B, K-1] = [0, 1, ..., K-2]
        top_scores_index = (
            torch.arange(draft_token_num - 1, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )

        # Build tree structures via CUDA kernel
        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens_flat,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens_for_tree,
            seq_lens=batch.seq_lens,
            seq_lens_sum=batch.seq_lens.sum().item(),
            topk=1,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=draft_token_num,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens_flat,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens.sum().item(),
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def _update_spec_info_after_verify(
        self,
        batch: ScheduleBatch,
        verify_output,
        cache_hits: torch.Tensor,
    ):
        """Update batch.spec_info with SSD-specific state after verify.

        verify() sets batch.spec_info = res.draft_input (EagleDraftInput).
        We wrap it with SSD-specific fields needed for the next async decode.
        """
        eagle_draft_input = batch.spec_info
        if not isinstance(eagle_draft_input, EagleDraftInput):
            return

        # Use accept_length from draft_input (not verify_output) because
        # when requests finish, draft_input is filtered to unfinished only,
        # while verify_output.accept_length_per_req_cpu has ALL requests.
        if eagle_draft_input.accept_length_cpu is not None:
            accept_lens_cpu = eagle_draft_input.accept_length_cpu
        else:
            accept_lens_cpu = verify_output.accept_length_per_req_cpu
        accept_lens = torch.tensor(
            accept_lens_cpu, dtype=torch.int64, device=self.device,
        )

        # Extract the LAST accepted prediction per request as the recovery token.
        # eagle_draft_input.verified_id is a flat array of accepted predictions
        # for unfinished requests: [req0_t0, req0_t1, ..., req1_t0, ...].
        # Each request i has (accept_lens[i] + 1) accepted predictions.
        # The recovery token is the last one per request (the bonus token).
        all_verified = eagle_draft_input.verified_id
        if all_verified.numel() > 0 and accept_lens.numel() > 0:
            cum_lens = (accept_lens + 1).cumsum(0)
            last_indices = cum_lens - 1
            last_verified_ids = all_verified[last_indices]
        else:
            last_verified_ids = all_verified

        batch.spec_info = SSDDraftInput(
            verified_id=last_verified_ids,
            last_accepted_lens=accept_lens + 1,  # +1 for the bonus token
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean shutdown of async draft process."""
        if self.async_speculator is not None:
            self.async_speculator.shutdown()
        if self.draft_process is not None:
            self.draft_process.join(timeout=10)
            if self.draft_process.is_alive():
                self.draft_process.terminate()
            logger.info("Draft process terminated")
