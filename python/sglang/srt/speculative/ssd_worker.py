"""SSD (Speculative Speculative Decoding) Worker.

Phase 1 (sync): Sequential draft + verify on same GPU.
  - Inherits EAGLEWorker's draft/verify pipeline with topk=1 (linear speculation).
  - Loads a full standalone draft model on the same GPU as target.

Phase 3 (async): Draft on separate GPU with NCCL + tree cache.
  - Spawns SSDDraftRunner as a separate process on its own GPU.
  - Target communicates with draft via SSDAsyncSpeculator (NCCL process group).
  - Tree cache enables speculative pre-computation across decode steps.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
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
        """
        from sglang.srt.speculative.ssd_async_speculator import SSDAsyncSpeculator
        from sglang.srt.speculative.ssd_draft_runner import run_draft_process

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

        # Determine draft GPU (last visible GPU)
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_gpus:
            gpu_list = [int(g) for g in visible_gpus.split(",")]
            self.draft_gpu_id = len(gpu_list) - 1  # last GPU in visible set
        else:
            self.draft_gpu_id = server_args.tp_size  # GPU after TP group

        # NCCL init for async process group
        nccl_port_async = nccl_port + 100  # offset to avoid conflict
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

        # Spawn draft process
        logger.info(
            f"Spawning async draft process on GPU {self.draft_gpu_id}, "
            f"NCCL port {nccl_port_async}"
        )
        self.draft_process = mp.Process(
            target=run_draft_process,
            kwargs=dict(
                draft_model_path=draft_model_path,
                gpu_id=self.draft_gpu_id,
                nccl_init_method=nccl_init_method,
                world_size=world_size,
                rank=draft_rank,
                spec_k=self.spec_k,
                fan_out=self.ssd_fan_out,
                fan_out_list=self.fan_out_list,
                fan_out_list_miss=self.fan_out_list_miss,
                vocab_size=vocab_size,
                dtype_str=dtype_str,
                max_blocks=max_blocks,
                block_size=block_size,
                jit_speculate=True,
            ),
            daemon=True,
        )
        self.draft_process.start()

        # Init target-side NCCL (rank 0)
        dist.init_process_group(
            backend="nccl",
            init_method=nccl_init_method,
            world_size=world_size,
            rank=0,
        )
        async_pg = dist.new_group(ranks=[0, draft_rank])

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

    def shutdown(self):
        """Clean shutdown of async draft process."""
        if self.async_speculator is not None:
            self.async_speculator.shutdown()
        if self.draft_process is not None:
            self.draft_process.join(timeout=10)
            if self.draft_process.is_alive():
                self.draft_process.terminate()
            logger.info("Draft process terminated")
