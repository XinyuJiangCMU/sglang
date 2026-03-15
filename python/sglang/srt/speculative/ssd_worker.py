"""SSD (Speculative Speculative Decoding) Worker.

Phase 1: Sync SD — sequential draft + verify on same GPU.
Inherits EAGLEWorker's draft/verify pipeline with topk=1 (linear speculation).
Phase 3 will add async mode with NCCL + tree cache on separate GPU.
"""

import logging
from typing import Optional

import torch

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

    Phase 1 (sync): Reuses EAGLEWorker's draft/verify pipeline with topk=1.
    - Loads a full standalone draft model (e.g. Llama-1B) on same GPU as target
    - Draft K tokens sequentially, target verifies in one forward pass
    - Draft and target share req_to_token_pool but have separate KV caches

    Phase 3 (async): Will add separate-GPU draft with NCCL + tree cache.
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

        # Parse fan-out lists (for Phase 3 async tree cache)
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

        # Initialize like StandaloneWorker (full draft model, shared allocator)
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

        logger.info("SSD worker initialized successfully")
