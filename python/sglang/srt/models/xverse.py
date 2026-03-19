# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/xverse.py#L1
"""Inference-only XVERSE model compatible with HuggingFace weights."""

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from sglang.srt.layers.quantization.fp8_utils import _use_aiter


class XverseMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

    def _forward_with_fp8_input(
        self,
        x: torch.Tensor,
        fp8_input: torch.Tensor,
        fp8_scale: torch.Tensor,
    ) -> torch.Tensor:
        """MLP forward using pre-quantized FP8 input for gate_up_proj (AMD AITER path).

        Skips redundant per_token_quant_hip inside gemm_a8w8_bpreshuffle by
        reusing FP8 already computed by the preceding fused RMSNorm+FP8 op.
        x is the float tensor before norm, used only for output dtype/shape.
        """
        gate_up, _ = self.gate_up_proj.forward_with_fp8_input(x, fp8_input, fp8_scale)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class XverseAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_with_fp8_input(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        fp8_input: torch.Tensor,
        fp8_scale: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_o_reduce: bool = False,
    ) -> torch.Tensor:
        """Attention forward using pre-quantized FP8 input for qkv_proj (AMD AITER path).

        Skips redundant per_token_quant_hip inside gemm_a8w8_bpreshuffle by
        reusing FP8 already computed by the preceding fused RMSNorm+FP8 op.
        x is the float tensor before norm, used only for output dtype/shape.
        skip_o_reduce: if True, skip the o_proj allreduce (caller will fuse it
        with the subsequent norm via forward_with_allreduce_fusion_fp8_out).
        """
        qkv, _ = self.qkv_proj.forward_with_fp8_input(x, fp8_input, fp8_scale)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output, skip_all_reduce=skip_o_reduce)
        return output


class XverseDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.self_attn = XverseAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = XverseMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Check if fused RMSNorm+FP8 quantization path is available (AMD AITER).
        if _use_aiter:
            from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsLinearMethod,
            )
            from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
            from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8LinearMethod
            from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
            from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod

            qm = getattr(self.self_attn.qkv_proj, "quant_method", None)
            # For scheme-based methods (CompressedTensors, Quark), check the underlying
            # scheme signals FP8 prequantization support.
            if isinstance(qm, (CompressedTensorsLinearMethod, QuarkLinearMethod)):
                scheme = getattr(self.self_attn.qkv_proj, "scheme", None)
                self._aiter_fp8 = hasattr(scheme, "_supports_prequantized_fp8")
            else:
                self._aiter_fp8 = isinstance(
                    qm, (W8A8Fp8LinearMethod, FBGEMMFp8LinearMethod, Fp8LinearMethod)
                )
        else:
            self._aiter_fp8 = False

    def _forward_aiter_fp8(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoder forward with fused add+RMSNorm+FP8 quantization (AMD AITER path).

        Replaces separate add+norm+per_token_quant (~26µs) with fused op (~13µs),
        saving ~13µs per norm call per layer on MI300X.  fp8_input/scale from
        forward_aiter_fp8_out() are passed directly to the first linear of each
        sub-layer via forward_with_fp8_input(), skipping re-quantization.

        For TP>1, also attempts to fuse the o_proj allreduce with the subsequent
        post_attention_layernorm via forward_with_allreduce_fusion_fp8_out(),
        saving an additional ~14µs per layer when AITER custom-AR is available.
        """
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        # Self Attention: fused add+norm+fp8 into qkv_proj
        if residual is None:
            residual = hidden_states
            fp8_hs, fp8_scale, _ = self.input_layernorm.forward_aiter_fp8_out(
                hidden_states
            )
        else:
            fp8_hs, fp8_scale, residual = self.input_layernorm.forward_aiter_fp8_out(
                hidden_states, residual
            )
        # Skip o_proj allreduce; attempt fused allreduce+add+norm+fp8 quant.
        hidden_states = self.self_attn._forward_with_fp8_input(
            positions, hidden_states, fp8_hs, fp8_scale, forward_batch,
            skip_o_reduce=True,
        )
        fused = self.post_attention_layernorm.forward_with_allreduce_fusion_fp8_out(
            hidden_states, residual
        )
        if fused is not None:
            # fused = (fp8_out, residual_out, scale_out)
            fp8_hs, residual, fp8_scale = fused
        else:
            # Fallback: explicit allreduce (no-op for TP=1) + fused norm+quant
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            fp8_hs, fp8_scale, residual = self.post_attention_layernorm.forward_aiter_fp8_out(
                hidden_states, residual
            )

        # Fully Connected: fused add+norm+fp8 into gate_up_proj
        hidden_states = self.mlp._forward_with_fp8_input(hidden_states, fp8_hs, fp8_scale)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # _aiter_fp8 may be True even when input_layernorm has been replaced with
        # an identity (e.g. Eagle layer_id=0); guard with isinstance.
        if self._aiter_fp8 and isinstance(self.input_layernorm, RMSNorm):
            return self._forward_aiter_fp8(positions, hidden_states, forward_batch, residual)

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class XverseModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                XverseDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            # print(f"layer[{i}].hidden_states: {hidden_states}")
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class XverseForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = XverseModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], name=None, loaded_weight=None
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        def load_weights_per_param(name, loaded_weight):
            if "rotary_emb.inv_freq" in name or "projector" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                return
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.startswith("model.vision_tower") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    return
                if name.startswith("model.vision_tower") and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        if name is None or loaded_weight is None:
            for name, loaded_weight in weights:
                load_weights_per_param(name, loaded_weight)
        else:
            load_weights_per_param(name, loaded_weight)


EntryClass = XverseForCausalLM
