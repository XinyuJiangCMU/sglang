from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_ptpc_linear,
    cutlass_fp8_supported,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe as aiter_fused_moe
    from aiter.ops.shuffle import shuffle_weight


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading
    - If CUTLASS is supported: Per-channel weight quantization is used
    - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = (
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
        )
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return W8A8Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8FP8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Fp8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Fp8Config):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = layer.weight_scale.detach()
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            if _is_fp8_fnuz:
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=weight_scale
                )

            if _use_aiter:
                layer.weight = Parameter(
                    shuffle_weight(weight.contiguous(), (16, 16)),
                    requires_grad=False,
                )
            else:
                layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            # If checkpoint not offline quantized, quantize the weights.
            if self.cutlass_fp8_supported or _use_aiter:
                # Per-channel quantization: cutlass requires it,
                # AITER uses it for fast gemm_a8w8_bpreshuffle.
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
                )
                # weight_scale is (N, 1) from per_token_group_quant_fp8.
                # AITER (gemm_a8w8_bpreshuffle) expects (N, 1) - keep as-is.
                # CUTLASS expects a flat vector; transpose to (1, N) for fp8_scaled_mm compat.
                if not _use_aiter:
                    weight_scale = weight_scale.t().contiguous()
            else:
                # per-tensor quantization fallback
                qweight, weight_scale = input_to_float8(layer.weight, dtype=fp8_dtype)

            # Update the layer with the new values.
            if _use_aiter:
                # AITER gemm_a8w8_bpreshuffle expects (N, K) preshuffled weight + (N, 1) scale.
                layer.weight = Parameter(
                    shuffle_weight(qweight.contiguous(), (16, 16)),
                    requires_grad=False,
                )
            else:
                layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # For checkpoint FP8, always allocate as float8_e4m3fn so bytes are
        # loaded verbatim.  On AMD, normalize_e4m3fn_to_e4m3fnuz in
        # process_weights_after_loading safely bitcasts to float8_e4m3fnuz.
        # fp8_dtype (fnuz on AMD) would NaN-ify values in [240, 448] on copy_.
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)
        else:
            layer.weight_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if _use_aiter:
            return apply_fp8_ptpc_linear(
                x,
                layer.weight,
                layer.weight_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
            )
        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )


class W8A8FP8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # WEIGHTS
        # Always allocate as float8_e4m3fn so checkpoint bytes are loaded via
        # bitwise copy (no value conversion).  On AMD we normalize to
        # float8_e4m3fnuz in process_weights_after_loading, which correctly
        # handles the NaN-bit pattern and doubles the scale.  Allocating as
        # float8_e4m3fnuz directly would cause values 240-448 to become NaN.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if is_fp8_fnuz():
            # Weights were loaded as float8_e4m3fn (bitwise copy from checkpoint).
            # Normalize to float8_e4m3fnuz: zero the NaN bit-pattern and double
            # the scale to account for the different bit-value mapping.
            w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                layer.w13_weight, layer.w13_weight_scale
            )
            w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale
            )
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(w13_weight_scale, requires_grad=False)
            layer.w2_weight_scale = Parameter(w2_weight_scale, requires_grad=False)
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(
                layer.w13_weight_scale.data, requires_grad=False
            )
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data, requires_grad=False
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        if _use_aiter:
            # AITER fused MoE: per-token dynamic FP8 quantization, much faster
            # than Triton on AMD MI300X for w8a8_fp8 inference
            if self.moe_runner_config.apply_router_weight_on_input:
                _, topk = topk_weights.shape
                assert (
                    topk == 1
                ), "apply_router_weight_on_input only supported with topk=1 in AITER path"
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)
            output = aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=(
                    ActivationType.Silu
                    if self.moe_runner_config.activation == "silu"
                    else ActivationType.Gelu
                ),
                quant_type=QuantType.per_Token,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
