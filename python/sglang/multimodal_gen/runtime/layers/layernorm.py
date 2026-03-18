# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level AITER layer_norm cache to avoid repeated import overhead.
# Shared by FP32LayerNorm and LayerNorm to resolve the forward-declaration order.
_aiter_layer_norm_fn = None
_aiter_layer_norm_checked: bool = False


def _get_aiter_layer_norm_fn():
    """Lazily import and return aiter.layer_norm, or None if unavailable."""
    global _aiter_layer_norm_fn, _aiter_layer_norm_checked
    if not _aiter_layer_norm_checked:
        _aiter_layer_norm_checked = True
        try:
            import aiter
            _aiter_layer_norm_fn = aiter.layer_norm
        except (ImportError, AttributeError):
            _aiter_layer_norm_fn = None
    return _aiter_layer_norm_fn
from sgl_kernel import fused_add_rmsnorm, rmsnorm

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.triton_ops import (
    fuse_scale_shift_kernel,
    norm_infer,
    rms_norm_fn,
)
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var


# Copied and adapted from sglang
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
            self._forward_method = self.forward_native

    def forward_triton(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        return rms_norm_fn(
            x, self.weight, bias=None, residual=residual, eps=self.variance_epsilon
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if residual is not None:
            residual_shape = residual.shape
            residual = residual.view(-1, shape[-1])

        if x.dtype == torch.float:
            # fp32
            out = self.forward_triton(x, residual)
        elif self.variance_size_override is not None:
            return self.forward_native(x, residual)
        elif residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x.view(shape), residual.view(residual_shape)
        else:
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        out = out.view(shape)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(x, residual)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Use AITER optimized RMSNorm on AMD ROCm (7x faster than native for bf16).
        # Fall back to native for variance_size_override (AITER normalizes all dims).
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        try:
            from aiter import rms_norm as aiter_rms_norm
            from aiter import rmsnorm2d_fwd_with_add as aiter_fused_add_rms_norm
        except ImportError:
            return self.forward_native(x, residual)

        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        # AITER kernels require weight dtype to match input dtype
        weight = self.weight.data.to(x.dtype)

        if residual is not None and x.dtype in (torch.float16, torch.bfloat16):
            # AITER fused residual+norm (only supports fp16/bf16)
            residual_shape = residual.shape
            residual_2d = residual.view(-1, shape[-1])
            if not residual_2d.is_contiguous():
                residual_2d = residual_2d.contiguous()
            out = torch.empty_like(x_2d)
            residual_out = torch.empty_like(residual_2d)
            aiter_fused_add_rms_norm(
                out, x_2d, residual_2d, residual_out,
                weight, self.variance_epsilon,
            )
            return out.view(shape), residual_out.view(residual_shape)
        else:
            out = aiter_rms_norm(x_2d, weight, self.variance_epsilon)
            if residual is not None:
                # Fallback: compute residual manually for non-fp16/bf16 dtypes
                return out.view(shape), (x_2d.view(shape) + residual).to(x.dtype)
            return out.view(shape)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# Copied and adapted from sglang
@CustomOp.register("layer_norm")
class LayerNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps=1e-5,
        bias: bool = True,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.bias = (
                torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
                if bias
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            # Lazy cache for ones vector (not a registered buffer to avoid FSDP/meta issues)
            self._weight_fallback_cache = None

    def _get_weight_fallback(self, x: torch.Tensor) -> torch.Tensor:
        wf = getattr(self, "_weight_fallback_cache", None)
        if (
            wf is None
            or wf.device != x.device
            or wf.dtype != x.dtype
            or wf.numel() != self.hidden_size
        ):
            wf = torch.ones(self.hidden_size, device=x.device, dtype=x.dtype)
            self._weight_fallback_cache = wf
        return wf

    def forward_triton(self, x: torch.Tensor):
        # Fast inference kernel without residual/dropout branches
        return norm_infer(
            x.view(-1, self.hidden_size),
            self.weight,
            self.bias,
            eps=self.eps,
            is_rms_norm=False,
        ).view(x.shape)

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.view(-1, self.hidden_size)
        return self.forward_triton(x).view(shape)

    @torch.compile(backend="inductor")
    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = self.weight * x
        # if no affine, this is a no-op
        if self.bias is not None:
            x = x + self.bias
        return x.to(input_dtype)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # On AMD ROCm, prefer AITER BF16 layernorm over Triton for bf16 inputs.
        # AITER is ~5x faster than F.layer_norm FP32 and also faster than Triton.
        if x.dtype == torch.bfloat16:
            aiter_fn = _get_aiter_layer_norm_fn()
            if aiter_fn is not None:
                shape = x.shape
                x_2d = x.view(-1, self.hidden_size)
                if not x_2d.is_contiguous():
                    x_2d = x_2d.contiguous()
                if self.weight is not None:
                    w = self.weight.to(x.dtype)
                    b = self.bias.to(x.dtype) if self.bias is not None else None
                    if b is None:
                        b = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
                else:
                    w, b = self._get_aiter_id_params(x_2d)
                return aiter_fn(x_2d, w, b, self.eps).view(shape)
        return self.forward_cuda(x)

    def _get_aiter_id_params(self, x_2d: torch.Tensor):
        """Return cached identity weight/bias for elementwise_affine=False."""
        cache = getattr(self, "_hip_id_cache", None)
        dev, dtype = x_2d.device, x_2d.dtype
        if cache is None or cache[0] != dev or cache[1] != dtype:
            w = torch.ones(self.hidden_size, dtype=dtype, device=dev)
            b = torch.zeros(self.hidden_size, dtype=dtype, device=dev)
            self._hip_id_cache = (dev, dtype, w, b)
            return w, b
        return cache[2], cache[3]

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class ScaleResidual(nn.Module):
    """
    Applies gated residual connection.
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward(
        self, residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Apply gated residual connection."""
        # x.shape: [batch_size, seq_len, inner_dim]
        # When gate is FP32 and x/residual are BF16, cast gate to x.dtype to
        # avoid a large [B,S,C] FP32 intermediate (~2.6x faster on MI300X).
        # Saves ~70ms per request (300 mlp_residual calls at Wan2.1 seq length).
        if gate.dtype != x.dtype:
            gate = gate.to(x.dtype)
        if gate.dim() == 4:
            # gate.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            return residual + (
                x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
            ).flatten(1, 2)
        else:
            # gate.shape: [batch_size, 1, inner_dim]
            return residual + x * gate


# adapted from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
# NOTE(will): Needed to match behavior of diffusers and wan2.1 even while using
# FSDP's MixedPrecisionPolicy
class FP32LayerNorm(nn.LayerNorm):
    def _get_aiter_identity_params(
        self, n: int, device: torch.device, dtype: torch.dtype
    ):
        """Return cached ones/zeros for elementwise_affine=False AITER calls.

        AITER layer_norm requires concrete weight/bias tensors even when there
        is no affine transform, so we cache identity tensors per-instance.
        """
        cache = getattr(self, "_aiter_id_cache", None)
        if cache is None or cache[0] != n or cache[1] != device or cache[2] != dtype:
            ones  = torch.ones(n, dtype=dtype, device=device)
            zeros = torch.zeros(n, dtype=dtype, device=device)
            self._aiter_id_cache = (n, device, dtype, ones, zeros)
            return ones, zeros
        return cache[3], cache[4]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        # On AMD ROCm, AITER BF16 layernorm is ~5x faster than F.layer_norm FP32
        # for BF16 inputs — avoids two dtype casts and uses a HIP-optimised kernel.
        # Benchmarked: 0.060 ms vs 0.314 ms at [32760, 1536] on MI300X (5.2x).
        # Saves ~152 ms total for 300 blocks (5 steps × 2 CFG × 30 blocks).
        if origin_dtype == torch.bfloat16:
            aiter_ln = _get_aiter_layer_norm_fn()
            if aiter_ln is not None:
                shape = inputs.shape
                n = self.normalized_shape[-1]
                # AITER layer_norm requires 2D input [N, C]; reshape [B, S, C].
                x_2d = inputs.view(-1, n)
                if not x_2d.is_contiguous():
                    x_2d = x_2d.contiguous()
                if self.weight is not None:
                    weight = self.weight.to(origin_dtype)
                    bias = (
                        self.bias.to(origin_dtype) if self.bias is not None
                        else torch.zeros(n, dtype=origin_dtype, device=inputs.device)
                    )
                else:
                    # elementwise_affine=False: use cached identity weight/bias
                    # (AITER requires concrete tensors, not None).
                    weight, bias = self._get_aiter_identity_params(
                        n, inputs.device, origin_dtype
                    )
                return aiter_ln(x_2d, weight, bias, self.eps).view(shape)
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class ScaleResidualLayerNormScaleShift(nn.Module):
    """
    Fused operation that combines:
    1. Gated residual connection
    2. LayerNorm
    3. Scale and shift operations

    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(
                hidden_size, has_weight=elementwise_affine, eps=eps, dtype=dtype
            )
        elif norm_type == "layer":
            if compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(
                    hidden_size, elementwise_affine=elementwise_affine, eps=eps
                )
            else:
                self.norm = LayerNorm(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                    dtype=dtype,
                )
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated residual connection, followed by layernorm and
        scale/shift in a single fused operation.

        Returns:
            Tuple containing:
            - normalized and modulated output of shape: [batch_size, seq_len, inner_dim]
            - residual value (value after residual connection
              but before normalization)
        """
        # x.shape: [batch_size, seq_len, inner_dim]
        # Apply residual connection with gating
        if isinstance(gate, int):
            # used by cross-attention, should be 1
            assert gate == 1
            residual_output = residual + x
        elif isinstance(gate, torch.Tensor):
            # Cast gate to x.dtype if they differ to avoid a large [B,S,C] FP32
            # intermediate when gate is FP32 and x is BF16 (~2.3x faster residual
            # op on MI300X). Keeping gate in BF16 also enables the AITER BF16
            # layernorm path in FP32LayerNorm, saving ~0.28ms/block × 300 blocks
            # ≈ 83ms per request for Wan2.1 at S=32760.
            if gate.dtype != x.dtype:
                gate = gate.to(x.dtype)
            if gate.dim() == 4:
                # gate.shape: [batch_size, num_frames, 1, inner_dim]
                num_frames = gate.shape[1]
                frame_seqlen = x.shape[1] // num_frames
                residual_output = residual + (
                    x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
                ).flatten(1, 2)
            else:
                # used by bidirectional self attention
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        # residual_output.shape: [batch_size, seq_len, inner_dim]

        # Apply normalization
        normalized = self.norm(residual_output)

        if scale.dim() != 4:
            # On AMD ROCm and for the common 3D case, native PyTorch is 3x faster
            # than the Triton fuse_scale_shift_kernel. The 4D case (temporal
            # conditioning in Wan2.2 TI2V) still requires the Triton kernel.
            #
            # When normalized is BF16 (from AITER layernorm) and scale/shift are
            # FP32, cast to normalized.dtype first to avoid a large [B,S,C] FP32
            # intermediate (~2x faster; saves ~50ms per request on MI300X).
            if normalized.dtype != scale.dtype:
                modulated = (normalized * (1 + scale).to(normalized.dtype)
                             + shift.to(normalized.dtype))
            else:
                modulated = normalized * (1 + scale) + shift
        else:
            modulated = fuse_scale_shift_kernel(
                normalized,
                scale,
                shift,
            )
        return modulated, residual_output


class LayerNormScaleShift(nn.Module):
    """
    Fused operation that combines LayerNorm with scale and shift operations.
    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size, has_weight=elementwise_affine, eps=eps)
        elif norm_type == "layer":
            if self.compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(
                    hidden_size, elementwise_affine=elementwise_affine, eps=eps
                )
            else:
                self.norm = nn.LayerNorm(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                    dtype=dtype,
                )
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Apply ln followed by scale and shift in a single fused operation."""
        # x.shape: [batch_size, seq_len, inner_dim]
        normalized = self.norm(x)
        if self.compute_dtype == torch.float32:
            normalized = normalized.float()

        if scale.dim() == 4:
            # scale.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = scale.shape[1]
            frame_seqlen = normalized.shape[1] // num_frames
            output = (
                normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1.0 + scale)
                + shift
            ).flatten(1, 2)
        else:
            # scale.shape: [batch_size, 1, inner_dim]
            # shift.shape: [batch_size, 1, inner_dim]
            output = normalized * (1.0 + scale) + shift

        if self.compute_dtype == torch.float32:
            output = output.to(x.dtype)

        return output


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: "RMSNorm",
    k_norm: "RMSNorm",
    head_dim: int,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply QK normalization for query and key tensors.

    Minimal multimodal_gen-only implementation: only the JIT fused inplace
    QK-norm kernel path is supported (no fallback).
    """

    batch_size = q.size(0)
    q_eps = q_norm.variance_epsilon
    k_eps = k_norm.variance_epsilon
    # Only try fused path on CUDA and when it won't introduce implicit copies.
    if (
        q.is_cuda
        and allow_inplace
        and (q_eps == k_eps)
        and can_use_fused_inplace_qknorm(head_dim)
    ):
        fused_inplace_qknorm(
            q=q.view(batch_size, -1, head_dim),
            k=k.view(batch_size, -1, head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            head_dim=head_dim,
            eps=q_eps,
        )
        return q, k

    raise RuntimeError(
        "apply_qk_norm: fused inplace QK-norm is not applicable "
        "(expected CUDA, contiguous q/k, matching eps, and supported head_dim)"
    )
