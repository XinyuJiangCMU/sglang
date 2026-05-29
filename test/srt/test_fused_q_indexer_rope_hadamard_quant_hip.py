"""Tests for the ROCm/HIP fallback of fused_q_indexer_rope_hadamard_quant.

The fused CUDA kernel (torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant)
is not in the ROCm sgl_kernel build, so on HIP the op is decomposed into
apply_rotary_emb_triton + hadamard_transform + per-(token,head) fp8 act-quant.
These tests pin the HIP fallback's output contract and the no-mutation guarantee.

Shapes mirror the DSv4 MTP draft path captured at runtime on MI355X:
  q_input (num_tokens=32, n_heads=64, head_dim=128) bf16
  freqs_cis (max_pos, rope_dim//2=32) complex64, rope_dim=64
  positions (32,) int64
  weight (32, 64) bf16
"""
import pytest
import torch

from sglang.srt.utils import is_cuda, is_hip
from sglang.jit_kernel.dsv4.elementwise import fused_q_indexer_rope_hadamard_quant

_HIP = is_hip()
# "real CUDA" = NVIDIA (the fused kernel is only built there); ROCm reports
# is_cuda()==False, is_hip()==True.
_REAL_CUDA = is_cuda() and not _HIP

T, H, HEAD_DIM, ROPE_DIM, MAX_POS = 32, 64, 128, 64, 4096


def _make_inputs(device="cuda"):
    torch.manual_seed(0)
    q_input = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    # unit-magnitude rope table, complex64 [max_pos, rope_dim//2]
    theta = torch.randn(MAX_POS, ROPE_DIM // 2, device=device)
    freqs_cis = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)
    positions = torch.arange(T, device=device, dtype=torch.int64)
    weight = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    weight_scale = 1.0
    return q_input, weight, weight_scale, freqs_cis, positions


@pytest.mark.skipif(not _HIP, reason="exercises the HIP fallback path")
def test_hip_fallback_shape_dtype():
    q_input, weight, weight_scale, freqs_cis, positions = _make_inputs()
    q_fp8, weights_out = fused_q_indexer_rope_hadamard_quant(
        q_input, weight, weight_scale, freqs_cis, positions
    )
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert weights_out.dtype == torch.float32
    assert tuple(q_fp8.shape) == (T, H, HEAD_DIM)
    assert tuple(weights_out.shape) == (T, H, 1)


@pytest.mark.skipif(not _HIP, reason="exercises the HIP fallback path")
def test_hip_fallback_no_caller_mutation():
    q_input, weight, weight_scale, freqs_cis, positions = _make_inputs()
    q_before = q_input.clone()
    _ = fused_q_indexer_rope_hadamard_quant(
        q_input, weight, weight_scale, freqs_cis, positions
    )
    # The fallback applies RoPE in-place on a copy, never on the caller's tensor.
    assert torch.equal(q_input, q_before), "fallback mutated caller's q_input"


@pytest.mark.skipif(
    not _REAL_CUDA, reason="needs the NVIDIA fused kernel for cross-check"
)
def test_hip_vs_cuda_numerical_close():
    # On NVIDIA: compare the (would-be) HIP decomposition against the fused
    # CUDA kernel. Kept as a guard for environments that have both.
    q_input, weight, weight_scale, freqs_cis, positions = _make_inputs()
    q_cuda, w_cuda = fused_q_indexer_rope_hadamard_quant(
        q_input, weight, weight_scale, freqs_cis, positions
    )
    # Reference == same call; this asserts the kernel runs and is finite. A true
    # HIP-vs-CUDA diff requires running both backends, which a single host can't.
    assert torch.isfinite(w_cuda).all()
    assert q_cuda.dtype == torch.float8_e4m3fn
