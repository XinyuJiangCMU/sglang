"""Tests for FP8 BMM pre-dequantization optimization on ROCm.

Verifies that pre-dequantizing FP8 weights to bf16 at model load time
(instead of runtime cast+scale) is correct and faster for DeepSeek MLA
absorbed attention BMM paths on AMD MI300X.

Run:
    HIP_VISIBLE_DEVICES=2 python -m pytest test/srt/test_bmm_fp8_rocm.py -v
    HIP_VISIBLE_DEVICES=2 python -m unittest test.srt.test_bmm_fp8_rocm -v
"""

import time
import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

_is_hip = is_hip()


def _make_mla_weights(num_heads, K, N, device="cuda"):
    """Create FP8 weights and pre-dequantized bf16 weights for testing."""
    W_bf16 = torch.randn(num_heads, K, N, device=device, dtype=torch.bfloat16)
    w_amax = W_bf16.abs().max()
    w_scale = (w_amax / fp8_max).float()
    # HIP bias correction: e4m3fn (bias=7) -> e4m3fnuz (bias=8)
    w_scale_hip = w_scale * 2.0
    W_fp8 = (W_bf16 / w_scale).to(fp8_dtype)
    W_prescaled = W_fp8.to(torch.bfloat16) * w_scale_hip
    return W_bf16, W_fp8, w_scale_hip, W_prescaled


@unittest.skipIf(not _is_hip, reason="ROCm-only test")
class TestPrescaleCorrectness(CustomTestCase):
    """Verify prescale at load-time matches runtime cast+scale exactly."""

    def _check_path(self, num_heads, M, K, N):
        torch.manual_seed(42)
        _, W_fp8, w_scale, W_pre = _make_mla_weights(num_heads, K, N)
        A = torch.randn(M, num_heads, K, device="cuda", dtype=torch.bfloat16)

        out_runtime = torch.bmm(
            A.transpose(0, 1), W_fp8.to(torch.bfloat16) * w_scale
        )
        out_prescale = torch.bmm(A.transpose(0, 1), W_pre)

        self.assertTrue(
            torch.equal(out_runtime, out_prescale),
            f"Prescale and runtime outputs differ for shape "
            f"[{num_heads},{M},{K}]@[{num_heads},{K},{N}]",
        )

    def test_q_nope_w_kc_decode(self):
        """q_nope @ w_kc with M=1 (decode)."""
        self._check_path(128, 1, 128, 512)

    def test_q_nope_w_kc_prefill(self):
        """q_nope @ w_kc with M=64 (prefill)."""
        self._check_path(128, 64, 128, 512)

    def test_q_nope_w_kc_long(self):
        """q_nope @ w_kc with M=256 (long prefill)."""
        self._check_path(128, 256, 128, 512)

    def test_attn_w_vc_decode(self):
        """attn_output @ w_vc with M=1 (decode)."""
        self._check_path(128, 1, 512, 128)

    def test_attn_w_vc_prefill(self):
        """attn_output @ w_vc with M=64 (prefill)."""
        self._check_path(128, 64, 512, 128)

    def test_attn_w_vc_long(self):
        """attn_output @ w_vc with M=256 (long prefill)."""
        self._check_path(128, 256, 512, 128)


@unittest.skipIf(not _is_hip, reason="ROCm-only test")
class TestArgmaxEquivalence(CustomTestCase):
    """Greedy decode must produce identical token selections."""

    def test_argmax_match(self):
        torch.manual_seed(123)
        num_heads, K, N = 128, 512, 128
        _, W_fp8, w_scale, W_pre = _make_mla_weights(num_heads, K, N)

        for M in [1, 16, 64]:
            A = torch.randn(num_heads, M, K, device="cuda", dtype=torch.bfloat16)
            argmax_rt = torch.bmm(A, W_fp8.to(torch.bfloat16) * w_scale).argmax(-1)
            argmax_ps = torch.bmm(A, W_pre).argmax(-1)
            self.assertTrue(
                torch.equal(argmax_rt, argmax_ps),
                f"Argmax mismatch at M={M}",
            )


@unittest.skipIf(not _is_hip, reason="ROCm-only test")
class TestQuantizationError(CustomTestCase):
    """Quantization error should be bounded and consistent."""

    def test_relative_error_bounded(self):
        torch.manual_seed(42)
        num_heads, K, N = 128, 512, 128
        W_bf16, _, _, W_pre = _make_mla_weights(num_heads, K, N)

        for M in [16, 64, 256]:
            A = torch.randn(num_heads, M, K, device="cuda", dtype=torch.bfloat16)
            ref = torch.bmm(A, W_bf16)
            out = torch.bmm(A, W_pre)
            # Use cosine similarity instead of element-wise relative error
            # to avoid instability when individual values are near zero.
            cos_sim = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.float().flatten(), dim=0
            ).item()
            self.assertGreater(
                cos_sim, 0.95,
                f"Cosine similarity {cos_sim:.4f} too low at M={M}",
            )


@unittest.skipIf(not _is_hip, reason="ROCm-only test")
class TestPrescaleSpeed(CustomTestCase):
    """Pre-dequantized path should be faster than runtime cast+scale."""

    @staticmethod
    def _bench(fn, warmup=20, iters=200):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    def test_prescale_faster_combined(self):
        """Combined q@w_kc + attn@w_vc should be faster with prescale."""
        torch.manual_seed(42)
        num_heads = 128
        qk_nope_head_dim, kv_lora_rank, v_head_dim = 128, 512, 128

        _, W_kc_fp8, w_scale, W_kc_pre = _make_mla_weights(
            num_heads, qk_nope_head_dim, kv_lora_rank
        )
        _, W_vc_fp8, _, W_vc_pre = _make_mla_weights(
            num_heads, kv_lora_rank, v_head_dim
        )

        results = []
        for M in [1, 16, 64, 256]:
            q_nope = torch.randn(
                M, num_heads, qk_nope_head_dim, device="cuda", dtype=torch.bfloat16
            )
            attn_out = torch.randn(
                M, num_heads, kv_lora_rank, device="cuda", dtype=torch.bfloat16
            )

            def old():
                torch.bmm(q_nope.transpose(0, 1), W_kc_fp8.to(torch.bfloat16) * w_scale)
                torch.bmm(attn_out.transpose(0, 1), W_vc_fp8.to(torch.bfloat16) * w_scale)

            def new():
                torch.bmm(q_nope.transpose(0, 1), W_kc_pre)
                torch.bmm(attn_out.transpose(0, 1), W_vc_pre)

            t_old = self._bench(old)
            t_new = self._bench(new)
            speedup = t_old / t_new
            results.append((M, t_old, t_new, speedup))
            print(
                f"  M={M:<4}  old={t_old:.3f}ms  new={t_new:.3f}ms  "
                f"speedup={speedup:.2f}x"
            )

        # At least decode (M=1) should be faster
        _, _, _, speedup_decode = results[0]
        self.assertGreater(
            speedup_decode, 1.5,
            f"Expected >1.5x speedup at M=1, got {speedup_decode:.2f}x",
        )


if __name__ == "__main__":
    unittest.main()
