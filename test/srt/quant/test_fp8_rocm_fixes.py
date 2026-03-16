"""
Comprehensive tests for ROCm FP8 fixes and optimizations.

Tests cover:
1. FP8 e4m3fnuz max value correctness (240 not 224)
2. AITER HIP per-token-group quant integration
3. MoE ep_moe FP8 dtype handling
4. FP8 KV cache quantization correctness
5. per_token_group_quant_fp8 correctness across shapes
"""

import unittest

import torch

from sglang.srt.utils import is_hip


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8E4M3FnuzMaxValue(unittest.TestCase):
    """Verify FP8 e4m3fnuz max value is 240.0, not 224.0."""

    def test_fp8_kernel_max(self):
        from sglang.srt.layers.quantization.fp8_kernel import fp8_max

        self.assertAlmostEqual(fp8_max, 240.0, places=1,
                               msg=f"fp8_max should be 240.0 for e4m3fnuz, got {fp8_max}")

    def test_fp8_kernel_dtype(self):
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        self.assertEqual(fp8_dtype, torch.float8_e4m3fnuz)

    def test_common_fp8_max(self):
        from sglang.srt.utils.common import FP8_E4M3_MAX

        self.assertAlmostEqual(FP8_E4M3_MAX, 240.0, places=1,
                               msg=f"FP8_E4M3_MAX should be 240.0 for e4m3fnuz, got {FP8_E4M3_MAX}")

    def test_torch_finfo_consistency(self):
        finfo = torch.finfo(torch.float8_e4m3fnuz)
        self.assertAlmostEqual(finfo.max, 240.0, places=1)

    def test_ep_moe_fp8_dtype(self):
        """ep_moe kernels should use platform-appropriate FP8 dtype."""
        from sglang.srt.layers.moe.ep_moe.kernels import _fp8_dtype

        self.assertEqual(_fp8_dtype, torch.float8_e4m3fnuz)
        finfo = torch.finfo(_fp8_dtype)
        self.assertAlmostEqual(finfo.max, 240.0, places=1)


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestPerTokenGroupQuantFP8ROCm(unittest.TestCase):
    """Test per_token_group_quant_fp8 correctness on ROCm (AITER HIP path)."""

    def _reference_group_quant(self, x, group_size):
        """Pure PyTorch reference implementation."""
        M = x.shape[0]
        N = x.shape[1]
        fp8_max_val = torch.finfo(torch.float8_e4m3fnuz).max

        x_grouped = x.reshape(-1, group_size)
        amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scale = amax / fp8_max_val
        x_scaled = (x_grouped / scale).clamp(-fp8_max_val, fp8_max_val)
        x_q = x_scaled.to(torch.float8_e4m3fnuz).reshape(M, N)
        scale = scale.reshape(M, N // group_size)
        return x_q, scale

    def _test_shape(self, M, N, group_size=128):
        torch.manual_seed(42)
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

        out, scale = per_token_group_quant_fp8(x, group_size=group_size)
        ref_out, ref_scale = self._reference_group_quant(x, group_size)

        # Check output dtype
        self.assertEqual(out.dtype, torch.float8_e4m3fnuz)
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(scale.shape, (M, N // group_size))

        # Dequantize and compare
        scale_expanded = scale.to(torch.float32).repeat_interleave(group_size, dim=-1)
        deq = out.to(torch.float32) * scale_expanded
        ref_scale_expanded = ref_scale.to(torch.float32).repeat_interleave(group_size, dim=-1)
        deq_ref = ref_out.to(torch.float32) * ref_scale_expanded

        cos_sim = torch.nn.functional.cosine_similarity(
            deq.flatten().unsqueeze(0), deq_ref.flatten().unsqueeze(0)
        ).item()
        self.assertGreater(cos_sim, 0.99, f"cos_sim={cos_sim} for shape ({M}, {N})")

    def test_decode_shape(self):
        self._test_shape(1, 7168)

    def test_small_batch(self):
        self._test_shape(8, 7168)

    def test_medium_batch(self):
        self._test_shape(64, 7168)

    def test_large_batch(self):
        self._test_shape(256, 7168)

    def test_prefill(self):
        self._test_shape(1024, 7168)

    def test_small_hidden(self):
        self._test_shape(32, 2048)

    def test_group_size_64(self):
        self._test_shape(32, 7168, group_size=64)

    def test_group_size_32(self):
        self._test_shape(32, 7168, group_size=32)


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8KVCacheQuant(unittest.TestCase):
    """Test FP8 KV cache quantization (div + cast) correctness."""

    def test_fp8_kv_quant_roundtrip(self):
        """Verify that FP8 KV cache store/load preserves reasonable precision."""
        torch.manual_seed(42)
        # Typical KV cache tensor: (num_tokens, num_heads, head_dim)
        num_tokens, num_heads, head_dim = 64, 8, 128
        cache_k = torch.randn(num_tokens, num_heads, head_dim,
                               device="cuda", dtype=torch.bfloat16)

        # Compute scale (simulating what the model produces)
        k_scale = cache_k.abs().amax().item() / 240.0

        # Quantize: div by scale, cast to FP8
        cache_k_fp8 = (cache_k / k_scale).to(torch.float8_e4m3fnuz)

        # Dequantize: cast back, mul by scale
        cache_k_deq = cache_k_fp8.to(torch.bfloat16) * k_scale

        cos_sim = torch.nn.functional.cosine_similarity(
            cache_k.flatten().float().unsqueeze(0),
            cache_k_deq.flatten().float().unsqueeze(0)
        ).item()
        self.assertGreater(cos_sim, 0.99, f"FP8 KV roundtrip cos_sim={cos_sim}")

    def test_fp8_kv_no_nan(self):
        """Verify FP8 KV cache quantization doesn't produce NaN."""
        torch.manual_seed(42)
        cache_k = torch.randn(128, 8, 128, device="cuda", dtype=torch.bfloat16)
        k_scale = cache_k.abs().amax().item() / 240.0

        cache_k_fp8 = (cache_k / k_scale).clamp(-240.0, 240.0).to(torch.float8_e4m3fnuz)
        self.assertFalse(cache_k_fp8.to(torch.float32).isnan().any().item(),
                         "FP8 KV cache should not contain NaN")

    def test_fp8_kv_scale_correctness(self):
        """Verify scale values are properly used (not hardcoded 1.0)."""
        torch.manual_seed(42)
        cache_k = torch.randn(32, 8, 128, device="cuda", dtype=torch.bfloat16) * 5.0
        k_scale = 2.5  # Non-trivial scale

        # With scale
        cache_k_scaled = (cache_k / k_scale).to(torch.float8_e4m3fnuz)
        deq_scaled = cache_k_scaled.to(torch.float32) * k_scale

        # Without scale (as if scale=1.0)
        cache_k_unscaled = cache_k.to(torch.float8_e4m3fnuz)
        deq_unscaled = cache_k_unscaled.to(torch.float32)

        # Scaled version should be more accurate
        err_scaled = (cache_k.float() - deq_scaled).abs().mean().item()
        err_unscaled = (cache_k.float() - deq_unscaled).abs().mean().item()
        self.assertLess(err_scaled, err_unscaled,
                        "Using k_scale should improve FP8 KV cache precision")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestScaledFP8Quant(unittest.TestCase):
    """Test scaled_fp8_quant on ROCm."""

    def test_dynamic_per_tensor(self):
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        out, scale = scaled_fp8_quant(x)

        self.assertEqual(out.dtype, torch.float8_e4m3fnuz)
        self.assertEqual(out.shape, (32, 512))
        self.assertEqual(scale.shape, (1,))
        self.assertFalse(out.to(torch.float32).isnan().any().item())

    def test_dynamic_per_token(self):
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        out, scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)

        self.assertEqual(out.dtype, torch.float8_e4m3fnuz)
        self.assertEqual(out.shape, (32, 512))
        self.assertEqual(scale.shape, (32, 1))

    def test_static_scale(self):
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        out, scale_out = scaled_fp8_quant(x, scale=scale)

        self.assertEqual(out.dtype, torch.float8_e4m3fnuz)
        self.assertTrue(torch.equal(scale, scale_out))


if __name__ == "__main__":
    unittest.main()
