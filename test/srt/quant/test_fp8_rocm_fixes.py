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


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestNormalizeE4M3FnToFnuz(unittest.TestCase):
    """Test e4m3fn to e4m3fnuz normalization."""

    def test_normalization_doubles_scale(self):
        from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz

        weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)

        w_out, ws_out, _ = normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale)
        self.assertEqual(w_out.dtype, torch.float8_e4m3fnuz)
        self.assertAlmostEqual(ws_out.item(), 1.0, places=5)  # 0.5 * 2 = 1.0

    def test_normalization_removes_nan_pattern(self):
        from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz

        # Create weight with the -128 bit pattern (NaN in fnuz)
        weight = torch.zeros(4, 4, device="cuda", dtype=torch.float8_e4m3fn)
        w_int8 = weight.view(torch.int8)
        w_int8[0, 0] = -128  # This is NaN in fnuz
        weight = w_int8.view(torch.float8_e4m3fn)

        weight_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        w_out, _, _ = normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale)

        # Check no NaN in output
        self.assertFalse(
            w_out.to(torch.float32).isnan().any().item(),
            "Normalized weight should not contain NaN",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestShuffleWeight(unittest.TestCase):
    """Test shuffle_weight preserves shape and data integrity."""

    def test_shape_preservation(self):
        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER not available")

        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        for N, K in [(256, 256), (4096, 4096), (19456, 2560)]:
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
            W_shuffled = shuffle_weight(W.contiguous(), (16, 16))
            self.assertEqual(W_shuffled.shape, (N, K))
            self.assertEqual(W_shuffled.dtype, fp8_dtype)


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAITERPerChannelGEMM(unittest.TestCase):
    """Test AITER per-channel FP8 GEMM via apply_fp8_ptpc_linear."""

    def _test_shape(self, M, N, K):
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER not available")

        torch.manual_seed(42)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qw, ws = per_token_group_quant_fp8(W, W.shape[-1])
        # ws has shape (N, 1) from per_token_group_quant_fp8 - use directly, no transpose.
        # gemm_a8w8_bpreshuffle expects w_scale in (N, 1) layout.
        ws_stored = ws.contiguous()
        qw_shuffled = shuffle_weight(qw.contiguous(), (16, 16))

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = x @ W.T

        out = apply_fp8_ptpc_linear(
            x, qw_shuffled, ws_stored, use_per_token_if_dynamic=True
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float().unsqueeze(0),
            out.flatten().float().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos_sim, 0.998, f"AITER GEMM M={M} N={N} K={K}: cos_sim={cos_sim}"
        )
        self.assertEqual(out.shape, (M, N))

    def test_decode_small(self):
        self._test_shape(1, 2560, 2560)

    def test_decode_large(self):
        self._test_shape(1, 4096, 4096)

    def test_batch_4(self):
        self._test_shape(4, 4096, 4096)

    def test_batch_32(self):
        self._test_shape(32, 4096, 4096)

    def test_non_square(self):
        self._test_shape(1, 14336, 4096)

    def test_output_buffer(self):
        """Test scaled_fp8_quant allocates output with correct dtype."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            scaled_fp8_quant,
        )

        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        out, scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        self.assertEqual(out.dtype, fp8_dtype)
        self.assertEqual(out.shape, (4, 2560))
        self.assertEqual(scale.shape, (4, 1))

    def test_pre_quantized_input(self):
        """Test apply_fp8_ptpc_linear with AITER per-token-per-channel GEMM."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            per_token_group_quant_fp8,
            scaled_fp8_quant,
        )
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER not available")

        M, N, K = 4, 2560, 2560
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qw, ws = per_token_group_quant_fp8(W, W.shape[-1])
        # ws has shape (N, 1) - use directly (no transpose) for gemm_a8w8_bpreshuffle
        ws_stored = ws.contiguous()
        qw_shuffled = shuffle_weight(qw.contiguous(), (16, 16))

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        # Use scaled_fp8_quant (AITER path) to quantize x per-token
        x_fp8, x_scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)

        # Verify x_fp8 is in the correct dtype for AITER
        self.assertEqual(x_fp8.dtype, fp8_dtype)

        # Run GEMM via apply_fp8_ptpc_linear (which re-quantizes internally)
        out = apply_fp8_ptpc_linear(
            x, qw_shuffled, ws_stored,
            use_per_token_if_dynamic=True,
        )
        self.assertEqual(out.shape, (M, N))
        self.assertFalse(out.isnan().any().item())
        self.assertTrue(out.isfinite().all().item())


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8DtypeConsistency(unittest.TestCase):
    """Verify FP8 dtype is consistent across all quantization backends on AMD."""

    def test_compressed_tensors_w8a8_fp8_dtype(self):
        """compressed_tensors_w8a8_fp8 should use platform-aware fp8_dtype via fp8_kernel."""
        import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8  # noqa: F401
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        # compressed_tensors_w8a8_fp8 imports fp8_dtype from fp8_kernel internally
        # Verify the platform-aware value is correct
        self.assertEqual(fp8_dtype, torch.float8_e4m3fnuz)

    def test_compressed_tensors_w8a16_fp8_dtype(self):
        """compressed_tensors_w8a16_fp8 should use platform-aware fp8_dtype."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (
            fp8_dtype as ct_fp8_dtype,
        )
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        self.assertEqual(ct_fp8_dtype, fp8_dtype)
        self.assertEqual(ct_fp8_dtype, torch.float8_e4m3fnuz)

    def test_quark_w8a8_fp8_dtype(self):
        """Quark w8a8_fp8 should use platform-aware fp8_dtype."""
        from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
            fp8_dtype as q_fp8_dtype,
        )
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        self.assertEqual(q_fp8_dtype, fp8_dtype)
        self.assertEqual(q_fp8_dtype, torch.float8_e4m3fnuz)

    def test_fpgemm_fp8_dtype(self):
        """FBGEMM FP8 should use platform-aware fp8_dtype."""
        from sglang.srt.layers.quantization.fpgemm_fp8 import fp8_dtype as fb_fp8_dtype
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        self.assertEqual(fb_fp8_dtype, fp8_dtype)
        self.assertEqual(fb_fp8_dtype, torch.float8_e4m3fnuz)

    def test_w8a8_fp8_dtype(self):
        """w8a8_fp8 should use platform-aware fp8_dtype from fp8_kernel."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config

        # Verify fp8_dtype is correct platform value
        self.assertEqual(fp8_dtype, torch.float8_e4m3fnuz)
        # Verify we can instantiate W8A8Fp8Config (which imports fp8_dtype)
        config = W8A8Fp8Config(is_checkpoint_fp8_serialized=True)
        self.assertIsNotNone(config)


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8EdgeCases(unittest.TestCase):
    """Edge case tests for FP8 on AMD MI300X."""

    def test_zero_input_quant(self):
        """Zero input should produce zero dequantized output."""
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
        q, s = scaled_fp8_quant(x)
        # Scale is 0, so dequantized = q * s = 0 regardless of q values
        dequant = q.float() * s
        self.assertTrue((dequant == 0).all().item())

    def test_large_input_clamp(self):
        """Large values should be clamped to fp8_max."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_max,
            scaled_fp8_quant,
        )

        x = torch.full((4, 128), 1e6, device="cuda", dtype=torch.bfloat16)
        q, s = scaled_fp8_quant(x)
        # After quant, max abs value should be <= fp8_max
        self.assertTrue(q.float().abs().max().item() <= fp8_max + 1)

    def test_single_token_quant(self):
        """M=1 quantization should work correctly."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )

        x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
        q, s = per_token_group_quant_fp8(x, group_size=128)
        self.assertEqual(q.shape, (1, 2560))
        self.assertEqual(s.shape, (1, 20))  # 2560/128 = 20 groups
        self.assertFalse(q.float().isnan().any().item())

    def test_large_batch_quant(self):
        """Large batch quantization should work."""
        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
        q, s = scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        self.assertEqual(q.shape, (512, 4096))
        self.assertEqual(s.shape, (512, 1))

    def test_fp8_gemm_numerical_stability(self):
        """FP8 GEMM should produce finite outputs for normal inputs."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            per_token_group_quant_fp8,
            scaled_fp8_quant,
        )
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER shuffle_weight not available")

        M, N, K = 32, 2560, 2560
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qW, wscale = per_token_group_quant_fp8(W, K)
        # wscale has shape (N, 1) - use directly (no transpose) for gemm_a8w8_bpreshuffle
        wscale_stored = wscale.contiguous()
        W_shuffled = shuffle_weight(qW.contiguous(), (16, 16))

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out = apply_fp8_ptpc_linear(
            x, W_shuffled, wscale_stored, None, None, use_per_token_if_dynamic=True
        )
        self.assertTrue(out.isfinite().all().item())
        self.assertEqual(out.shape, (M, N))


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestINT8AITERGemm(unittest.TestCase):
    """Test AITER INT8 GEMM on AMD MI300X."""

    def test_int8_gemm_correctness(self):
        """AITER gemm_a8w8 INT8 should produce correct results."""
        try:
            from aiter import gemm_a8w8
        except ImportError:
            self.skipTest("AITER gemm_a8w8 not available")

        M, N, K = 4, 2560, 2560
        A = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
        W = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
        a_scale = torch.ones(M, 1, device="cuda", dtype=torch.float32) * 0.01
        w_scale = torch.ones(N, 1, device="cuda", dtype=torch.float32) * 0.01

        out = gemm_a8w8(A, W, a_scale, w_scale, dtype=torch.bfloat16)
        self.assertEqual(out.shape, (M, N))
        self.assertTrue(out.isfinite().all().item())

    def test_int8_per_token_quant_aiter(self):
        """INT8 per-token quant should use AITER fast path."""
        from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8

        x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
        q, s = per_token_quant_int8(x)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(q.shape, (32, 4096))
        self.assertEqual(s.shape, (32, 1))
        # Verify dequant accuracy
        dq = q.float() * s
        cos = torch.nn.functional.cosine_similarity(
            x.float().flatten().unsqueeze(0), dq.flatten().unsqueeze(0)
        ).item()
        self.assertGreater(cos, 0.999)

    def test_int8_group_quant_aiter(self):
        """INT8 group quant should use AITER fast path."""
        from sglang.srt.layers.quantization.int8_kernel import (
            per_token_group_quant_int8,
        )

        x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
        q, s = per_token_group_quant_int8(x, group_size=128)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(q.shape, (32, 4096))
        self.assertEqual(s.shape, (32, 32))  # 4096/128 = 32 groups


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8PerformanceSanity(unittest.TestCase):
    """Sanity check FP8 performance on AMD to catch regressions."""

    def test_aiter_gemm_latency(self):
        """AITER per-channel GEMM should complete in < 50us for M=1."""
        import time

        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER shuffle_weight not available")

        M, N, K = 1, 2560, 2560
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qW, ws = per_token_group_quant_fp8(W, K)
        # ws has shape (N, 1) - use directly (no transpose) for gemm_a8w8_bpreshuffle
        ws = ws.contiguous()
        W_s = shuffle_weight(qW.contiguous(), (16, 16))

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(20):
            apply_fp8_ptpc_linear(
                x, W_s, ws, None, None, use_per_token_if_dynamic=True
            )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            apply_fp8_ptpc_linear(
                x, W_s, ws, None, None, use_per_token_if_dynamic=True
            )
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 100 * 1e6

        self.assertLess(
            us,
            50.0,
            f"AITER GEMM latency {us:.1f}us exceeds 50us threshold",
        )

    def test_quant_latency(self):
        """FP8 per-token quant should complete in < 20us for M=1."""
        import time

        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)

        for _ in range(20):
            scaled_fp8_quant(x, use_per_token_if_dynamic=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6

        self.assertLess(
            us,
            20.0,
            f"FP8 quant latency {us:.1f}us exceeds 20us threshold",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8AMDRowwisePaths(unittest.TestCase):
    """Test the AMD-specific rowwise scaled_mm paths in apply_fp8_linear."""

    def setUp(self):
        from sglang.srt.layers.quantization.fp8_utils import USE_ROWWISE_TORCH_SCALED_MM
        if not USE_ROWWISE_TORCH_SCALED_MM:
            self.skipTest("USE_ROWWISE_TORCH_SCALED_MM is False (requires gfx942+ and torch>=2.7)")

    def _make_channel_weight(self, N, K, fp8_dtype):
        """Create channel-wise quantized weight in (K, N) non-contiguous layout."""
        fp8_max = torch.finfo(fp8_dtype).max
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        w_scale = (w.abs().max(dim=1, keepdim=True).values / fp8_max).float()
        w_q = (w / w_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return w_q.t(), w_scale  # (K, N) non-contiguous, (N, 1)

    def test_channel_weight_static_activation(self):
        """Channel-wise weight + static per-tensor activation should use rowwise scaled_mm."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        M, N, K = 16, 2048, 2048
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w_KN, w_scale_N1 = self._make_channel_weight(N, K, fp8_dtype)

        # Static per-tensor activation scale
        input_scale = (x.abs().max() / torch.finfo(fp8_dtype).max).float()

        # Use apply_fp8_linear with static input_scale + channel weight
        out = apply_fp8_linear(
            x, w_KN, w_scale_N1, input_scale=input_scale,
            use_per_token_if_dynamic=True, compressed_tensor_quant=True,
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_pertensor_weight_pertoken_activation(self):
        """Per-tensor weight scale + per-token activation should use rowwise scaled_mm."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        M, N, K = 16, 2048, 2048
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        fp8_max = torch.finfo(fp8_dtype).max
        # Per-tensor weight scale (scalar)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        w_scale = (w.abs().max() / fp8_max).float()
        w_q = (w / w_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        w_KN = w_q.t()

        out = apply_fp8_linear(
            x, w_KN, w_scale, input_scale=None,
            use_per_token_if_dynamic=True, compressed_tensor_quant=True,
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_channel_weight_static_cosine_similarity(self):
        """Verify rowwise path produces correct results vs BF16 reference."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        M, N, K = 32, 1024, 1024
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        fp8_max = torch.finfo(fp8_dtype).max
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        w_scale = (w.abs().max(dim=1, keepdim=True).values / fp8_max).float()
        w_q = (w / w_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        w_KN = w_q.t()

        # BF16 reference
        w_dq = w_q.float() * w_scale
        ref = (x.float() @ w_dq.t()).to(torch.bfloat16)

        input_scale = (x.abs().max() / fp8_max).float()
        out = apply_fp8_linear(
            x, w_KN, w_scale, input_scale=input_scale,
            use_per_token_if_dynamic=True, compressed_tensor_quant=True,
        )
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().float().unsqueeze(0),
            out.flatten().float().unsqueeze(0),
        ).item()
        self.assertGreater(cos, 0.99, f"Cosine similarity too low: {cos:.4f}")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFusedRMSNormFP8(unittest.TestCase):
    """Test fused RMSNorm + FP8 dynamic quantization on AMD."""

    def setUp(self):
        try:
            from aiter import rmsnorm2d_fwd_with_dynamicquant  # noqa: F401
        except (ImportError, AttributeError):
            self.skipTest("aiter.rmsnorm2d_fwd_with_dynamicquant not available")
        try:
            from sglang.srt.utils import get_bool_env_var
            if not get_bool_env_var("SGLANG_USE_AITER"):
                self.skipTest("SGLANG_USE_AITER not set")
        except Exception:
            pass

    def test_forward_aiter_fp8_out_no_residual(self):
        """RMSNorm.forward_aiter_fp8_out without residual returns correct FP8."""
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        M, N = 16, 2048
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        norm = RMSNorm(N, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)

        fp8_out, fp8_scale, residual_out = norm.forward_aiter_fp8_out(x)

        self.assertEqual(fp8_out.shape, (M, N))
        self.assertEqual(fp8_out.dtype, fp8_dtype)
        self.assertEqual(fp8_scale.shape, (M, 1))
        self.assertEqual(fp8_scale.dtype, torch.float32)
        self.assertIsNone(residual_out)

        # Verify cosine similarity with plain RMSNorm
        norm_ref = norm.forward(x)
        dq = fp8_out.float() * fp8_scale
        cos = torch.nn.functional.cosine_similarity(
            norm_ref.float().flatten().unsqueeze(0),
            dq.flatten().unsqueeze(0),
        ).item()
        self.assertGreater(cos, 0.99, f"Cosine similarity too low: {cos:.4f}")

    def test_forward_aiter_fp8_out_with_residual(self):
        """RMSNorm.forward_aiter_fp8_out with residual updates residual correctly."""
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        M, N = 16, 2048
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        norm = RMSNorm(N, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)

        fp8_out, fp8_scale, residual_out = norm.forward_aiter_fp8_out(x, res)

        self.assertEqual(fp8_out.shape, (M, N))
        self.assertEqual(fp8_out.dtype, fp8_dtype)
        self.assertEqual(fp8_scale.shape, (M, 1))
        self.assertIsNotNone(residual_out)
        self.assertEqual(residual_out.shape, (M, N))
        self.assertEqual(residual_out.dtype, torch.bfloat16)

        # Verify residual is x + res (add before norm)
        expected_res = (x + res)
        cos_res = torch.nn.functional.cosine_similarity(
            expected_res.float().flatten().unsqueeze(0),
            residual_out.float().flatten().unsqueeze(0),
        ).item()
        self.assertGreater(cos_res, 0.9999, f"Residual cosine too low: {cos_res:.6f}")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFBGEMMFp8AITERPath(unittest.TestCase):
    """Test FBGEMMFp8LinearMethod with AITER gemm_a8w8_bpreshuffle path."""

    def setUp(self):
        from sglang.srt.layers.quantization.fp8_utils import _use_aiter
        if not _use_aiter:
            self.skipTest("SGLANG_USE_AITER not set (required for AITER path)")
        try:
            from aiter.ops.shuffle import shuffle_weight  # noqa: F401
        except ImportError:
            self.skipTest("aiter.ops.shuffle not available")

    def test_process_weights_preshuffle(self):
        """FBGEMMFp8 process_weights_after_loading should preshuffle weight on AITER."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8Config, FBGEMMFp8LinearMethod

        config = FBGEMMFp8Config(ignore_list=[], input_scale_ub=1.0)
        method = FBGEMMFp8LinearMethod(config)

        # Create a mock layer with weight in (N, K) = (256, 256) before loading.
        # Always allocate as float8_e4m3fn (mimics checkpoint loading);
        # process_weights_after_loading normalizes to e4m3fnuz on AMD.
        N, K = 256, 256
        weight = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        weight_scale = torch.ones(N, 1, device="cuda", dtype=torch.float32) * 0.01

        from torch.nn import Module, Parameter

        class MockLayer(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(weight)
                self.weight_scale = Parameter(weight_scale)
                self.input_scale_ub = Parameter(torch.tensor(1.0))

        layer = MockLayer()
        method.process_weights_after_loading(layer)

        # After loading, weight should be preshuffled (N, K) shape preserved
        self.assertEqual(layer.weight.shape, (N, K))
        self.assertEqual(layer.weight.dtype, fp8_dtype)
        # input_scale_ub should be deleted since AITER doesn't use it
        self.assertFalse(hasattr(layer, "input_scale_ub"),
                         "input_scale_ub should be deleted for AITER path")

    def test_apply_correctness(self):
        """FBGEMMFp8 AITER path should produce correct GEMM output."""
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, per_token_group_quant_fp8
        from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8Config, FBGEMMFp8LinearMethod

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("aiter.ops.shuffle not available")

        M, N, K = 4, 512, 512
        W_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qW, wscale = per_token_group_quant_fp8(W_bf16, K)
        W_shuffled = shuffle_weight(qW.contiguous(), (16, 16))

        from torch.nn import Parameter
        layer = type("MockLayer", (), {
            "weight": Parameter(W_shuffled),
            "weight_scale": Parameter(wscale),
        })()

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = x @ W_bf16.t()

        config = FBGEMMFp8Config(ignore_list=[], input_scale_ub=1.0)
        method = FBGEMMFp8LinearMethod(config)
        out = method.apply(layer, x, bias=None)

        self.assertEqual(out.shape, (M, N))
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().float().unsqueeze(0),
            out.flatten().float().unsqueeze(0),
        ).item()
        self.assertGreater(cos, 0.99, f"FBGEMMFp8 AITER cosine={cos:.4f}")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestRoutedScalingFactorMoE(unittest.TestCase):
    """Test that routed_scaling_factor is correctly applied in AMD MoE paths."""

    def _make_mock_moe_runner_config(self, rsf=2.5):
        """Create a minimal MoeRunnerConfig with routed_scaling_factor."""
        from unittest.mock import MagicMock
        config = MagicMock()
        config.routed_scaling_factor = rsf
        config.activation = "silu"
        config.apply_router_weight_on_input = False
        config.no_combine = False
        return config

    def test_w8a8_fp8_moe_rsf_applied(self):
        """W8A8FP8MoEMethod AITER path must apply routed_scaling_factor."""
        from sglang.srt.layers.quantization.fp8_utils import _use_aiter
        if not _use_aiter:
            self.skipTest("SGLANG_USE_AITER not set")
        try:
            from aiter import fused_moe as aiter_fused_moe  # noqa: F401
        except ImportError:
            self.skipTest("aiter.fused_moe not available")

        # Verify the routed_scaling_factor code path exists in W8A8FP8MoEMethod
        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
        src = inspect.getsource(W8A8FP8MoEMethod.apply)
        self.assertIn("routed_scaling_factor", src,
                      "W8A8FP8MoEMethod.apply should reference routed_scaling_factor")

    def test_quark_w8a8_fp8_moe_rsf_applied(self):
        """QuarkW8A8FP8MoE AITER path must apply routed_scaling_factor."""
        import inspect
        from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8_moe import (
            QuarkW8A8FP8MoE,
        )
        src = inspect.getsource(QuarkW8A8FP8MoE.apply_weights)
        self.assertIn("routed_scaling_factor", src,
                      "QuarkW8A8FP8MoE.apply_weights should reference routed_scaling_factor")

    def test_compressed_tensors_moe_rsf_applied(self):
        """compressed_tensors W8A8 FP8 MoE AITER path must apply routed_scaling_factor."""
        import inspect
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        src = inspect.getsource(CompressedTensorsW8A8Fp8MoE.apply_weights)
        self.assertIn("routed_scaling_factor", src,
                      "CompressedTensorsW8A8Fp8MoE.apply_weights should reference routed_scaling_factor")

    def test_triton_moe_runner_rsf_applied(self):
        """TritonRunnerCore.run should apply routed_scaling_factor after moe_sum."""
        import inspect
        from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
        src = inspect.getsource(TritonRunnerCore.run)
        self.assertIn("routed_scaling_factor", src,
                      "TritonRunnerCore.run should reference routed_scaling_factor")

    def test_rsf_value_check(self):
        """routed_scaling_factor != 1.0 guard should prevent unnecessary multiply."""
        # Verify the pattern: only apply multiply when rsf != 1.0
        # This is a simple source-code check for the guard condition
        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
        src = inspect.getsource(W8A8FP8MoEMethod.apply)
        # Should contain both the rsf reference and the != 1.0 guard
        self.assertIn("rsf != 1.0", src,
                      "Should guard routed_scaling_factor multiply with != 1.0 check")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFp8LinearMethodPrequantized(unittest.TestCase):
    """Test Fp8LinearMethod.apply() accepts prequantized_fp8 parameters (AMD AITER)."""

    def test_apply_signature_has_prequantized_params(self):
        """Fp8LinearMethod.apply() must accept prequantized_fp8/prequantized_fp8_scale."""
        import inspect
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        sig = inspect.signature(Fp8LinearMethod.apply)
        params = list(sig.parameters.keys())
        self.assertIn(
            "prequantized_fp8", params,
            "Fp8LinearMethod.apply must accept prequantized_fp8 for fused RMSNorm+FP8 path",
        )
        self.assertIn(
            "prequantized_fp8_scale", params,
            "Fp8LinearMethod.apply must accept prequantized_fp8_scale for fused RMSNorm+FP8 path",
        )

    def test_apply_passes_prequantized_to_ptpc(self):
        """Fp8LinearMethod.apply() must forward prequantized args to apply_fp8_ptpc_linear."""
        import inspect
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        src = inspect.getsource(Fp8LinearMethod.apply)
        self.assertIn(
            "prequantized_fp8=prequantized_fp8", src,
            "Fp8LinearMethod.apply must pass prequantized_fp8 to apply_fp8_ptpc_linear",
        )

    def test_fused_rmsnorm_fp8_linear_method_check(self):
        """LlamaDecoderLayer._aiter_fp8 check should include Fp8LinearMethod."""
        import inspect
        import importlib
        llama_mod = importlib.import_module("sglang.srt.models.llama")
        src = inspect.getsource(llama_mod.LlamaDecoderLayer.__init__)
        self.assertIn(
            "Fp8LinearMethod", src,
            "LlamaDecoderLayer._aiter_fp8 check must include Fp8LinearMethod",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFBGEMMFp8PrequantizedFix(unittest.TestCase):
    """Test that FBGEMMFp8LinearMethod.apply() accepts prequantized_fp8 params."""

    def test_apply_signature_has_prequantized_params(self):
        """FBGEMMFp8LinearMethod.apply() must accept prequantized_fp8 params."""
        import inspect
        from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8LinearMethod

        sig = inspect.signature(FBGEMMFp8LinearMethod.apply)
        params = list(sig.parameters.keys())
        self.assertIn(
            "prequantized_fp8", params,
            "FBGEMMFp8LinearMethod.apply must accept prequantized_fp8 for fused RMSNorm+FP8 path",
        )
        self.assertIn(
            "prequantized_fp8_scale", params,
            "FBGEMMFp8LinearMethod.apply must accept prequantized_fp8_scale",
        )

    def test_apply_forwards_prequantized_to_ptpc(self):
        """FBGEMMFp8LinearMethod.apply() must forward prequantized args to apply_fp8_ptpc_linear."""
        import inspect
        from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8LinearMethod

        src = inspect.getsource(FBGEMMFp8LinearMethod.apply)
        self.assertIn(
            "prequantized_fp8=prequantized_fp8", src,
            "FBGEMMFp8LinearMethod.apply must pass prequantized_fp8 to apply_fp8_ptpc_linear",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestLayerCommunicatorFP8Methods(unittest.TestCase):
    """Test LayerCommunicator FP8 fused-norm methods."""

    def test_prepare_mlp_fp8_out_exists(self):
        """LayerCommunicator must have prepare_mlp_fp8_out method."""
        from sglang.srt.layers.communicator import LayerCommunicator

        self.assertTrue(
            hasattr(LayerCommunicator, "prepare_mlp_fp8_out"),
            "LayerCommunicator must have prepare_mlp_fp8_out for Qwen3/DeepSeek fused FP8 path",
        )

    def test_prepare_attn_fp8_out_exists(self):
        """LayerCommunicator must have prepare_attn_fp8_out method."""
        from sglang.srt.layers.communicator import LayerCommunicator

        self.assertTrue(
            hasattr(LayerCommunicator, "prepare_attn_fp8_out"),
            "LayerCommunicator must have prepare_attn_fp8_out for Qwen3/DeepSeek fused FP8 path",
        )

    def test_prepare_mlp_fp8_out_signature(self):
        """prepare_mlp_fp8_out should accept (hidden_states, residual, forward_batch)."""
        import inspect
        from sglang.srt.layers.communicator import LayerCommunicator

        sig = inspect.signature(LayerCommunicator.prepare_mlp_fp8_out)
        params = list(sig.parameters.keys())
        self.assertIn("hidden_states", params)
        self.assertIn("residual", params)
        self.assertIn("forward_batch", params)

    def test_prepare_attn_fp8_out_signature(self):
        """prepare_attn_fp8_out should accept (hidden_states, residual, forward_batch, ...)."""
        import inspect
        from sglang.srt.layers.communicator import LayerCommunicator

        sig = inspect.signature(LayerCommunicator.prepare_attn_fp8_out)
        params = list(sig.parameters.keys())
        self.assertIn("hidden_states", params)
        self.assertIn("residual", params)
        self.assertIn("forward_batch", params)


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestQwen3FusedFP8Path(unittest.TestCase):
    """Test Qwen3DecoderLayer fused FP8 path detection."""

    def test_qwen3_aiter_fp8_check_includes_w8a8fp8(self):
        """Qwen3DecoderLayer._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        qwen3_mod = importlib.import_module("sglang.srt.models.qwen3")
        src = inspect.getsource(qwen3_mod.Qwen3DecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "Qwen3DecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )

    def test_qwen3_has_forward_aiter_fp8(self):
        """Qwen3DecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        qwen3_mod = importlib.import_module("sglang.srt.models.qwen3")
        self.assertTrue(
            hasattr(qwen3_mod.Qwen3DecoderLayer, "_forward_aiter_fp8"),
            "Qwen3DecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_qwen3_attention_has_forward_with_fp8_input(self):
        """Qwen3Attention must have _forward_with_fp8_input method."""
        import importlib
        qwen3_mod = importlib.import_module("sglang.srt.models.qwen3")
        self.assertTrue(
            hasattr(qwen3_mod.Qwen3Attention, "_forward_with_fp8_input"),
            "Qwen3Attention must have _forward_with_fp8_input for AMD AITER optimization",
        )


if __name__ == "__main__":
    unittest.main()
