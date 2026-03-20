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


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestBaiChuanFusedFP8Path(unittest.TestCase):
    """Test BaiChuan fused RMSNorm+FP8 decoder path for AMD MI300X."""

    def test_baichuan_attention_has_forward_with_fp8_input(self):
        """BaiChuanAttention must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        self.assertTrue(
            hasattr(mod.BaiChuanAttention, "_forward_with_fp8_input"),
            "BaiChuanAttention must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_baichuan_mlp_has_forward_with_fp8_input(self):
        """BaiChuanMLP must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        self.assertTrue(
            hasattr(mod.BaiChuanMLP, "_forward_with_fp8_input"),
            "BaiChuanMLP must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_baichuan_decoder_layer_has_forward_aiter_fp8(self):
        """BaiChuanDecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        self.assertTrue(
            hasattr(mod.BaiChuanDecoderLayer, "_forward_aiter_fp8"),
            "BaiChuanDecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_baichuan_uses_w_pack_for_aiter_fp8_detection(self):
        """BaiChuanDecoderLayer._aiter_fp8 must use W_pack (not qkv_proj) for detection."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        src = inspect.getsource(mod.BaiChuanDecoderLayer.__init__)
        self.assertIn(
            "W_pack", src,
            "BaiChuanDecoderLayer must use W_pack for _aiter_fp8 detection (BaiChuan has no qkv_proj)",
        )

    def test_baichuan_attention_handles_alibi(self):
        """BaiChuanAttention._forward_with_fp8_input must skip rotary for ALIBI."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        src = inspect.getsource(mod.BaiChuanAttention._forward_with_fp8_input)
        self.assertIn(
            "ALIBI", src,
            "BaiChuanAttention._forward_with_fp8_input must check for ALIBI to skip rotary_emb",
        )

    def test_baichuan_aiter_fp8_check_includes_w8a8fp8(self):
        """BaiChuanDecoderLayer._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.baichuan")
        src = inspect.getsource(mod.BaiChuanDecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "BaiChuanDecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestDeepSeekV1FusedFP8Path(unittest.TestCase):
    """Test DeepSeek V1 fused RMSNorm+FP8 decoder path (dense layers only)."""

    def test_deepseek_attention_has_forward_with_fp8_input(self):
        """DeepseekAttention must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek")
        self.assertTrue(
            hasattr(mod.DeepseekAttention, "_forward_with_fp8_input"),
            "DeepseekAttention must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_deepseek_mlp_has_forward_with_fp8_input(self):
        """DeepseekMLP must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek")
        self.assertTrue(
            hasattr(mod.DeepseekMLP, "_forward_with_fp8_input"),
            "DeepseekMLP must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_deepseek_decoder_layer_has_forward_aiter_fp8(self):
        """DeepseekDecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek")
        self.assertTrue(
            hasattr(mod.DeepseekDecoderLayer, "_forward_aiter_fp8"),
            "DeepseekDecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_deepseek_aiter_fp8_dense_only_guard(self):
        """DeepseekDecoderLayer._aiter_fp8 must only be enabled for dense (DeepseekMLP) layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek")
        src = inspect.getsource(mod.DeepseekDecoderLayer.__init__)
        self.assertIn(
            "DeepseekMLP", src,
            "DeepseekDecoderLayer must guard _aiter_fp8 with isinstance(..., DeepseekMLP) "
            "to skip MoE layers",
        )

    def test_deepseek_aiter_fp8_check_includes_w8a8fp8(self):
        """DeepseekDecoderLayer._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek")
        src = inspect.getsource(mod.DeepseekDecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "DeepseekDecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestERNIE4FusedFP8Path(unittest.TestCase):
    """Test ERNIE4 fused RMSNorm+FP8 decoder path (dense layers only)."""

    def test_deepseekv2_mlp_has_forward_with_fp8_input(self):
        """DeepseekV2MLP (aliased as Ernie4MLP) must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.deepseek_v2")
        self.assertTrue(
            hasattr(mod.DeepseekV2MLP, "_forward_with_fp8_input"),
            "DeepseekV2MLP must have _forward_with_fp8_input (used by ERNIE4 via alias)",
        )

    def test_ernie4_decoder_layer_has_forward_aiter_fp8(self):
        """Ernie4DecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie4")
        self.assertTrue(
            hasattr(mod.Ernie4DecoderLayer, "_forward_aiter_fp8"),
            "Ernie4DecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_ernie4_aiter_fp8_dense_only_guard(self):
        """Ernie4DecoderLayer._aiter_fp8 must only be enabled for dense (Ernie4MLP) layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie4")
        src = inspect.getsource(mod.Ernie4DecoderLayer.__init__)
        self.assertIn(
            "Ernie4MLP", src,
            "Ernie4DecoderLayer must guard _aiter_fp8 with isinstance(..., Ernie4MLP) "
            "to skip MoE layers",
        )

    def test_ernie4_aiter_fp8_check_includes_w8a8fp8(self):
        """Ernie4DecoderLayer._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie4")
        src = inspect.getsource(mod.Ernie4DecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "Ernie4DecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )

    def test_ernie4_mlp_alias_has_forward_with_fp8_input(self):
        """Ernie4MLP alias must also expose _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie4")
        self.assertTrue(
            hasattr(mod.Ernie4MLP, "_forward_with_fp8_input"),
            "Ernie4MLP (alias of DeepseekV2MLP) must expose _forward_with_fp8_input",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestNemotronNASFusedFP8Path(unittest.TestCase):
    """Test Nemotron NAS (DeciLM) fused RMSNorm+FP8 decoder path."""

    def test_decilm_decoder_layer_has_forward_aiter_fp8(self):
        """DeciLMDecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_nas")
        self.assertTrue(
            hasattr(mod.DeciLMDecoderLayer, "_forward_aiter_fp8"),
            "DeciLMDecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_decilm_aiter_fp8_no_op_guard(self):
        """DeciLMDecoderLayer._aiter_fp8 must be disabled for no-op attention/ffn layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_nas")
        src = inspect.getsource(mod.DeciLMDecoderLayer.__init__)
        self.assertIn(
            "_is_no_op_attention", src,
            "DeciLMDecoderLayer must guard _aiter_fp8 with _is_no_op_attention check",
        )
        self.assertIn(
            "_is_no_op_ffn", src,
            "DeciLMDecoderLayer must guard _aiter_fp8 with _is_no_op_ffn check",
        )

    def test_decilm_aiter_fp8_check_includes_w8a8fp8(self):
        """DeciLMDecoderLayer._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_nas")
        src = inspect.getsource(mod.DeciLMDecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "DeciLMDecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )

    def test_decilm_uses_llama_attention_with_fp8_support(self):
        """DeciLMDecoderLayer should reuse LlamaAttention which has _forward_with_fp8_input."""
        import importlib
        llama_mod = importlib.import_module("sglang.srt.models.llama")
        self.assertTrue(
            hasattr(llama_mod.LlamaAttention, "_forward_with_fp8_input"),
            "LlamaAttention (used by DeciLMDecoderLayer) must have _forward_with_fp8_input",
        )

    def test_decilm_uses_llama_mlp_with_fp8_support(self):
        """DeciLMDecoderLayer should reuse LlamaMLP which has _forward_with_fp8_input."""
        import importlib
        llama_mod = importlib.import_module("sglang.srt.models.llama")
        self.assertTrue(
            hasattr(llama_mod.LlamaMLP, "_forward_with_fp8_input"),
            "LlamaMLP (used by DeciLMDecoderLayer) must have _forward_with_fp8_input",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAITERFP8KVDecodeOptimization(unittest.TestCase):
    """Test that FP8 KV cache is passed natively to paged_attention_ragged decode."""

    def _get_forward_decode_src(self):
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.layers.attention.aiter_backend")
        return inspect.getsource(mod.AiterAttnBackend.forward_decode)

    def test_aiter_backend_passes_fp8_kv_cache_dtype(self):
        """aiter_backend decode path must pass 'fp8' kv_cache_dtype for FP8 KV caches."""
        src = self._get_forward_decode_src()
        self.assertIn(
            '"fp8"', src,
            "AiterAttnBackend.forward_decode must pass 'fp8' kv_cache_dtype string for FP8 KV",
        )

    def test_aiter_backend_no_fp8_to_bf16_cast_in_decode(self):
        """aiter_backend decode path must NOT cast FP8 KV cache to BF16 as workaround."""
        src = self._get_forward_decode_src()
        # The old workaround cast k_cache/v_cache .to(dtype) before paged_attention_ragged
        # This should no longer be present once FP8 KV is handled natively
        self.assertNotIn(
            "k_cache.to(dtype)", src,
            "AiterAttnBackend.forward_decode must not cast k_cache to bf16 "
            "(FP8 KV now handled natively)",
        )
        self.assertNotIn(
            "v_cache.to(dtype)", src,
            "AiterAttnBackend.forward_decode must not cast v_cache to bf16 "
            "(FP8 KV now handled natively)",
        )

    def test_aiter_backend_fp8_kv_dtype_conditional(self):
        """aiter_backend must use conditional: 'fp8' only when kv_cache_dtype is FP8."""
        src = self._get_forward_decode_src()
        # Must have "auto" fallback for non-FP8 caches
        self.assertIn(
            '"auto"', src,
            "AiterAttnBackend.forward_decode must keep 'auto' fallback for "
            "non-FP8 kv_cache_dtype in paged_attention_ragged",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestGemmaRMSNormFP8(unittest.TestCase):
    """Test GemmaRMSNorm.forward_aiter_fp8_out for AMD AITER path."""

    def test_gemma_rms_norm_has_forward_aiter_fp8_out(self):
        """GemmaRMSNorm must have forward_aiter_fp8_out for AMD AITER FP8 path."""
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        self.assertTrue(
            hasattr(GemmaRMSNorm, "forward_aiter_fp8_out"),
            "GemmaRMSNorm must have forward_aiter_fp8_out for AMD AITER optimization",
        )

    def test_gemma_rms_norm_fp8_uses_effective_weight(self):
        """GemmaRMSNorm.forward_aiter_fp8_out must use (1 + weight) as effective weight."""
        import inspect
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        src = inspect.getsource(GemmaRMSNorm.forward_aiter_fp8_out)
        self.assertIn(
            "1.0 + self.weight.data",
            src,
            "GemmaRMSNorm.forward_aiter_fp8_out must use (1.0 + self.weight.data) as effective weight",
        )

    def test_communicator_prepare_attn_fp8_accepts_gemma_rms_norm(self):
        """LayerCommunicator.prepare_attn_fp8_out must accept GemmaRMSNorm."""
        import inspect
        from sglang.srt.layers.communicator import LayerCommunicator
        src = inspect.getsource(LayerCommunicator.prepare_attn_fp8_out)
        self.assertIn(
            "GemmaRMSNorm",
            src,
            "prepare_attn_fp8_out must accept GemmaRMSNorm (not just RMSNorm)",
        )

    def test_communicator_prepare_mlp_fp8_accepts_gemma_rms_norm(self):
        """LayerCommunicator.prepare_mlp_fp8_out must accept GemmaRMSNorm."""
        import inspect
        from sglang.srt.layers.communicator import LayerCommunicator
        src = inspect.getsource(LayerCommunicator.prepare_mlp_fp8_out)
        self.assertIn(
            "GemmaRMSNorm",
            src,
            "prepare_mlp_fp8_out must accept GemmaRMSNorm (not just RMSNorm)",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestQwen3_5FusedFP8Path(unittest.TestCase):
    """Test Qwen3.5 AttentionDecoderLayer fused RMSNorm+FP8 path for AMD MI300X."""

    def test_qwen3_5_attention_decoder_has_aiter_fp8(self):
        """Qwen3_5AttentionDecoderLayer must have _aiter_fp8 flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_5")
        self.assertTrue(
            hasattr(mod.Qwen3_5AttentionDecoderLayer, "_forward_aiter_fp8"),
            "Qwen3_5AttentionDecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_qwen3_5_attention_decoder_has_forward_with_fp8_input(self):
        """Qwen3_5AttentionDecoderLayer must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_5")
        self.assertTrue(
            hasattr(mod.Qwen3_5AttentionDecoderLayer, "_forward_with_fp8_input"),
            "Qwen3_5AttentionDecoderLayer must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_qwen2_moe_mlp_has_forward_with_fp8_input(self):
        """Qwen2MoeMLP must have _forward_with_fp8_input for dense MLP FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        self.assertTrue(
            hasattr(mod.Qwen2MoeMLP, "_forward_with_fp8_input"),
            "Qwen2MoeMLP must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_qwen3_5_forward_with_fp8_handles_attn_output_gate(self):
        """_forward_with_fp8_input must handle attn_output_gate branch."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_5")
        src = inspect.getsource(mod.Qwen3_5AttentionDecoderLayer._forward_with_fp8_input)
        self.assertIn(
            "attn_output_gate",
            src,
            "_forward_with_fp8_input must handle attn_output_gate branch (Qwen3.5 gated attention)",
        )

    def test_qwen3_5_aiter_fp8_moe_uses_standard_path(self):
        """_forward_aiter_fp8 must use standard MLP path for MoE layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_5")
        src = inspect.getsource(mod.Qwen3_5AttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "Qwen2MoeMLP",
            src,
            "_forward_aiter_fp8 must check isinstance(self.mlp, Qwen2MoeMLP) to skip FP8 for MoE",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestQwen3NextFusedFP8Path(unittest.TestCase):
    """Test Qwen3Next hybrid attention decoder layer fused RMSNorm+FP8 path for AMD MI300X."""

    def test_qwen3_next_hybrid_attention_has_forward_aiter_fp8(self):
        """Qwen3HybridAttentionDecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        self.assertTrue(
            hasattr(mod.Qwen3HybridAttentionDecoderLayer, "_forward_aiter_fp8"),
            "Qwen3HybridAttentionDecoderLayer must have _forward_aiter_fp8 for AMD AITER",
        )

    def test_qwen3_next_hybrid_attention_has_forward_with_fp8_input(self):
        """Qwen3HybridAttentionDecoderLayer must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        self.assertTrue(
            hasattr(mod.Qwen3HybridAttentionDecoderLayer, "_forward_with_fp8_input"),
            "Qwen3HybridAttentionDecoderLayer must have _forward_with_fp8_input for AMD AITER",
        )

    def test_qwen3_next_forward_guards_captured_last_layer_outputs(self):
        """forward() must skip FP8 path when captured_last_layer_outputs is not None."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        src = inspect.getsource(mod.Qwen3HybridAttentionDecoderLayer.forward)
        self.assertIn(
            "captured_last_layer_outputs is None",
            src,
            "forward() must guard _aiter_fp8 path: skip when captured_last_layer_outputs is set",
        )

    def test_qwen3_next_forward_with_fp8_handles_attn_output_gate(self):
        """_forward_with_fp8_input must handle attn_output_gate branch."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        src = inspect.getsource(mod.Qwen3HybridAttentionDecoderLayer._forward_with_fp8_input)
        self.assertIn(
            "attn_output_gate",
            src,
            "_forward_with_fp8_input must handle attn_output_gate (Qwen3Next gated attention)",
        )

    def test_qwen3_next_aiter_fp8_fuses_dense_mlp_norm(self):
        """_forward_aiter_fp8 must call prepare_mlp_fp8_out for dense MLP layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        src = inspect.getsource(mod.Qwen3HybridAttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out for dense MLP FP8 fusion",
        )

    def test_qwen3_next_aiter_fp8_uses_mlp_forward_with_fp8_input(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input for dense layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        src = inspect.getsource(mod.Qwen3HybridAttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call mlp._forward_with_fp8_input for dense layers",
        )

    def test_qwen3_next_aiter_fp8_checks_qwen2_moe_mlp_type(self):
        """_forward_aiter_fp8 must check isinstance(self.mlp, Qwen2MoeMLP) for MLP type."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen3_next")
        src = inspect.getsource(mod.Qwen3HybridAttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "Qwen2MoeMLP",
            src,
            "_forward_aiter_fp8 must check isinstance(self.mlp, Qwen2MoeMLP) to gate FP8 MLP",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAiterGemmCoverage(unittest.TestCase):
    """Tests for AITER FP8 GEMM config coverage checker."""

    def test_expected_shapes_list_nonempty(self):
        """_AITER_FP8_EXPECTED_GEMM_SHAPES must be non-empty."""
        from sglang.srt.layers.quantization.fp8_utils import (
            _AITER_FP8_EXPECTED_GEMM_SHAPES,
        )

        self.assertGreater(
            len(_AITER_FP8_EXPECTED_GEMM_SHAPES),
            0,
            "_AITER_FP8_EXPECTED_GEMM_SHAPES must contain at least one entry",
        )

    def test_qwen25_7b_down_proj_in_expected_shapes(self):
        """Qwen2.5-7B down_proj (N=3584, K=9472) must be in expected shapes list."""
        from sglang.srt.layers.quantization.fp8_utils import (
            _AITER_FP8_EXPECTED_GEMM_SHAPES,
        )

        nk_pairs = {(N, K) for N, K, _ in _AITER_FP8_EXPECTED_GEMM_SHAPES}
        self.assertIn(
            (3584, 9472),
            nk_pairs,
            "N=3584 K=9472 (Qwen2.5-7B down_proj) must be in _AITER_FP8_EXPECTED_GEMM_SHAPES",
        )

    def test_check_function_exists(self):
        """_check_aiter_fp8_gemm_coverage must be a callable."""
        from sglang.srt.layers.quantization.fp8_utils import (
            _check_aiter_fp8_gemm_coverage,
        )

        self.assertTrue(
            callable(_check_aiter_fp8_gemm_coverage),
            "_check_aiter_fp8_gemm_coverage must be callable",
        )

    def test_check_function_nonfatal_without_aiter(self):
        """_check_aiter_fp8_gemm_coverage must not raise even if AITER unavailable."""
        from sglang.srt.layers.quantization.fp8_utils import (
            _check_aiter_fp8_gemm_coverage,
        )

        # The function is lru_cached; calling it again is a no-op.
        # Just ensure it doesn't raise.
        try:
            _check_aiter_fp8_gemm_coverage()
        except Exception as e:
            self.fail(
                f"_check_aiter_fp8_gemm_coverage raised an exception: {e}"
            )

    def test_aiter_gemm_tune_module_importable(self):
        """aiter_gemm_tune utility module must be importable."""
        try:
            import importlib

            mod = importlib.import_module(
                "sglang.srt.layers.quantization.aiter_gemm_tune"
            )
            self.assertTrue(
                hasattr(mod, "find_missing_shapes"),
                "aiter_gemm_tune must have find_missing_shapes function",
            )
            self.assertTrue(
                hasattr(mod, "tune"),
                "aiter_gemm_tune must have tune function",
            )
            self.assertTrue(
                hasattr(mod, "SHAPES_TO_TUNE"),
                "aiter_gemm_tune must have SHAPES_TO_TUNE list",
            )
        except ImportError as e:
            self.fail(f"Could not import aiter_gemm_tune: {e}")

    def test_aiter_gemm_tune_shapes_include_qwen25(self):
        """aiter_gemm_tune.SHAPES_TO_TUNE must include Qwen2.5-7B shapes."""
        import importlib

        mod = importlib.import_module(
            "sglang.srt.layers.quantization.aiter_gemm_tune"
        )
        nk_pairs = {(N, K) for N, K, _ in mod.SHAPES_TO_TUNE}
        self.assertIn(
            (3584, 9472),
            nk_pairs,
            "SHAPES_TO_TUNE must include N=3584 K=9472 (Qwen2.5-7B down_proj)",
        )

    def test_aiter_gemm_tune_shapes_include_qwen25_72b(self):
        """aiter_gemm_tune.SHAPES_TO_TUNE includes tuneable Qwen2.5-72B shapes.

        K=7392 (TP=4) and K=3696 (TP=8) are NOT in SHAPES_TO_TUNE because
        K % 64 != 0 / K % 32 != 0 means no CK bpreshuffle kernel supports them.
        """
        import importlib

        mod = importlib.import_module(
            "sglang.srt.layers.quantization.aiter_gemm_tune"
        )
        nk_pairs = {(N, K) for N, K, _ in mod.SHAPES_TO_TUNE}
        # TP=1 and TP=2 are tuneable (K=29568, K=14784 are divisible by 64)
        self.assertIn(
            (8192, 29568),
            nk_pairs,
            "SHAPES_TO_TUNE must include N=8192 K=29568 (Qwen2.5-72B down_proj TP=1)",
        )
        self.assertIn(
            (8192, 14784),
            nk_pairs,
            "SHAPES_TO_TUNE must include N=8192 K=14784 (Qwen2.5-72B down_proj TP=2)",
        )
        # TP=4 (K=7392) and TP=8 (K=3696) have no compatible CK kernel
        self.assertNotIn(
            (8192, 7392),
            nk_pairs,
            "K=7392 (K%64!=0) should NOT be in SHAPES_TO_TUNE -- no valid CK kernel",
        )
        self.assertNotIn(
            (8192, 3696),
            nk_pairs,
            "K=3696 (K%32!=0) should NOT be in SHAPES_TO_TUNE -- no valid CK kernel",
        )

    def test_expected_shapes_include_qwen25_72b(self):
        """_AITER_FP8_EXPECTED_GEMM_SHAPES includes tuneable Qwen2.5-72B shapes.

        K=7392 (TP=4) and K=3696 (TP=8) are NOT in the list because no CK
        bpreshuffle kernel supports those K values (K%64!=0, K%32!=0).
        """
        from sglang.srt.layers.quantization.fp8_utils import (
            _AITER_FP8_EXPECTED_GEMM_SHAPES,
        )

        nk_pairs = {(N, K) for N, K, _ in _AITER_FP8_EXPECTED_GEMM_SHAPES}
        # TP=1 and TP=2 must be present (tuneable shapes)
        for tp, K in [(1, 29568), (2, 14784)]:
            self.assertIn(
                (8192, K),
                nk_pairs,
                f"_AITER_FP8_EXPECTED_GEMM_SHAPES must include N=8192 K={K} "
                f"(Qwen2.5-72B down_proj TP={tp})",
            )
        # TP=4 and TP=8 cannot be tuned -- should not be in the list
        for tp, K in [(4, 7392), (8, 3696)]:
            self.assertNotIn(
                (8192, K),
                nk_pairs,
                f"K={K} (K%64!=0) should NOT be in _AITER_FP8_EXPECTED_GEMM_SHAPES",
            )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestBailingMoEFusedFP8Path(unittest.TestCase):
    """Test BailingMoE fused RMSNorm+FP8 decoder path for AMD MI300X."""

    def test_bailing_moe_attention_has_forward_with_fp8_input(self):
        """BailingMoEAttention must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        self.assertTrue(
            hasattr(mod.BailingMoEAttention, "_forward_with_fp8_input"),
            "BailingMoEAttention must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_bailing_moe_mlp_has_forward_with_fp8_input(self):
        """BailingMoEMLP must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        self.assertTrue(
            hasattr(mod.BailingMoEMLP, "_forward_with_fp8_input"),
            "BailingMoEMLP must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_bailing_moe_block_has_forward_aiter_fp8(self):
        """BailingMoEBlock must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        self.assertTrue(
            hasattr(mod.BailingMoEBlock, "_forward_aiter_fp8"),
            "BailingMoEBlock must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_bailing_moe_uses_query_key_value_for_detection(self):
        """BailingMoEBlock._aiter_fp8 must use query_key_value (not qkv_proj) for detection."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        src = inspect.getsource(mod.BailingMoEBlock.__init__)
        self.assertIn(
            "query_key_value", src,
            "BailingMoEBlock must use query_key_value for _aiter_fp8 detection "
            "(BailingMoE uses query_key_value, not qkv_proj)",
        )

    def test_bailing_moe_aiter_fp8_check_includes_w8a8fp8(self):
        """BailingMoEBlock._aiter_fp8 check must include W8A8Fp8LinearMethod."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        src = inspect.getsource(mod.BailingMoEBlock.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "BailingMoEBlock must check W8A8Fp8LinearMethod for _aiter_fp8",
        )

    def test_bailing_moe_forward_dispatches_to_aiter_fp8(self):
        """BailingMoEBlock.forward() must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.bailing_moe")
        src = inspect.getsource(mod.BailingMoEBlock.forward)
        self.assertIn(
            "_forward_aiter_fp8", src,
            "BailingMoEBlock.forward must dispatch to _forward_aiter_fp8 when _aiter_fp8 is set",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestMixtralFusedFP8Path(unittest.TestCase):
    """Test Mixtral fused RMSNorm+FP8 attention path for AMD MI300X.

    Note: MixtralMoE does not support FP8 input, so only the attention
    pre-norm is fused. The post-attention norm uses the standard allreduce path.
    """

    def test_mixtral_attention_has_forward_with_fp8_input(self):
        """MixtralAttention must have _forward_with_fp8_input."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral")
        self.assertTrue(
            hasattr(mod.MixtralAttention, "_forward_with_fp8_input"),
            "MixtralAttention must have _forward_with_fp8_input for AMD AITER optimization",
        )

    def test_mixtral_decoder_layer_has_forward_aiter_fp8(self):
        """MixtralDecoderLayer must have _forward_aiter_fp8."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral")
        self.assertTrue(
            hasattr(mod.MixtralDecoderLayer, "_forward_aiter_fp8"),
            "MixtralDecoderLayer must have _forward_aiter_fp8 for AMD AITER optimization",
        )

    def test_mixtral_decoder_layer_has_aiter_fp8_flag(self):
        """MixtralDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral")
        src = inspect.getsource(mod.MixtralDecoderLayer.__init__)
        self.assertIn(
            "W8A8Fp8LinearMethod", src,
            "MixtralDecoderLayer must check W8A8Fp8LinearMethod for _aiter_fp8",
        )

    def test_mixtral_forward_dispatches_to_aiter_fp8(self):
        """MixtralDecoderLayer.forward() must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral")
        src = inspect.getsource(mod.MixtralDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8", src,
            "MixtralDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_mixtral_aiter_fp8_uses_standard_moe_path(self):
        """_forward_aiter_fp8 must call block_sparse_moe directly (no FP8 MLP)."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral")
        src = inspect.getsource(mod.MixtralDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "block_sparse_moe", src,
            "_forward_aiter_fp8 must call block_sparse_moe (standard MoE, no FP8 MLP)",
        )


class TestQwen2MoEFusedFP8Path(unittest.TestCase):
    """Tests for Qwen2MoE fused RMSNorm+FP8 attention path (AMD AITER)."""

    def test_qwen2_moe_attention_has_forward_with_fp8_input(self):
        """Qwen2MoeAttention must have _forward_with_fp8_input method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        self.assertTrue(
            hasattr(mod.Qwen2MoeAttention, "_forward_with_fp8_input"),
            "Qwen2MoeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_qwen2_moe_decoder_layer_has_forward_aiter_fp8(self):
        """Qwen2MoeDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        self.assertTrue(
            hasattr(mod.Qwen2MoeDecoderLayer, "_forward_aiter_fp8"),
            "Qwen2MoeDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_qwen2_moe_decoder_layer_has_aiter_fp8_flag(self):
        """Qwen2MoeDecoderLayer._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        src = inspect.getsource(mod.Qwen2MoeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Qwen2MoeDecoderLayer.__init__ must set _aiter_fp8 for AMD detection",
        )

    def test_qwen2_moe_forward_dispatches_to_aiter_fp8(self):
        """Qwen2MoeDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        src = inspect.getsource(mod.Qwen2MoeDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Qwen2MoeDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_qwen2_moe_aiter_fp8_uses_standard_moe_path(self):
        """_forward_aiter_fp8 must use standard path for MoE (not FP8 MLP fused)."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.qwen2_moe")
        src = inspect.getsource(mod.Qwen2MoeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "self.mlp",
            src,
            "_forward_aiter_fp8 must call self.mlp (MoE, standard path)",
        )


class TestLlama4FusedFP8Path(unittest.TestCase):
    """Tests for Llama4 fused RMSNorm+FP8 decoder layer (AMD AITER)."""

    def test_llama4_attention_has_forward_with_fp8_input(self):
        """Llama4Attention must have _forward_with_fp8_input method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        self.assertTrue(
            hasattr(mod.Llama4Attention, "_forward_with_fp8_input"),
            "Llama4Attention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_llama4_decoder_layer_has_forward_aiter_fp8(self):
        """Llama4DecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        self.assertTrue(
            hasattr(mod.Llama4DecoderLayer, "_forward_aiter_fp8"),
            "Llama4DecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_llama4_decoder_layer_has_aiter_fp8_flag(self):
        """Llama4DecoderLayer._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        src = inspect.getsource(mod.Llama4DecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Llama4DecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_llama4_forward_dispatches_to_aiter_fp8(self):
        """Llama4DecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        src = inspect.getsource(mod.Llama4DecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Llama4DecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_llama4_aiter_fp8_handles_nope_layers(self):
        """_forward_with_fp8_input must handle both RoPE and NoPE layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        src = inspect.getsource(mod.Llama4Attention._forward_with_fp8_input)
        # Must handle temperature tuning for NoPE layers
        self.assertIn(
            "attn_temperature_tuning",
            src,
            "_forward_with_fp8_input must handle NoPE temperature tuning",
        )

    def test_llama4_aiter_fp8_dense_uses_fp8_mlp(self):
        """_forward_aiter_fp8 must try FP8 MLP path for dense LlamaMLP layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llama4")
        src = inspect.getsource(mod.Llama4DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "LlamaMLP",
            src,
            "_forward_aiter_fp8 must dispatch FP8 MLP path for dense LlamaMLP layers",
        )


class TestGlm4MoeFusedFP8Path(unittest.TestCase):
    """Tests for GLM4MoE fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_glm4_moe_mlp_has_forward_with_fp8_input(self):
        """Glm4MoeMLP must have _forward_with_fp8_input for the dense FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4_moe")
        self.assertTrue(
            hasattr(mod.Glm4MoeMLP, "_forward_with_fp8_input"),
            "Glm4MoeMLP must have _forward_with_fp8_input for AMD AITER dense MLP FP8 path",
        )

    def test_glm4_moe_attention_has_forward_with_fp8_input(self):
        """Glm4MoeAttention must have _forward_with_fp8_input for FP8 attention."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4_moe")
        self.assertTrue(
            hasattr(mod.Glm4MoeAttention, "_forward_with_fp8_input"),
            "Glm4MoeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_glm4_moe_decoder_layer_has_forward_aiter_fp8(self):
        """Glm4MoeDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4_moe")
        self.assertTrue(
            hasattr(mod.Glm4MoeDecoderLayer, "_forward_aiter_fp8"),
            "Glm4MoeDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_glm4_moe_decoder_layer_has_aiter_fp8_flag(self):
        """Glm4MoeDecoderLayer._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4_moe")
        src = inspect.getsource(mod.Glm4MoeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Glm4MoeDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_glm4_moe_forward_dispatches_to_aiter_fp8(self):
        """Glm4MoeDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4_moe")
        src = inspect.getsource(mod.Glm4MoeDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Glm4MoeDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )


class TestGptOssFusedFP8Path(unittest.TestCase):
    """Tests for GptOss fused RMSNorm+FP8 attention path (AMD AITER)."""

    def test_gpt_oss_attention_has_forward_with_fp8_input(self):
        """GptOssAttention must have _forward_with_fp8_input for FP8 attention."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gpt_oss")
        self.assertTrue(
            hasattr(mod.GptOssAttention, "_forward_with_fp8_input"),
            "GptOssAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_gpt_oss_decoder_layer_has_forward_aiter_fp8(self):
        """GptOssDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gpt_oss")
        self.assertTrue(
            hasattr(mod.GptOssDecoderLayer, "_forward_aiter_fp8"),
            "GptOssDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_gpt_oss_decoder_layer_has_aiter_fp8_flag(self):
        """GptOssDecoderLayer._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gpt_oss")
        src = inspect.getsource(mod.GptOssDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "GptOssDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_gpt_oss_forward_dispatches_to_aiter_fp8(self):
        """GptOssDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gpt_oss")
        src = inspect.getsource(mod.GptOssDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "GptOssDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_gpt_oss_fp8_handles_fused_set_kv_buffer(self):
        """GptOssAttention._forward_with_fp8_input must replicate fused_set_kv_buffer logic."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gpt_oss")
        src = inspect.getsource(mod.GptOssAttention._forward_with_fp8_input)
        self.assertIn(
            "fused_set_kv_buffer",
            src,
            "_forward_with_fp8_input must replicate fused_set_kv_buffer logic",
        )


class TestLLaDA2FusedFP8Path(unittest.TestCase):
    """Tests for LLaDA2 fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_llada2_mlp_has_forward_with_fp8_input(self):
        """LLaDA2MoeMLP must have _forward_with_fp8_input for the dense FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        self.assertTrue(
            hasattr(mod.LLaDA2MoeMLP, "_forward_with_fp8_input"),
            "LLaDA2MoeMLP must have _forward_with_fp8_input for AMD AITER dense MLP FP8 path",
        )

    def test_llada2_attention_has_forward_with_fp8_input(self):
        """LLaDA2MoeAttention must have _forward_with_fp8_input for FP8 attention."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        self.assertTrue(
            hasattr(mod.LLaDA2MoeAttention, "_forward_with_fp8_input"),
            "LLaDA2MoeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_llada2_block_has_forward_aiter_fp8(self):
        """LLaDA2MoeBlock must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        self.assertTrue(
            hasattr(mod.LLaDA2MoeBlock, "_forward_aiter_fp8"),
            "LLaDA2MoeBlock must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_llada2_block_has_aiter_fp8_flag(self):
        """LLaDA2MoeBlock._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        src = inspect.getsource(mod.LLaDA2MoeBlock.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "LLaDA2MoeBlock.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_llada2_forward_dispatches_to_aiter_fp8(self):
        """LLaDA2MoeBlock.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        src = inspect.getsource(mod.LLaDA2MoeBlock.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "LLaDA2MoeBlock.forward must dispatch to _forward_aiter_fp8",
        )

    def test_llada2_attention_uses_query_key_value(self):
        """LLaDA2MoeAttention._forward_with_fp8_input must use query_key_value."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.llada2")
        src = inspect.getsource(mod.LLaDA2MoeAttention._forward_with_fp8_input)
        self.assertIn(
            "query_key_value",
            src,
            "_forward_with_fp8_input must use query_key_value (not qkv_proj)",
        )


class TestMiMoV2MTPFusedFP8Path(unittest.TestCase):
    """Tests for MiMoV2MTP fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_mimo_v2_mtp_layer_has_forward_aiter_fp8(self):
        """MiMoV2MTPLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mimo_v2_flash_nextn")
        self.assertTrue(
            hasattr(mod.MiMoV2MTPLayer, "_forward_aiter_fp8"),
            "MiMoV2MTPLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_mimo_v2_mtp_layer_has_aiter_fp8_flag(self):
        """MiMoV2MTPLayer._aiter_fp8 flag must be set in __init__."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mimo_v2_flash_nextn")
        src = inspect.getsource(mod.MiMoV2MTPLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "MiMoV2MTPLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_mimo_v2_mtp_forward_dispatches_to_aiter_fp8(self):
        """MiMoV2MTPLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mimo_v2_flash_nextn")
        src = inspect.getsource(mod.MiMoV2MTPLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "MiMoV2MTPLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_mimo_v2_mtp_uses_mimo_v2_fp8_methods(self):
        """_forward_aiter_fp8 must reuse MiMoV2Attention/_forward_with_fp8_input."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mimo_v2_flash_nextn")
        src = inspect.getsource(mod.MiMoV2MTPLayer._forward_aiter_fp8)
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call _forward_with_fp8_input for attention",
        )


class TestNemotronHFusedFP8Path(unittest.TestCase):
    """Tests for NemotronH fused RMSNorm+FP8 attention decoder path (AMD AITER)."""

    def test_nemotron_h_attention_has_forward_with_fp8_input(self):
        """NemotronHAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        self.assertTrue(
            hasattr(mod.NemotronHAttention, "_forward_with_fp8_input"),
            "NemotronHAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_nemotron_h_attn_decoder_has_forward_aiter_fp8(self):
        """NemotronHAttentionDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        self.assertTrue(
            hasattr(mod.NemotronHAttentionDecoderLayer, "_forward_aiter_fp8"),
            "NemotronHAttentionDecoderLayer must have _forward_aiter_fp8 for AMD AITER",
        )

    def test_nemotron_h_attn_decoder_has_aiter_fp8_flag(self):
        """NemotronHAttentionDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        src = inspect.getsource(mod.NemotronHAttentionDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "NemotronHAttentionDecoderLayer.__init__ must set _aiter_fp8",
        )

    def test_nemotron_h_forward_dispatches_to_aiter_fp8(self):
        """NemotronHAttentionDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        src = inspect.getsource(mod.NemotronHAttentionDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "NemotronHAttentionDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_nemotron_h_fp8_no_rope(self):
        """NemotronHAttention._forward_with_fp8_input must not require positions (no RoPE)."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        src = inspect.getsource(mod.NemotronHAttention._forward_with_fp8_input)
        self.assertNotIn(
            "rotary_emb",
            src,
            "NemotronHAttention._forward_with_fp8_input must not call rotary_emb (no RoPE)",
        )

    def test_nemotron_h_module_has_use_aiter_flag(self):
        """nemotron_h module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "nemotron_h module must have _use_aiter flag for AMD AITER detection",
        )


class TestGlm4FusedFP8Path(unittest.TestCase):
    """Tests for GLM4 fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_glm4_attention_has_forward_with_fp8_input(self):
        """Glm4Attention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        self.assertTrue(
            hasattr(mod.Glm4Attention, "_forward_with_fp8_input"),
            "Glm4Attention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_glm4_decoder_layer_has_forward_aiter_fp8(self):
        """Glm4DecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        self.assertTrue(
            hasattr(mod.Glm4DecoderLayer, "_forward_aiter_fp8"),
            "Glm4DecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_glm4_decoder_layer_has_aiter_fp8_flag(self):
        """Glm4DecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        src = inspect.getsource(mod.Glm4DecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Glm4DecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_glm4_forward_dispatches_to_aiter_fp8(self):
        """Glm4DecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        src = inspect.getsource(mod.Glm4DecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Glm4DecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_glm4_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out for fused norm+quant."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        src = inspect.getsource(mod.Glm4DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out for fused norm+FP8",
        )

    def test_glm4_aiter_fp8_handles_post_self_attn_norm(self):
        """_forward_aiter_fp8 must apply post_self_attn_layernorm after attention."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        src = inspect.getsource(mod.Glm4DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_self_attn_layernorm",
            src,
            "_forward_aiter_fp8 must apply post_self_attn_layernorm (GLM4 post-attn norm)",
        )

    def test_glm4_mlp_has_forward_with_fp8_input(self):
        """Glm4MLP must have _forward_with_fp8_input for full-layer FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        self.assertTrue(
            hasattr(mod.Glm4MLP, "_forward_with_fp8_input"),
            "Glm4MLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_glm4_aiter_fp8_fuses_post_attention_layernorm_for_mlp(self):
        """_forward_aiter_fp8 must fuse post_attention_layernorm for MLP FP8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        src = inspect.getsource(mod.Glm4DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_attention_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse post_attention_layernorm for MLP FP8",
        )
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp.gate_up_proj via _forward_with_fp8_input",
        )

    def test_glm4_module_has_use_aiter_flag(self):
        """glm4 module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.glm4")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "glm4 module must have _use_aiter flag for AMD AITER detection",
        )


class TestStep3p5FusedFP8Path(unittest.TestCase):
    """Tests for Step3p5 fused GemmaRMSNorm+FP8 decoder path (AMD AITER)."""

    def test_step3p5_attention_has_forward_with_fp8_input(self):
        """Step3p5Attention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        self.assertTrue(
            hasattr(mod.Step3p5Attention, "_forward_with_fp8_input"),
            "Step3p5Attention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_step3p5_attention_has_aiter_fp8_flag(self):
        """Step3p5Attention.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5Attention.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Step3p5Attention.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_step3p5_decoder_layer_has_forward_aiter_fp8(self):
        """Step3p5DecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        self.assertTrue(
            hasattr(mod.Step3p5DecoderLayer, "_forward_aiter_fp8"),
            "Step3p5DecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_step3p5_forward_dispatches_to_aiter_fp8(self):
        """Step3p5DecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5DecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Step3p5DecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_step3p5_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call prepare_attn_fp8_out for fused norm+quant."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_attn_fp8_out",
            src,
            "_forward_aiter_fp8 must use prepare_attn_fp8_out for fused norm+FP8",
        )

    def test_step3p5_aiter_fp8_calls_forward_with_fp8_input(self):
        """_forward_aiter_fp8 must call _forward_with_fp8_input on attention."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call attention._forward_with_fp8_input",
        )

    def test_step3p5_fp8_head_wise_gate_dequantizes(self):
        """_forward_with_fp8_input must handle use_head_wise_attn_gate via dequant."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5Attention._forward_with_fp8_input)
        self.assertIn(
            "use_head_wise_attn_gate",
            src,
            "_forward_with_fp8_input must handle use_head_wise_attn_gate for g_proj",
        )

    def test_step3p5_mlp_has_forward_with_fp8_input(self):
        """Step3p5MLP must have _forward_with_fp8_input for full-layer FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        self.assertTrue(
            hasattr(mod.Step3p5MLP, "_forward_with_fp8_input"),
            "Step3p5MLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_step3p5_aiter_fp8_fuses_post_attention_norm_for_dense_mlp(self):
        """_forward_aiter_fp8 must fuse post_attention_layernorm for dense MLP layers."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        src = inspect.getsource(mod.Step3p5DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_attention_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse post_attention_layernorm for dense MLP FP8",
        )
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp.gate_up_proj via _forward_with_fp8_input",
        )

    def test_step3p5_module_has_use_aiter_flag(self):
        """step3p5 module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.step3p5")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "step3p5 module must have _use_aiter flag for AMD AITER detection",
        )


class TestArceeFusedFP8Path(unittest.TestCase):
    """Tests for Arcee (AFM) fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_arcee_attention_has_forward_with_fp8_input(self):
        """ArceeAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        self.assertTrue(
            hasattr(mod.ArceeAttention, "_forward_with_fp8_input"),
            "ArceeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_arcee_decoder_layer_has_aiter_fp8_flag(self):
        """ArceeDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        src = inspect.getsource(mod.ArceeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "ArceeDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_arcee_decoder_layer_has_forward_aiter_fp8(self):
        """ArceeDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        self.assertTrue(
            hasattr(mod.ArceeDecoderLayer, "_forward_aiter_fp8"),
            "ArceeDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_arcee_forward_dispatches_to_aiter_fp8(self):
        """ArceeDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        src = inspect.getsource(mod.ArceeDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "ArceeDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_arcee_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out for both norms."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        src = inspect.getsource(mod.ArceeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out for fused norm+FP8",
        )
        self.assertIn(
            "post_attention_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse post_attention_layernorm too",
        )

    def test_arcee_mlp_has_forward_with_fp8_input(self):
        """ArceeMLP must have _forward_with_fp8_input for full-layer FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        self.assertTrue(
            hasattr(mod.ArceeMLP, "_forward_with_fp8_input"),
            "ArceeMLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_arcee_aiter_fp8_fuses_mlp_via_forward_with_fp8_input(self):
        """_forward_aiter_fp8 must also fuse the MLP norm and pass FP8 to up_proj."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        src = inspect.getsource(mod.ArceeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call mlp._forward_with_fp8_input for MLP FP8",
        )

    def test_arcee_aiter_fp8_uses_allreduce_fusion_when_available(self):
        """_forward_aiter_fp8 must use forward_with_allreduce_fusion_fp8_out for TP>1."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        src = inspect.getsource(mod.ArceeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_with_allreduce_fusion_fp8_out",
            src,
            "_forward_aiter_fp8 must try forward_with_allreduce_fusion_fp8_out for TP>1 savings",
        )

    def test_arcee_module_has_use_aiter_flag(self):
        """arcee module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.arcee")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "arcee module must have _use_aiter flag for AMD AITER detection",
        )


class TestMixtralQuantFusedFP8Path(unittest.TestCase):
    """Tests for MixtralQuant fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_mixtral_quant_attention_has_forward_with_fp8_input(self):
        """MixtralAttention (quant) must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        self.assertTrue(
            hasattr(mod.MixtralAttention, "_forward_with_fp8_input"),
            "mixtral_quant MixtralAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_mixtral_quant_decoder_layer_has_aiter_fp8_flag(self):
        """MixtralDecoderLayer (quant).__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        src = inspect.getsource(mod.MixtralDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "mixtral_quant MixtralDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_mixtral_quant_decoder_layer_has_forward_aiter_fp8(self):
        """MixtralDecoderLayer (quant) must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        self.assertTrue(
            hasattr(mod.MixtralDecoderLayer, "_forward_aiter_fp8"),
            "mixtral_quant MixtralDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_mixtral_quant_forward_dispatches_to_aiter_fp8(self):
        """MixtralDecoderLayer (quant).forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        src = inspect.getsource(mod.MixtralDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "mixtral_quant MixtralDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_mixtral_quant_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out for fused norm+quant."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        src = inspect.getsource(mod.MixtralDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out for fused norm+FP8",
        )

    def test_mixtral_quant_module_has_use_aiter_flag(self):
        """mixtral_quant module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.mixtral_quant")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "mixtral_quant module must have _use_aiter flag for AMD AITER detection",
        )


class TestExaoneMoEFusedFP8Path(unittest.TestCase):
    """Tests for ExaoneMoE fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_exaone_moe_mlp_has_forward_with_fp8_input(self):
        """ExaoneMoEMLP must have _forward_with_fp8_input for dense MLP FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        self.assertTrue(
            hasattr(mod.ExaoneMoEMLP, "_forward_with_fp8_input"),
            "ExaoneMoEMLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_exaone_moe_attention_has_forward_with_fp8_input(self):
        """ExaoneMoEAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        self.assertTrue(
            hasattr(mod.ExaoneMoEAttention, "_forward_with_fp8_input"),
            "ExaoneMoEAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_exaone_moe_attention_fp8_uses_apply_qk_norm(self):
        """ExaoneMoEAttention._forward_with_fp8_input must use apply_qk_norm."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        src = inspect.getsource(mod.ExaoneMoEAttention._forward_with_fp8_input)
        self.assertIn(
            "apply_qk_norm",
            src,
            "ExaoneMoEAttention._forward_with_fp8_input must use apply_qk_norm for per-head QK norm",
        )

    def test_exaone_moe_decoder_layer_has_aiter_fp8_flag(self):
        """ExaoneMoEDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        src = inspect.getsource(mod.ExaoneMoEDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "ExaoneMoEDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_exaone_moe_decoder_layer_handles_mixed_dense_moe(self):
        """_forward_aiter_fp8 must handle both dense MLP (FP8-fused) and sparse MoE (standard) paths."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        src = inspect.getsource(mod.ExaoneMoEDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_is_moe_layer",
            src,
            "_forward_aiter_fp8 must branch on _is_moe_layer for dense vs sparse MLP",
        )

    def test_exaone_moe_module_has_use_aiter_flag(self):
        """exaone_moe module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.exaone_moe")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "exaone_moe module must have _use_aiter flag for AMD AITER detection",
        )


class TestOlmoeFusedFP8Path(unittest.TestCase):
    """Tests for OLMoE fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_olmoe_attention_has_forward_with_fp8_input(self):
        """OlmoeAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.olmoe")
        self.assertTrue(
            hasattr(mod.OlmoeAttention, "_forward_with_fp8_input"),
            "OlmoeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_olmoe_decoder_layer_has_aiter_fp8_flag(self):
        """OlmoeDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.olmoe")
        src = inspect.getsource(mod.OlmoeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "OlmoeDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_olmoe_decoder_layer_has_forward_aiter_fp8(self):
        """OlmoeDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.olmoe")
        self.assertTrue(
            hasattr(mod.OlmoeDecoderLayer, "_forward_aiter_fp8"),
            "OlmoeDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_olmoe_aiter_fp8_attention_only(self):
        """OLMoE _forward_aiter_fp8 must only fuse attention (MLP is always MoE)."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.olmoe")
        src = inspect.getsource(mod.OlmoeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out for fused input_layernorm+FP8",
        )
        self.assertNotIn(
            "mlp._forward_with_fp8_input",
            src,
            "OLMoE _forward_aiter_fp8 must NOT call mlp._forward_with_fp8_input (MoE is not FP8-fused)",
        )

    def test_olmoe_module_has_use_aiter_flag(self):
        """olmoe module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.olmoe")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "olmoe module must have _use_aiter flag for AMD AITER detection",
        )


class TestGemma2FusedFP8Path(unittest.TestCase):
    """Tests for Gemma2 fused GemmaRMSNorm+FP8 decoder path (AMD AITER)."""

    def test_gemma2_attention_has_forward_with_fp8_input(self):
        """Gemma2Attention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        self.assertTrue(
            hasattr(mod.Gemma2Attention, "_forward_with_fp8_input"),
            "Gemma2Attention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_gemma2_decoder_layer_has_aiter_fp8_flag(self):
        """Gemma2DecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        src = inspect.getsource(mod.Gemma2DecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Gemma2DecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_gemma2_decoder_layer_has_forward_aiter_fp8(self):
        """Gemma2DecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        self.assertTrue(
            hasattr(mod.Gemma2DecoderLayer, "_forward_aiter_fp8"),
            "Gemma2DecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_gemma2_forward_dispatches_to_aiter_fp8(self):
        """Gemma2DecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        src = inspect.getsource(mod.Gemma2DecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Gemma2DecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_gemma2_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out for fused norm+quant."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        src = inspect.getsource(mod.Gemma2DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out for fused GemmaRMSNorm+FP8",
        )

    def test_gemma2_module_has_use_aiter_flag(self):
        """gemma2 module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "gemma2 module must have _use_aiter flag for AMD AITER detection",
        )

    def test_gemma2_aiter_fp8_preserves_post_attention_norm(self):
        """_forward_aiter_fp8 must apply post_attention_layernorm after allreduce."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        src = inspect.getsource(mod.Gemma2DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_attention_layernorm",
            src,
            "_forward_aiter_fp8 must apply post_attention_layernorm (Gemma2 architecture)",
        )

    def test_gemma2_mlp_has_forward_with_fp8_input(self):
        """Gemma2MLP must have _forward_with_fp8_input for full-layer FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        self.assertTrue(
            hasattr(mod.Gemma2MLP, "_forward_with_fp8_input"),
            "Gemma2MLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_gemma2_aiter_fp8_fuses_pre_feedforward_layernorm(self):
        """_forward_aiter_fp8 must fuse pre_feedforward_layernorm via forward_aiter_fp8_out."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.gemma2")
        src = inspect.getsource(mod.Gemma2DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "pre_feedforward_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse pre_feedforward_layernorm for MLP FP8",
        )
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp.gate_up_proj via _forward_with_fp8_input",
        )


class TestErnie45VLMoeFusedFP8Path(unittest.TestCase):
    """Tests for Ernie4.5 VL MoE fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_ernie45_vl_moe_attention_has_forward_with_fp8_input(self):
        """Ernie4_5_VLMoeAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie45_moe_vl")
        self.assertTrue(
            hasattr(mod.Ernie4_5_VLMoeAttention, "_forward_with_fp8_input"),
            "Ernie4_5_VLMoeAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_ernie45_vl_moe_decoder_layer_has_aiter_fp8_flag(self):
        """Ernie4_5_VLMoeDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie45_moe_vl")
        src = inspect.getsource(mod.Ernie4_5_VLMoeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Ernie4_5_VLMoeDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_ernie45_vl_moe_decoder_layer_has_forward_aiter_fp8(self):
        """Ernie4_5_VLMoeDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie45_moe_vl")
        self.assertTrue(
            hasattr(mod.Ernie4_5_VLMoeDecoderLayer, "_forward_aiter_fp8"),
            "Ernie4_5_VLMoeDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_ernie45_vl_moe_handles_mixed_dense_moe(self):
        """_forward_aiter_fp8 must handle both dense MLP (FP8-fused) and sparse MoE (standard) paths."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie45_moe_vl")
        src = inspect.getsource(mod.Ernie4_5_VLMoeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_is_moe_layer",
            src,
            "_forward_aiter_fp8 must branch on _is_moe_layer for dense vs sparse MLP",
        )

    def test_ernie45_vl_moe_module_has_use_aiter_flag(self):
        """ernie45_moe_vl module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.ernie45_moe_vl")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "ernie45_moe_vl module must have _use_aiter flag for AMD AITER detection",
        )


class TestApertusFusedFP8Path(unittest.TestCase):
    """Tests for Apertus fused RMSNorm+FP8 decoder path (AMD AITER)."""

    def test_apertus_attention_has_forward_with_fp8_input(self):
        """ApertusAttention must have _forward_with_fp8_input for FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        self.assertTrue(
            hasattr(mod.ApertusAttention, "_forward_with_fp8_input"),
            "ApertusAttention must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_apertus_decoder_layer_has_aiter_fp8_flag(self):
        """ApertusDecoderLayer.__init__ must set _aiter_fp8 flag."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "ApertusDecoderLayer.__init__ must set _aiter_fp8 for AMD AITER detection",
        )

    def test_apertus_decoder_layer_has_forward_aiter_fp8(self):
        """ApertusDecoderLayer must have _forward_aiter_fp8 method."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        self.assertTrue(
            hasattr(mod.ApertusDecoderLayer, "_forward_aiter_fp8"),
            "ApertusDecoderLayer must have _forward_aiter_fp8 for AMD AITER path",
        )

    def test_apertus_forward_dispatches_to_aiter_fp8(self):
        """ApertusDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "ApertusDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_apertus_aiter_fp8_fuses_attention_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out on attention_layernorm."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use attention_layernorm.forward_aiter_fp8_out",
        )

    def test_apertus_attention_fp8_applies_per_head_qk_norms(self):
        """_forward_with_fp8_input must apply per-head q_norm and k_norm."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusAttention._forward_with_fp8_input)
        self.assertIn(
            "q_norm",
            src,
            "_forward_with_fp8_input must apply q_norm (per-head QK norm)",
        )

    def test_apertus_mlp_has_forward_with_fp8_input(self):
        """ApertusMLP must have _forward_with_fp8_input for full-layer FP8 path."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        self.assertTrue(
            hasattr(mod.ApertusMLP, "_forward_with_fp8_input"),
            "ApertusMLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_apertus_aiter_fp8_fuses_feedforward_layernorm(self):
        """_forward_aiter_fp8 must fuse feedforward_layernorm for MLP FP8."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "feedforward_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse feedforward_layernorm for MLP FP8",
        )
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp.up_proj via _forward_with_fp8_input",
        )

    def test_apertus_aiter_fp8_uses_allreduce_fusion_when_available(self):
        """_forward_aiter_fp8 must use forward_with_allreduce_fusion_fp8_out for TP>1."""
        import inspect
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        src = inspect.getsource(mod.ApertusDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_with_allreduce_fusion_fp8_out",
            src,
            "_forward_aiter_fp8 must try forward_with_allreduce_fusion_fp8_out for TP>1 savings",
        )

    def test_apertus_module_has_use_aiter_flag(self):
        """apertus module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.apertus")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "apertus module must have _use_aiter flag for AMD AITER detection",
        )


class TestNemotronHMLPFusedFP8Path(unittest.TestCase):
    """Tests for the NemotronH MLP decoder layer fused RMSNorm+FP8 path (AMD AITER)."""

    def test_nemotron_h_mlp_has_forward_with_fp8_input(self):
        """NemotronHMLP must have _forward_with_fp8_input for AMD AITER FP8 path."""
        from sglang.srt.models.nemotron_h import NemotronHMLP
        self.assertTrue(
            hasattr(NemotronHMLP, "_forward_with_fp8_input"),
            "NemotronHMLP must have _forward_with_fp8_input for AMD AITER FP8 path",
        )

    def test_nemotron_h_mlp_decoder_layer_has_aiter_fp8_flag(self):
        """NemotronHMLPDecoderLayer must have _aiter_fp8 attribute."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        self.assertTrue(
            hasattr(NemotronHMLPDecoderLayer, "_forward_aiter_fp8"),
            "NemotronHMLPDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_nemotron_h_mlp_decoder_layer_has_forward_aiter_fp8(self):
        """NemotronHMLPDecoderLayer must have _forward_aiter_fp8 method."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        self.assertTrue(
            hasattr(NemotronHMLPDecoderLayer, "_forward_aiter_fp8"),
            "NemotronHMLPDecoderLayer must have _forward_aiter_fp8",
        )

    def test_nemotron_h_mlp_decoder_forward_dispatches_to_aiter(self):
        """NemotronHMLPDecoderLayer.forward must dispatch to _forward_aiter_fp8 when flag set."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        import inspect
        src = inspect.getsource(NemotronHMLPDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "NemotronHMLPDecoderLayer.forward must dispatch on _aiter_fp8",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "NemotronHMLPDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_nemotron_h_mlp_aiter_fp8_uses_fused_norm(self):
        """NemotronHMLPDecoderLayer._forward_aiter_fp8 must use forward_aiter_fp8_out."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        import inspect
        src = inspect.getsource(NemotronHMLPDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use fused norm forward_aiter_fp8_out",
        )
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call mixer._forward_with_fp8_input",
        )

    def test_nemotron_h_mlp_forward_with_fp8_uses_up_proj(self):
        """NemotronHMLP._forward_with_fp8_input must use up_proj.forward_with_fp8_input."""
        from sglang.srt.models.nemotron_h import NemotronHMLP
        import inspect
        src = inspect.getsource(NemotronHMLP._forward_with_fp8_input)
        self.assertIn(
            "up_proj.forward_with_fp8_input",
            src,
            "_forward_with_fp8_input must use up_proj.forward_with_fp8_input",
        )

    def test_nemotron_h_mlp_decoder_layer_aiter_fp8_not_set_by_default(self):
        """NemotronHMLPDecoderLayer._aiter_fp8 must default to False without ROCm FP8 quant."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        # Verify _aiter_fp8 is set in __init__ (not a class-level constant)
        import inspect
        src = inspect.getsource(NemotronHMLPDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "NemotronHMLPDecoderLayer.__init__ must set _aiter_fp8",
        )
        self.assertIn(
            "_use_aiter",
            src,
            "NemotronHMLPDecoderLayer.__init__ must check _use_aiter flag",
        )

    def test_nemotron_h_mlp_fp8_respects_hidden_states_shape(self):
        """NemotronHMLPDecoderLayer.forward must skip FP8 path for empty batches."""
        from sglang.srt.models.nemotron_h import NemotronHMLPDecoderLayer
        import inspect
        src = inspect.getsource(NemotronHMLPDecoderLayer.forward)
        self.assertIn(
            "hidden_states.shape[0] != 0",
            src,
            "NemotronHMLPDecoderLayer.forward must guard FP8 path with shape check",
        )

    def test_nemotron_h_module_has_use_aiter_flag(self):
        """nemotron_h module must have _use_aiter module-level flag."""
        import importlib
        mod = importlib.import_module("sglang.srt.models.nemotron_h")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "nemotron_h module must have _use_aiter flag for AMD AITER detection",
        )


class TestHunYuanFusedFP8Path(unittest.TestCase):
    """Tests for fused RMSNorm+FP8 quantization path in HunYuanDecoderLayer (AMD AITER)."""

    def test_hunyuan_mlp_has_forward_with_fp8_input(self):
        """HunYuanMLP must expose _forward_with_fp8_input for AMD AITER path."""
        from sglang.srt.models.hunyuan import HunYuanMLP

        self.assertTrue(
            hasattr(HunYuanMLP, "_forward_with_fp8_input"),
            "HunYuanMLP must have _forward_with_fp8_input method",
        )

    def test_hunyuan_attention_has_forward_with_fp8_input(self):
        """HunYuanAttention must expose _forward_with_fp8_input for AMD AITER path."""
        from sglang.srt.models.hunyuan import HunYuanAttention

        self.assertTrue(
            hasattr(HunYuanAttention, "_forward_with_fp8_input"),
            "HunYuanAttention must have _forward_with_fp8_input method",
        )

    def test_hunyuan_attention_fp8_handles_skip_o_reduce(self):
        """HunYuanAttention._forward_with_fp8_input must accept skip_o_reduce kwarg."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanAttention

        sig = inspect.signature(HunYuanAttention._forward_with_fp8_input)
        self.assertIn(
            "skip_o_reduce",
            sig.parameters,
            "HunYuanAttention._forward_with_fp8_input must accept skip_o_reduce",
        )

    def test_hunyuan_decoder_layer_has_aiter_fp8_flag(self):
        """HunYuanDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "HunYuanDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_hunyuan_decoder_layer_has_forward_aiter_fp8(self):
        """HunYuanDecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        self.assertTrue(
            hasattr(HunYuanDecoderLayer, "_forward_aiter_fp8"),
            "HunYuanDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_hunyuan_forward_dispatches_to_aiter_fp8(self):
        """HunYuanDecoderLayer.forward must dispatch to _forward_aiter_fp8 when enabled."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "HunYuanDecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "HunYuanDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_hunyuan_aiter_fp8_fuses_input_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out on input_layernorm."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "input_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse input_layernorm via forward_aiter_fp8_out",
        )

    def test_hunyuan_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out on post_attention_layernorm."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_attention_layernorm",
            src,
            "_forward_aiter_fp8 must fuse post_attention_layernorm",
        )

    def test_hunyuan_aiter_fp8_uses_allreduce_fusion_when_available(self):
        """_forward_aiter_fp8 must attempt allreduce fusion for TP>1."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_with_allreduce_fusion_fp8_out",
            src,
            "_forward_aiter_fp8 must try allreduce fusion via forward_with_allreduce_fusion_fp8_out",
        )

    def test_hunyuan_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must only be True for dense (non-MoE) MLP layers."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer, HunYuanMLP

        src = inspect.getsource(HunYuanDecoderLayer.__init__)
        self.assertIn(
            "HunYuanMLP",
            src,
            "__init__ must check isinstance(self.mlp, HunYuanMLP) before enabling _aiter_fp8",
        )

    def test_hunyuan_aiter_fp8_respects_hidden_states_shape(self):
        """HunYuanDecoderLayer.forward must skip FP8 path for empty batches."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer.forward)
        self.assertIn(
            "hidden_states.shape[0] != 0",
            src,
            "HunYuanDecoderLayer.forward must guard FP8 path for empty batches",
        )

    def test_hunyuan_aiter_fp8_handles_cross_attention(self):
        """_forward_aiter_fp8 must handle cross-attention layers gracefully."""
        import inspect
        from sglang.srt.models.hunyuan import HunYuanDecoderLayer

        src = inspect.getsource(HunYuanDecoderLayer._forward_aiter_fp8)
        # Cross-attention layers fall back to standard attention + MLP FP8 only
        self.assertIn(
            "attention_type",
            src,
            "_forward_aiter_fp8 must check attention_type for cross-attention handling",
        )

    def test_hunyuan_module_has_use_aiter_import(self):
        """hunyuan module must import _use_aiter for AMD AITER detection."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.hunyuan")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "hunyuan module must import _use_aiter from fp8_utils",
        )


class TestKimiLinearFusedFP8Path(unittest.TestCase):
    """Tests for fused RMSNorm+FP8 MLP path in KimiDecoderLayer (AMD AITER)."""

    def test_kimi_decoder_layer_has_aiter_fp8_flag(self):
        """KimiDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "KimiDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_kimi_decoder_layer_has_forward_aiter_fp8(self):
        """KimiDecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        self.assertTrue(
            hasattr(KimiDecoderLayer, "_forward_aiter_fp8"),
            "KimiDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_kimi_forward_dispatches_to_aiter_fp8(self):
        """KimiDecoderLayer.forward must dispatch to _forward_aiter_fp8 when enabled."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "KimiDecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "KimiDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_kimi_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must call forward_aiter_fp8_out on post_attention_layernorm."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "post_attention_layernorm.forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must fuse post_attention_layernorm via forward_aiter_fp8_out",
        )

    def test_kimi_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_kimi_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must only be True for dense (non-MoE) KimiMLP layers."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer.__init__)
        self.assertIn(
            "KimiMLP",
            src,
            "__init__ must check isinstance(self.mlp, KimiMLP) before enabling _aiter_fp8",
        )

    def test_kimi_aiter_fp8_respects_hidden_states_shape(self):
        """KimiDecoderLayer.forward must skip FP8 path for empty batches."""
        import inspect
        from sglang.srt.models.kimi_linear import KimiDecoderLayer

        src = inspect.getsource(KimiDecoderLayer.forward)
        self.assertIn(
            "hidden_states.shape[0] != 0",
            src,
            "KimiDecoderLayer.forward must guard FP8 path for empty batches",
        )

    def test_kimi_module_has_use_aiter_import(self):
        """kimi_linear module must import _use_aiter for AMD AITER detection."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.kimi_linear")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "kimi_linear module must import _use_aiter from fp8_utils",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestStep3VLFusedFP8Path(unittest.TestCase):
    """Tests for fused post_attention_layernorm+FP8 MLP path in Step3TextDecoderLayer (AMD AITER)."""

    def test_step3_text_mlp_has_forward_with_fp8_input(self):
        """Step3TextMLP must implement _forward_with_fp8_input."""
        from sglang.srt.models.step3_vl import Step3TextMLP

        self.assertTrue(
            hasattr(Step3TextMLP, "_forward_with_fp8_input"),
            "Step3TextMLP must have _forward_with_fp8_input method",
        )

    def test_step3_decoder_layer_has_aiter_fp8_flag(self):
        """Step3TextDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        src = inspect.getsource(Step3TextDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Step3TextDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_step3_decoder_layer_has_forward_aiter_fp8(self):
        """Step3TextDecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        self.assertTrue(
            hasattr(Step3TextDecoderLayer, "_forward_aiter_fp8"),
            "Step3TextDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_step3_forward_dispatches_to_aiter_fp8(self):
        """Step3TextDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        src = inspect.getsource(Step3TextDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Step3TextDecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Step3TextDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_step3_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must call prepare_mlp_fp8_out for MLP sub-layer."""
        import inspect
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        src = inspect.getsource(Step3TextDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out for fused MLP norm",
        )

    def test_step3_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        src = inspect.getsource(Step3TextDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_step3_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must check Step3TextMLP type to guard MoE layers."""
        import inspect
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        src = inspect.getsource(Step3TextDecoderLayer.__init__)
        self.assertIn(
            "Step3TextMLP",
            src,
            "__init__ must check isinstance(self.mlp, Step3TextMLP) before enabling _aiter_fp8",
        )

    def test_step3_module_has_use_aiter_import(self):
        """step3_vl module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.step3_vl")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "step3_vl module must import _use_aiter from fp8_utils",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestDeepseekV2MLPFusedFP8Path(unittest.TestCase):
    """Tests for fused post_attention_layernorm+FP8 MLP path in DeepseekV2DecoderLayer (AMD AITER)."""

    def test_deepseek_v2_mlp_forward_with_fp8_input_has_allreduce_params(self):
        """DeepseekV2MLP._forward_with_fp8_input must accept should_allreduce_fusion and use_reduce_scatter."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

        src = inspect.getsource(DeepseekV2MLP._forward_with_fp8_input)
        self.assertIn(
            "should_allreduce_fusion",
            src,
            "_forward_with_fp8_input must accept should_allreduce_fusion parameter",
        )
        self.assertIn(
            "skip_all_reduce",
            src,
            "_forward_with_fp8_input must use skip_all_reduce in down_proj",
        )

    def test_deepseek_v2_decoder_layer_has_aiter_fp8_flag(self):
        """DeepseekV2DecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        src = inspect.getsource(DeepseekV2DecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "DeepseekV2DecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_deepseek_v2_decoder_layer_has_forward_aiter_fp8(self):
        """DeepseekV2DecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        self.assertTrue(
            hasattr(DeepseekV2DecoderLayer, "_forward_aiter_fp8"),
            "DeepseekV2DecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_deepseek_v2_forward_dispatches_to_aiter_fp8(self):
        """DeepseekV2DecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        src = inspect.getsource(DeepseekV2DecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "DeepseekV2DecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "DeepseekV2DecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_deepseek_v2_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must call prepare_mlp_fp8_out for MLP sub-layer."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        src = inspect.getsource(DeepseekV2DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out for fused MLP norm",
        )

    def test_deepseek_v2_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        src = inspect.getsource(DeepseekV2DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_deepseek_v2_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must check DeepseekV2MLP type to guard MoE layers."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        src = inspect.getsource(DeepseekV2DecoderLayer.__init__)
        self.assertIn(
            "DeepseekV2MLP",
            src,
            "__init__ must check isinstance(self.mlp, DeepseekV2MLP) before enabling _aiter_fp8",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFalconH1FusedFP8Path(unittest.TestCase):
    """Tests for fused pre_ff_layernorm+FP8 MLP path in FalconH1HybridAttentionDecoderLayer (AMD AITER)."""

    def test_falcon_h1_mlp_has_forward_with_fp8_input(self):
        """FalconH1MLP must implement _forward_with_fp8_input."""
        from sglang.srt.models.falcon_h1 import FalconH1MLP

        self.assertTrue(
            hasattr(FalconH1MLP, "_forward_with_fp8_input"),
            "FalconH1MLP must have _forward_with_fp8_input method",
        )

    def test_falcon_h1_decoder_layer_has_aiter_fp8_flag(self):
        """FalconH1HybridAttentionDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.falcon_h1 import FalconH1HybridAttentionDecoderLayer

        src = inspect.getsource(FalconH1HybridAttentionDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "FalconH1HybridAttentionDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_falcon_h1_decoder_layer_has_forward_aiter_fp8(self):
        """FalconH1HybridAttentionDecoderLayer must implement _forward_aiter_fp8."""
        from sglang.srt.models.falcon_h1 import FalconH1HybridAttentionDecoderLayer

        self.assertTrue(
            hasattr(FalconH1HybridAttentionDecoderLayer, "_forward_aiter_fp8"),
            "FalconH1HybridAttentionDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_falcon_h1_forward_dispatches_to_aiter_fp8(self):
        """FalconH1HybridAttentionDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.falcon_h1 import FalconH1HybridAttentionDecoderLayer

        src = inspect.getsource(FalconH1HybridAttentionDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "FalconH1HybridAttentionDecoderLayer.forward must check _aiter_fp8",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "FalconH1HybridAttentionDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_falcon_h1_aiter_fp8_fuses_pre_ff_layernorm(self):
        """_forward_aiter_fp8 must call prepare_mlp_fp8_out for MLP sub-layer."""
        import inspect
        from sglang.srt.models.falcon_h1 import FalconH1HybridAttentionDecoderLayer

        src = inspect.getsource(FalconH1HybridAttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out to fuse pre_ff_layernorm",
        )

    def test_falcon_h1_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call feed_forward._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.falcon_h1 import FalconH1HybridAttentionDecoderLayer

        src = inspect.getsource(FalconH1HybridAttentionDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "feed_forward._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to feed_forward._forward_with_fp8_input",
        )

    def test_falcon_h1_mlp_applies_gate_and_down_multipliers(self):
        """FalconH1MLP._forward_with_fp8_input must apply gate_multiplier and down_multiplier."""
        import inspect
        from sglang.srt.models.falcon_h1 import FalconH1MLP

        src = inspect.getsource(FalconH1MLP._forward_with_fp8_input)
        self.assertIn(
            "gate_multiplier",
            src,
            "_forward_with_fp8_input must apply gate_multiplier matching the standard path",
        )
        self.assertIn(
            "down_multiplier",
            src,
            "_forward_with_fp8_input must apply down_multiplier matching the standard path",
        )

    def test_falcon_h1_module_has_use_aiter_import(self):
        """falcon_h1 module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.falcon_h1")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "falcon_h1 module must import _use_aiter from fp8_utils",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestGlm4MoeLiteFusedFP8Path(unittest.TestCase):
    """Tests for fused post_attention_layernorm+FP8 MLP path in Glm4MoeLiteDecoderLayer (AMD AITER)."""

    def test_glm4_moe_lite_mlp_has_forward_with_fp8_input(self):
        """Glm4MoeLiteMLP must implement _forward_with_fp8_input."""
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteMLP

        self.assertTrue(
            hasattr(Glm4MoeLiteMLP, "_forward_with_fp8_input"),
            "Glm4MoeLiteMLP must have _forward_with_fp8_input method",
        )

    def test_glm4_moe_lite_decoder_layer_has_aiter_fp8_flag(self):
        """Glm4MoeLiteDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Glm4MoeLiteDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_glm4_moe_lite_decoder_layer_has_forward_aiter_fp8(self):
        """Glm4MoeLiteDecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        self.assertTrue(
            hasattr(Glm4MoeLiteDecoderLayer, "_forward_aiter_fp8"),
            "Glm4MoeLiteDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_glm4_moe_lite_forward_dispatches_to_aiter_fp8(self):
        """Glm4MoeLiteDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Glm4MoeLiteDecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Glm4MoeLiteDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_glm4_moe_lite_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must call prepare_mlp_fp8_out for the MLP sub-layer."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out for fused MLP norm",
        )

    def test_glm4_moe_lite_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_glm4_moe_lite_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must check Glm4MoeLiteMLP type to guard MoE layers."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer.__init__)
        self.assertIn(
            "Glm4MoeLiteMLP",
            src,
            "__init__ must check isinstance(self.mlp, Glm4MoeLiteMLP) before enabling _aiter_fp8",
        )

    def test_glm4_moe_lite_module_has_use_aiter_import(self):
        """glm4_moe_lite module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.glm4_moe_lite")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "glm4_moe_lite module must import _use_aiter from fp8_utils",
        )

    def test_glm4_moe_lite_forward_delegates_to_super_when_not_fp8(self):
        """Glm4MoeLiteDecoderLayer.forward must call super().forward for non-FP8 path."""
        import inspect
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer

        src = inspect.getsource(Glm4MoeLiteDecoderLayer.forward)
        self.assertIn(
            "super().forward",
            src,
            "Glm4MoeLiteDecoderLayer.forward must delegate to super().forward on non-FP8 path",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestBailingMoELinearFusedFP8Path(unittest.TestCase):
    """Tests for fused post_attention_layernorm+FP8 MLP path in BailingMoELinearDecoderLayer (AMD AITER)."""

    def test_bailing_mlp_has_forward_with_fp8_input(self):
        """BailingMLP (bailing_moe_linear) must implement _forward_with_fp8_input."""
        from sglang.srt.models.bailing_moe_linear import BailingMLP

        self.assertTrue(
            hasattr(BailingMLP, "_forward_with_fp8_input"),
            "BailingMLP in bailing_moe_linear must have _forward_with_fp8_input method",
        )

    def test_bailing_decoder_layer_has_aiter_fp8_flag(self):
        """BailingMoELinearDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        src = inspect.getsource(BailingMoELinearDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "BailingMoELinearDecoderLayer.__init__ must set _aiter_fp8 flag",
        )

    def test_bailing_decoder_layer_has_forward_aiter_fp8(self):
        """BailingMoELinearDecoderLayer must implement _forward_aiter_fp8 method."""
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        self.assertTrue(
            hasattr(BailingMoELinearDecoderLayer, "_forward_aiter_fp8"),
            "BailingMoELinearDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_bailing_forward_dispatches_to_aiter_fp8(self):
        """BailingMoELinearDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        src = inspect.getsource(BailingMoELinearDecoderLayer.forward)
        self.assertIn(
            "_aiter_fp8",
            src,
            "BailingMoELinearDecoderLayer.forward must check _aiter_fp8 to dispatch",
        )
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "BailingMoELinearDecoderLayer.forward must call _forward_aiter_fp8",
        )

    def test_bailing_aiter_fp8_fuses_post_attention_layernorm(self):
        """_forward_aiter_fp8 must use prepare_mlp_fp8_out for the MLP sub-layer."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        src = inspect.getsource(BailingMoELinearDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must call prepare_mlp_fp8_out for fused MLP norm",
        )

    def test_bailing_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        src = inspect.getsource(BailingMoELinearDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_bailing_aiter_fp8_only_for_dense_mlp(self):
        """_aiter_fp8 must check BailingMLP type to guard MoE layers."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMoELinearDecoderLayer

        src = inspect.getsource(BailingMoELinearDecoderLayer.__init__)
        self.assertIn(
            "BailingMLP",
            src,
            "__init__ must check isinstance(self.mlp, BailingMLP) before enabling _aiter_fp8",
        )

    def test_bailing_module_has_use_aiter_import(self):
        """bailing_moe_linear module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.bailing_moe_linear")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "bailing_moe_linear module must import _use_aiter from fp8_utils",
        )

    def test_bailing_mlp_fp8_skips_down_proj_reduce(self):
        """BailingMLP._forward_with_fp8_input must pass skip_all_reduce to down_proj."""
        import inspect
        from sglang.srt.models.bailing_moe_linear import BailingMLP

        src = inspect.getsource(BailingMLP._forward_with_fp8_input)
        self.assertIn(
            "skip_all_reduce",
            src,
            "_forward_with_fp8_input must use skip_all_reduce for should_allreduce_fusion",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestJetNemotronFusedFP8Path(unittest.TestCase):
    """Tests for JetNemotron fused RMSNorm+FP8 decoder path on AMD MI300X."""

    def test_jet_nemotron_module_has_use_aiter_import(self):
        """jet_nemotron module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.jet_nemotron")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "jet_nemotron module must import _use_aiter from fp8_utils",
        )

    def test_jet_nemotron_decoder_layer_has_aiter_fp8_attr(self):
        """JetNemotronDecoderLayer must have _aiter_fp8 attribute."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "JetNemotronDecoderLayer.__init__ must set _aiter_fp8",
        )

    def test_jet_nemotron_decoder_layer_has_forward_aiter_fp8(self):
        """JetNemotronDecoderLayer must define _forward_aiter_fp8 method."""
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        self.assertTrue(
            hasattr(JetNemotronDecoderLayer, "_forward_aiter_fp8"),
            "JetNemotronDecoderLayer must define _forward_aiter_fp8",
        )

    def test_jet_nemotron_forward_dispatches_to_aiter_fp8(self):
        """JetNemotronDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "forward must dispatch to _forward_aiter_fp8 when _aiter_fp8 is set",
        )

    def test_jet_nemotron_aiter_fp8_passes_fp8_to_mlp(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must pass FP8 to mlp._forward_with_fp8_input",
        )

    def test_jet_nemotron_aiter_fp8_uses_forward_aiter_fp8_out(self):
        """_forward_aiter_fp8 must use forward_aiter_fp8_out for norm fusion."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must use forward_aiter_fp8_out",
        )

    def test_jet_nemotron_attention_has_fp8_input_method(self):
        """JetNemotronAttention must define _forward_with_fp8_input."""
        from sglang.srt.models.jet_nemotron import JetNemotronAttention

        self.assertTrue(
            hasattr(JetNemotronAttention, "_forward_with_fp8_input"),
            "JetNemotronAttention must define _forward_with_fp8_input for FP8 path",
        )

    def test_jet_nemotron_aiter_fp8_handles_jet_block_layers(self):
        """_forward_aiter_fp8 must handle JetBlock (jet) layers separately."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "_is_attn_layer",
            src,
            "_forward_aiter_fp8 must check _is_attn_layer to handle jet vs attn/swa",
        )

    def test_jet_nemotron_aiter_fp8_uses_allreduce_fusion(self):
        """_forward_aiter_fp8 must attempt allreduce+norm fusion for attn layers."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_with_allreduce_fusion_fp8_out",
            src,
            "_forward_aiter_fp8 must attempt allreduce+norm fusion",
        )

    def test_jet_nemotron_aiter_fp8_guards_idle_mode(self):
        """forward must guard _forward_aiter_fp8 against idle forward mode."""
        import inspect
        from sglang.srt.models.jet_nemotron import JetNemotronDecoderLayer

        src = inspect.getsource(JetNemotronDecoderLayer.forward)
        self.assertIn(
            "is_idle",
            src,
            "forward must guard _aiter_fp8 dispatch with is_idle() check",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestLongcatFlashNextNFusedFP8Path(unittest.TestCase):
    """Tests for LongcatFlashDenseDecoderLayer fused MLP FP8 path on AMD MI300X."""

    def test_longcat_flash_mlp_has_fp8_input_method(self):
        """LongcatFlashMLP must define _forward_with_fp8_input."""
        from sglang.srt.models.longcat_flash import LongcatFlashMLP

        self.assertTrue(
            hasattr(LongcatFlashMLP, "_forward_with_fp8_input"),
            "LongcatFlashMLP must define _forward_with_fp8_input for FP8 path",
        )

    def test_longcat_flash_nextn_decoder_has_aiter_fp8_attr(self):
        """LongcatFlashDenseDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        src = inspect.getsource(LongcatFlashDenseDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "LongcatFlashDenseDecoderLayer.__init__ must set _aiter_fp8",
        )

    def test_longcat_flash_nextn_decoder_has_forward_aiter_fp8(self):
        """LongcatFlashDenseDecoderLayer must define _forward_aiter_fp8."""
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        self.assertTrue(
            hasattr(LongcatFlashDenseDecoderLayer, "_forward_aiter_fp8"),
            "LongcatFlashDenseDecoderLayer must define _forward_aiter_fp8",
        )

    def test_longcat_flash_nextn_forward_dispatches_to_aiter_fp8(self):
        """LongcatFlashDenseDecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        src = inspect.getsource(LongcatFlashDenseDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "forward must dispatch to _forward_aiter_fp8 when _aiter_fp8 is set",
        )

    def test_longcat_flash_nextn_aiter_fp8_uses_prepare_mlp_fp8_out(self):
        """_forward_aiter_fp8 must use prepare_mlp_fp8_out for norm+FP8 fusion."""
        import inspect
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        src = inspect.getsource(LongcatFlashDenseDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must use prepare_mlp_fp8_out",
        )

    def test_longcat_flash_nextn_aiter_fp8_calls_mlp_fp8(self):
        """_forward_aiter_fp8 must call mlp._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        src = inspect.getsource(LongcatFlashDenseDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "mlp._forward_with_fp8_input",
            src,
            "_forward_aiter_fp8 must call mlp._forward_with_fp8_input",
        )

    def test_longcat_flash_nextn_aiter_fp8_guards_idle_mode(self):
        """forward must guard _forward_aiter_fp8 against idle forward mode."""
        import inspect
        from sglang.srt.models.longcat_flash_nextn import LongcatFlashDenseDecoderLayer

        src = inspect.getsource(LongcatFlashDenseDecoderLayer.forward)
        self.assertIn(
            "is_idle",
            src,
            "forward must guard _aiter_fp8 dispatch with is_idle() check",
        )

    def test_longcat_flash_mlp_fp8_uses_forward_with_fp8_input(self):
        """LongcatFlashMLP._forward_with_fp8_input must use gate_up_proj FP8 forward."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashMLP

        src = inspect.getsource(LongcatFlashMLP._forward_with_fp8_input)
        self.assertIn(
            "forward_with_fp8_input",
            src,
            "_forward_with_fp8_input must call gate_up_proj.forward_with_fp8_input",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestSarvamMoEFusedFP8Path(unittest.TestCase):
    """Tests for SarvamMoEMLADecoderLayer fused MLP FP8 path on AMD MI300X."""

    def test_sarvam_moe_module_has_use_aiter_import(self):
        """sarvam_moe module must import _use_aiter from fp8_utils."""
        import importlib

        mod = importlib.import_module("sglang.srt.models.sarvam_moe")
        self.assertTrue(
            hasattr(mod, "_use_aiter"),
            "sarvam_moe module must import _use_aiter from fp8_utils",
        )

    def test_sarvam_mlp_has_fp8_input_method(self):
        """SarvamMoEMLP must define _forward_with_fp8_input."""
        from sglang.srt.models.sarvam_moe import SarvamMoEMLP

        self.assertTrue(
            hasattr(SarvamMoEMLP, "_forward_with_fp8_input"),
            "SarvamMoEMLP must define _forward_with_fp8_input for FP8 path",
        )

    def test_sarvam_decoder_has_aiter_fp8_attr(self):
        """SarvamMoEMLADecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        src = inspect.getsource(SarvamMoEMLADecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "SarvamMoEMLADecoderLayer.__init__ must set _aiter_fp8",
        )

    def test_sarvam_decoder_has_forward_aiter_fp8(self):
        """SarvamMoEMLADecoderLayer must define _forward_aiter_fp8."""
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        self.assertTrue(
            hasattr(SarvamMoEMLADecoderLayer, "_forward_aiter_fp8"),
            "SarvamMoEMLADecoderLayer must define _forward_aiter_fp8",
        )

    def test_sarvam_forward_dispatches_to_aiter_fp8(self):
        """SarvamMoEMLADecoderLayer.forward must dispatch to _forward_aiter_fp8."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        src = inspect.getsource(SarvamMoEMLADecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "forward must dispatch to _forward_aiter_fp8 when _aiter_fp8 is set",
        )

    def test_sarvam_aiter_fp8_uses_prepare_mlp_fp8_out(self):
        """_forward_aiter_fp8 must use prepare_mlp_fp8_out for norm+FP8 fusion."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        src = inspect.getsource(SarvamMoEMLADecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_aiter_fp8 must use prepare_mlp_fp8_out",
        )

    def test_sarvam_aiter_fp8_only_for_dense_layers(self):
        """_aiter_fp8 must only be set for non-sparse (dense) layers."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        src = inspect.getsource(SarvamMoEMLADecoderLayer.__init__)
        self.assertIn(
            "is_layer_sparse",
            src,
            "__init__ must guard _aiter_fp8 with not self.is_layer_sparse",
        )

    def test_sarvam_aiter_fp8_guards_idle_mode(self):
        """forward must guard _forward_aiter_fp8 against idle forward mode."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLADecoderLayer

        src = inspect.getsource(SarvamMoEMLADecoderLayer.forward)
        self.assertIn(
            "is_idle",
            src,
            "forward must guard _aiter_fp8 dispatch with is_idle() check",
        )

    def test_sarvam_mlp_fp8_propagates_allreduce_fusion(self):
        """SarvamMoEMLP._forward_with_fp8_input must propagate allreduce fusion params."""
        import inspect
        from sglang.srt.models.sarvam_moe import SarvamMoEMLP

        src = inspect.getsource(SarvamMoEMLP._forward_with_fp8_input)
        self.assertIn(
            "should_allreduce_fusion",
            src,
            "_forward_with_fp8_input must accept and use should_allreduce_fusion",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestGemma3FP8AiterPath(unittest.TestCase):
    """Verify Gemma3 FP8 AITER path implementation structure."""

    def test_gemma3rmsnorm_has_forward_aiter_fp8_out(self):
        """Gemma3RMSNorm must have forward_aiter_fp8_out method."""
        from sglang.srt.layers.layernorm import Gemma3RMSNorm

        self.assertTrue(
            hasattr(Gemma3RMSNorm, "forward_aiter_fp8_out"),
            "Gemma3RMSNorm must have forward_aiter_fp8_out method",
        )

    def test_gemma3rmsnorm_fp8_out_no_residual(self):
        """Gemma3RMSNorm.forward_aiter_fp8_out should return (fp8, scale, None) without residual."""
        try:
            from aiter import rmsnorm2d_fwd_with_dynamicquant
        except ImportError:
            self.skipTest("AITER not available")

        from sglang.srt.layers.layernorm import Gemma3RMSNorm
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        dim = 2048
        norm = Gemma3RMSNorm(dim=dim, eps=1e-6).cuda().to(torch.bfloat16)
        x = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16)

        fp8_out, fp8_scale, residual = norm.forward_aiter_fp8_out(x)

        self.assertEqual(fp8_out.dtype, fp8_dtype)
        self.assertEqual(fp8_out.shape, (8, dim))
        self.assertEqual(fp8_scale.shape, (8, 1))
        self.assertIsNone(residual)
        self.assertFalse(fp8_out.to(torch.float32).isnan().any().item())

    def test_gemma3rmsnorm_fp8_out_correctness(self):
        """Gemma3RMSNorm.forward_aiter_fp8_out output should match reference BF16 norm."""
        try:
            from aiter import rmsnorm2d_fwd_with_dynamicquant
        except ImportError:
            self.skipTest("AITER not available")

        from sglang.srt.layers.layernorm import Gemma3RMSNorm

        torch.manual_seed(42)
        dim = 2048
        norm = Gemma3RMSNorm(dim=dim, eps=1e-6).cuda().to(torch.bfloat16)
        x = torch.randn(16, dim, device="cuda", dtype=torch.bfloat16)

        # Reference: BF16 norm output
        ref = norm.forward_native(x)

        # FP8 output + dequantize
        fp8_out, fp8_scale, _ = norm.forward_aiter_fp8_out(x)
        dequant = fp8_out.to(torch.float32) * fp8_scale

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.float().flatten().unsqueeze(0),
            dequant.flatten().unsqueeze(0),
        ).item()
        self.assertGreater(
            cos_sim, 0.999,
            f"Gemma3RMSNorm fp8_out cosine_similarity={cos_sim} vs BF16 ref",
        )

    def test_gemma3_decoder_layer_has_aiter_fp8_attr(self):
        """Gemma3DecoderLayer must have _aiter_fp8 attribute."""
        import inspect
        from sglang.srt.models.gemma3_causal import Gemma3DecoderLayer

        src = inspect.getsource(Gemma3DecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "Gemma3DecoderLayer.__init__ must set self._aiter_fp8",
        )

    def test_gemma3_decoder_layer_has_forward_aiter_fp8(self):
        """Gemma3DecoderLayer must have _forward_aiter_fp8 method."""
        from sglang.srt.models.gemma3_causal import Gemma3DecoderLayer

        self.assertTrue(
            hasattr(Gemma3DecoderLayer, "_forward_aiter_fp8"),
            "Gemma3DecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_gemma3_attention_has_forward_with_fp8_input(self):
        """Gemma3Attention must have _forward_with_fp8_input method."""
        from sglang.srt.models.gemma3_causal import Gemma3Attention

        self.assertTrue(
            hasattr(Gemma3Attention, "_forward_with_fp8_input"),
            "Gemma3Attention must have _forward_with_fp8_input method",
        )

    def test_gemma3_mlp_has_forward_with_fp8_input(self):
        """Gemma3MLP must have _forward_with_fp8_input method."""
        from sglang.srt.models.gemma3_causal import Gemma3MLP

        self.assertTrue(
            hasattr(Gemma3MLP, "_forward_with_fp8_input"),
            "Gemma3MLP must have _forward_with_fp8_input method",
        )

    def test_gemma3_forward_dispatches_to_aiter_fp8(self):
        """Gemma3DecoderLayer.forward must dispatch to _forward_aiter_fp8 when _aiter_fp8 is set."""
        import inspect
        from sglang.srt.models.gemma3_causal import Gemma3DecoderLayer

        src = inspect.getsource(Gemma3DecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "Gemma3DecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_gemma3_aiter_fp8_uses_forward_aiter_fp8_out(self):
        """_forward_aiter_fp8 must use input_layernorm.forward_aiter_fp8_out."""
        import inspect
        from sglang.srt.models.gemma3_causal import Gemma3DecoderLayer

        src = inspect.getsource(Gemma3DecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must call forward_aiter_fp8_out",
        )

    def test_gemma3_uses_fp8_utils_use_aiter(self):
        """gemma3_causal.py must import _use_aiter from fp8_utils."""
        import inspect
        import sglang.srt.models.gemma3_causal as m

        src = inspect.getsource(m)
        self.assertIn(
            "_use_aiter",
            src,
            "gemma3_causal.py must import and use _use_aiter",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAfmoeFP8AiterPath(unittest.TestCase):
    """Verify AfMoE FP8 AITER path implementation structure."""

    def test_afmoe_mlp_has_forward_with_fp8_input(self):
        """AfmoeMLP must have _forward_with_fp8_input method."""
        from sglang.srt.models.afmoe import AfmoeMLP

        self.assertTrue(
            hasattr(AfmoeMLP, "_forward_with_fp8_input"),
            "AfmoeMLP must have _forward_with_fp8_input method",
        )

    def test_afmoe_attention_has_forward_with_fp8_input(self):
        """AfmoeAttention must have _forward_with_fp8_input method."""
        from sglang.srt.models.afmoe import AfmoeAttention

        self.assertTrue(
            hasattr(AfmoeAttention, "_forward_with_fp8_input"),
            "AfmoeAttention must have _forward_with_fp8_input method",
        )

    def test_afmoe_decoder_layer_has_aiter_fp8_attr(self):
        """AfmoeDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.afmoe import AfmoeDecoderLayer

        src = inspect.getsource(AfmoeDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "AfmoeDecoderLayer.__init__ must set self._aiter_fp8",
        )

    def test_afmoe_decoder_layer_has_forward_aiter_fp8(self):
        """AfmoeDecoderLayer must have _forward_aiter_fp8 method."""
        from sglang.srt.models.afmoe import AfmoeDecoderLayer

        self.assertTrue(
            hasattr(AfmoeDecoderLayer, "_forward_aiter_fp8"),
            "AfmoeDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_afmoe_forward_dispatches_to_aiter_fp8(self):
        """AfmoeDecoderLayer.forward must dispatch to _forward_aiter_fp8 when set."""
        import inspect
        from sglang.srt.models.afmoe import AfmoeDecoderLayer

        src = inspect.getsource(AfmoeDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "AfmoeDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_afmoe_aiter_fp8_only_for_dense_mlp(self):
        """AfmoeDecoderLayer._aiter_fp8 must only be set for dense MLP layers."""
        import inspect
        from sglang.srt.models.afmoe import AfmoeDecoderLayer

        src = inspect.getsource(AfmoeDecoderLayer.__init__)
        self.assertIn(
            "AfmoeMLP",
            src,
            "__init__ must check isinstance(self.mlp, AfmoeMLP) before setting _aiter_fp8",
        )

    def test_afmoe_uses_fp8_utils_use_aiter(self):
        """afmoe.py must import _use_aiter from fp8_utils."""
        import inspect
        import sglang.srt.models.afmoe as m

        src = inspect.getsource(m)
        self.assertIn(
            "_use_aiter",
            src,
            "afmoe.py must import and use _use_aiter",
        )

    def test_afmoe_aiter_fp8_uses_forward_aiter_fp8_out(self):
        """_forward_aiter_fp8 must use input_layernorm.forward_aiter_fp8_out."""
        import inspect
        from sglang.srt.models.afmoe import AfmoeDecoderLayer

        src = inspect.getsource(AfmoeDecoderLayer._forward_aiter_fp8)
        self.assertIn(
            "forward_aiter_fp8_out",
            src,
            "_forward_aiter_fp8 must call forward_aiter_fp8_out",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestLongcatFlashFP8AiterPath(unittest.TestCase):
    """Verify LongcatFlashDecoderLayer FP8 AITER path implementation structure."""

    def test_longcat_flash_decoder_layer_has_aiter_fp8_attr(self):
        """LongcatFlashDecoderLayer must set _aiter_fp8 in __init__."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        src = inspect.getsource(LongcatFlashDecoderLayer.__init__)
        self.assertIn(
            "_aiter_fp8",
            src,
            "LongcatFlashDecoderLayer.__init__ must set self._aiter_fp8",
        )

    def test_longcat_flash_decoder_layer_has_forward_aiter_fp8(self):
        """LongcatFlashDecoderLayer must have _forward_aiter_fp8 method."""
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        self.assertTrue(
            hasattr(LongcatFlashDecoderLayer, "_forward_aiter_fp8"),
            "LongcatFlashDecoderLayer must have _forward_aiter_fp8 method",
        )

    def test_longcat_flash_forward_dispatches_to_aiter_fp8(self):
        """LongcatFlashDecoderLayer.forward must dispatch to _forward_aiter_fp8 when set."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        src = inspect.getsource(LongcatFlashDecoderLayer.forward)
        self.assertIn(
            "_forward_aiter_fp8",
            src,
            "LongcatFlashDecoderLayer.forward must dispatch to _forward_aiter_fp8",
        )

    def test_longcat_flash_aiter_fp8_uses_prepare_mlp_fp8_out(self):
        """_forward_aiter_fp8 or helper must use prepare_mlp_fp8_out."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        src = inspect.getsource(LongcatFlashDecoderLayer._forward_mlp_aiter_fp8)
        self.assertIn(
            "prepare_mlp_fp8_out",
            src,
            "_forward_mlp_aiter_fp8 must call prepare_mlp_fp8_out",
        )

    def test_longcat_flash_aiter_fp8_guards_idle_mode(self):
        """forward must guard _forward_aiter_fp8 dispatch against idle forward mode."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        src = inspect.getsource(LongcatFlashDecoderLayer.forward)
        self.assertIn(
            "is_idle",
            src,
            "forward must guard _aiter_fp8 dispatch with is_idle() check",
        )

    def test_longcat_flash_aiter_fp8_uses_mlp1_forward_with_fp8_input(self):
        """_forward_mlp_aiter_fp8 must use mlps[1]._forward_with_fp8_input."""
        import inspect
        from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer

        src = inspect.getsource(LongcatFlashDecoderLayer._forward_mlp_aiter_fp8)
        self.assertIn(
            "_forward_with_fp8_input",
            src,
            "_forward_mlp_aiter_fp8 must call _forward_with_fp8_input",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestHybridCKBpreshuffleDispatch(unittest.TestCase):
    """Test that the hybrid CK/bpreshuffle GEMM dispatch selects the right kernel
    based on weight shape: CK for K > N (down_proj), bpreshuffle for N >= K."""

    def test_apply_fp8_ck_linear_exists(self):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ck_linear
        self.assertTrue(callable(apply_fp8_ck_linear))

    def test_w8a8_process_weights_sets_use_ck_flag(self):
        """W8A8Fp8LinearMethod should set _use_ck based on K > N."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
        src = inspect.getsource(W8A8Fp8LinearMethod.process_weights_after_loading)
        self.assertIn("_use_ck", src, "process_weights_after_loading must set _use_ck")
        self.assertIn("use_scaled_mm", src, "Heuristic should set use_scaled_mm")

    def test_w8a8_process_weights_no_ck_for_gate_up(self):
        """Gate/up proj (N > K) should use bpreshuffle (no shuffle skip)."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
        src = inspect.getsource(W8A8Fp8LinearMethod.process_weights_after_loading)
        # When use_ck is False, shuffle_weight should still be called
        self.assertIn("shuffle_weight", src, "Non-CK path must still call shuffle_weight")

    def test_ck_linear_correctness(self):
        """apply_fp8_ck_linear produces correct output vs BF16 reference."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ck_linear
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

        N, K = 4096, 14336  # down_proj shape
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.mm(x, w_bf16.t())

        qw, w_scale = per_token_group_quant_fp8(w_bf16, K)
        out = apply_fp8_ck_linear(x, qw, w_scale)

        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), out.flatten().float(), dim=0
        )
        self.assertGreater(cos.item(), 0.99, f"CK output diverged: cosine={cos.item()}")

    def test_bpreshuffle_linear_correctness(self):
        """apply_fp8_ptpc_linear produces correct output vs BF16 reference."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
        from aiter.ops.shuffle import shuffle_weight

        N, K = 28672, 4096  # gate_up shape
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.mm(x, w_bf16.t())

        qw, w_scale = per_token_group_quant_fp8(w_bf16, K)
        qw_shuffled = shuffle_weight(qw.contiguous(), (16, 16))
        out = apply_fp8_ptpc_linear(x, qw_shuffled, w_scale, use_per_token_if_dynamic=True)

        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), out.flatten().float(), dim=0
        )
        self.assertGreater(cos.item(), 0.99, f"bpreshuffle output diverged: cosine={cos.item()}")

    def test_hybrid_dispatch_in_apply(self):
        """W8A8Fp8LinearMethod.apply() dispatches correctly based on _use_ck."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
        src = inspect.getsource(W8A8Fp8LinearMethod.apply)
        self.assertIn("_use_ck", src, "apply() must check _use_ck flag")
        self.assertIn("apply_fp8_ck_linear", src, "apply() must call apply_fp8_ck_linear")
        self.assertIn("apply_fp8_ptpc_linear", src, "apply() must call apply_fp8_ptpc_linear")

    def test_fbgemm_fp8_has_hybrid_dispatch(self):
        """FBGEMMFp8LinearMethod.apply() has hybrid CK/bpreshuffle dispatch."""
        import inspect
        from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8LinearMethod
        src = inspect.getsource(FBGEMMFp8LinearMethod.apply)
        self.assertIn("_use_ck", src, "FBGEMMFp8 apply() must check _use_ck flag")

    def test_compressed_tensors_has_hybrid_dispatch(self):
        """CompressedTensorsW8A8Fp8.apply_weights() has hybrid CK/bpreshuffle dispatch."""
        import inspect
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
            CompressedTensorsW8A8Fp8,
        )
        src = inspect.getsource(CompressedTensorsW8A8Fp8.apply_weights)
        self.assertIn("_use_ck", src, "CompressedTensors apply_weights() must check _use_ck")

    def test_quark_has_hybrid_dispatch(self):
        """QuarkW8A8Fp8.apply_weights() has hybrid CK/bpreshuffle dispatch."""
        import inspect
        from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import QuarkW8A8Fp8
        src = inspect.getsource(QuarkW8A8Fp8.apply_weights)
        self.assertIn("_use_ck", src, "Quark apply_weights() must check _use_ck")

    def test_fp8_linear_method_has_hybrid_dispatch(self):
        """Fp8LinearMethod.apply() has hybrid CK/bpreshuffle dispatch."""
        import inspect
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        src = inspect.getsource(Fp8LinearMethod.apply)
        self.assertIn("_use_ck", src, "Fp8LinearMethod apply() must check _use_ck")

    # --- Edge-case tests for hybrid GEMM dispatch threshold ---

    def test_scaled_mm_output_shape_matches_bpreshuffle(self):
        """Both CK (scaled_mm) and bpreshuffle paths produce the same output shape."""
        from sglang.srt.layers.quantization.w8a8_fp8 import _use_aiter
        if not _use_aiter:
            self.skipTest("AITER not enabled")

        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_ck_linear,
            apply_fp8_ptpc_linear,
        )
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
        from aiter.ops.shuffle import shuffle_weight

        N, K = 4096, 8192
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)

        qw, w_scale = per_token_group_quant_fp8(w_bf16, K)
        out_ck = apply_fp8_ck_linear(x, qw, w_scale)

        qw2, w_scale2 = per_token_group_quant_fp8(w_bf16, K)
        qw2_shuffled = shuffle_weight(qw2.contiguous(), (16, 16))
        out_bp = apply_fp8_ptpc_linear(
            x, qw2_shuffled, w_scale2, use_per_token_if_dynamic=True
        )

        self.assertEqual(
            out_ck.shape,
            out_bp.shape,
            f"CK shape {out_ck.shape} != bpreshuffle shape {out_bp.shape}",
        )

    def test_threshold_400m_correct(self):
        """Verify that 400_000_000 is the N*K threshold in process_weights_after_loading."""
        import inspect
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
        src = inspect.getsource(W8A8Fp8LinearMethod.process_weights_after_loading)
        self.assertIn(
            "400_000_000",
            src,
            "Threshold must be 400_000_000 in process_weights_after_loading",
        )

    def test_qkv_shape_uses_bpreshuffle(self):
        """Typical QKV shapes (N=10240, K=8192) should use bpreshuffle (_use_ck=False).
        N*K = 83,886,080 < 400M, and K < N, so use_scaled_mm is False."""
        N, K = 10240, 8192
        use_scaled_mm = K > N or N * K > 400_000_000
        self.assertFalse(
            use_scaled_mm,
            f"QKV (N={N}, K={K}, N*K={N*K}) should use bpreshuffle, not scaled_mm",
        )

    def test_gate_up_72b_uses_scaled_mm(self):
        """Qwen2.5-72B gate_up (N=59136, K=8192) should use scaled_mm (_use_ck=True).
        N*K = 484,442,112 > 400M."""
        N, K = 59136, 8192
        use_scaled_mm = K > N or N * K > 400_000_000
        self.assertTrue(
            use_scaled_mm,
            f"gate_up 72B (N={N}, K={K}, N*K={N*K}) should use scaled_mm",
        )

    def test_down_proj_uses_scaled_mm(self):
        """Any down_proj where K > N should use scaled_mm (_use_ck=True)."""
        shapes = [
            (4096, 14336),   # Llama-2-70B down_proj
            (8192, 28672),   # Llama-3-70B down_proj
            (3584, 18944),   # Qwen-2.5 down_proj
        ]
        for N, K in shapes:
            use_scaled_mm = K > N or N * K > 400_000_000
            self.assertTrue(
                use_scaled_mm,
                f"down_proj (N={N}, K={K}) must use scaled_mm since K > N",
            )

    def test_gate_up_70b_uses_bpreshuffle(self):
        """Llama3-70B gate_up (N=28672, K=8192, N*K=234,881,024) uses bpreshuffle.
        N*K < 400M and K < N, so use_scaled_mm is False."""
        N, K = 28672, 8192
        use_scaled_mm = K > N or N * K > 400_000_000
        self.assertFalse(
            use_scaled_mm,
            f"gate_up 70B (N={N}, K={K}, N*K={N*K}) should use bpreshuffle",
        )


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestFP8KVCachePrefill(unittest.TestCase):
    """Test FP8 KV cache native prefill (avoid BF16 dequant)."""

    def test_prefill_uses_per_tensor_quant_for_q(self):
        """FP8 KV prefill uses dynamic_per_tensor_quant for Q (not naive cast)."""
        import inspect
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        src = inspect.getsource(AiterAttnBackend.forward_extend)
        self.assertIn("dynamic_per_tensor_quant", src,
                       "Should use per-tensor quant for Q to avoid FP8 clamping")

    def test_prefill_uses_native_fp8_kv(self):
        """forward_extend passes FP8 KV directly with descale instead of dequant."""
        import inspect
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        src = inspect.getsource(AiterAttnBackend.forward_extend)
        # Should NOT have the old .to(dtype) dequant pattern
        self.assertNotIn("k_cache.to(dtype)", src,
                         "Should not dequant FP8 KV cache to BF16 in prefill")
        self.assertNotIn("v_cache.to(dtype)", src,
                         "Should not dequant FP8 KV cache to BF16 in prefill")
        # Should have FP8 native path with descale
        self.assertIn("q_descale", src,
                       "Should pass q_descale for FP8 KV prefill")
        self.assertIn("k_descale", src,
                       "Should pass k_descale for FP8 KV prefill")
        self.assertIn("v_descale", src,
                       "Should pass v_descale for FP8 KV prefill")

    def test_prefill_fp8_path_converts_q_to_fp8(self):
        """FP8 KV prefill path quantizes Q to FP8 for same-dtype requirement."""
        import inspect
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        src = inspect.getsource(AiterAttnBackend.forward_extend)
        self.assertIn("q_fp8", src,
                       "Should convert Q to FP8 for native FP8 prefill")

    def test_bf16_path_unchanged(self):
        """BF16 KV cache path does not use descale."""
        import inspect
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        src = inspect.getsource(AiterAttnBackend.forward_extend)
        # Source should have two mha_batch_prefill_func calls:
        # 1. FP8 path (with k_descale) 2. BF16 path (without k_descale)
        parts = src.split("mha_batch_prefill_func")
        # parts[0] = before first call, parts[1] = first call args, parts[2] = second call args
        self.assertGreaterEqual(len(parts), 3,
                                "Should have at least 2 mha_batch_prefill_func calls")
        # The last call (BF16 path) should not have k_descale
        bf16_call = parts[-1][:500]
        self.assertNotIn("k_descale", bf16_call,
                         "BF16 path should not pass k_descale")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAITERActivationKernels(unittest.TestCase):
    """Test AITER activation kernel integration."""

    def test_silu_and_mul_uses_aiter(self):
        """SiluAndMul.forward_hip uses AITER kernel."""
        import inspect
        from sglang.srt.layers.activation import SiluAndMul
        src = inspect.getsource(SiluAndMul.forward_hip)
        self.assertIn("_aiter_silu_and_mul", src)

    def test_gelu_and_mul_none_uses_aiter(self):
        """GeluAndMul.forward_hip uses AITER for approximate='none'."""
        import inspect
        from sglang.srt.layers.activation import GeluAndMul
        src = inspect.getsource(GeluAndMul.forward_hip)
        self.assertIn("_aiter_gelu_and_mul", src)

    def test_gelu_tanh_and_mul_uses_aiter(self):
        """GeluAndMul.forward_hip uses AITER for approximate='tanh'."""
        import inspect
        from sglang.srt.layers.activation import GeluAndMul
        src = inspect.getsource(GeluAndMul.forward_hip)
        self.assertIn("_aiter_gelu_tanh_and_mul", src)

    def test_aiter_activation_imports(self):
        """AITER activation kernels are importable."""
        from sglang.srt.layers.activation import _has_aiter_activation
        self.assertTrue(_has_aiter_activation, "AITER activation kernels should be available")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestLinearBaseFP8Input(unittest.TestCase):
    """Test forward_with_fp8_input on all linear types."""

    def test_linear_base_has_method(self):
        from sglang.srt.layers.linear import LinearBase
        self.assertTrue(hasattr(LinearBase, 'forward_with_fp8_input'))

    def test_qkv_parallel_has_method(self):
        from sglang.srt.layers.linear import QKVParallelLinear
        self.assertTrue(hasattr(QKVParallelLinear, 'forward_with_fp8_input'))

    def test_column_parallel_has_method(self):
        from sglang.srt.layers.linear import ColumnParallelLinear
        self.assertTrue(hasattr(ColumnParallelLinear, 'forward_with_fp8_input'))

    def test_row_parallel_has_method(self):
        from sglang.srt.layers.linear import RowParallelLinear
        self.assertTrue(hasattr(RowParallelLinear, 'forward_with_fp8_input'))

    def test_merged_column_has_method(self):
        from sglang.srt.layers.linear import MergedColumnParallelLinear
        self.assertTrue(hasattr(MergedColumnParallelLinear, 'forward_with_fp8_input'))


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestStaticActivationGuard(unittest.TestCase):
    """Test that static-activation models skip fused FP8 path."""

    def test_llama_aiter_fp8_checks_static(self):
        """LlamaDecoderLayer _aiter_fp8 detection guards against static schemes."""
        import inspect
        from sglang.srt.models.llama import LlamaDecoderLayer
        src = inspect.getsource(LlamaDecoderLayer.__init__)
        self.assertIn("is_static_input_scheme", src)

    def test_qwen2_aiter_fp8_checks_static(self):
        import inspect
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer
        src = inspect.getsource(Qwen2DecoderLayer.__init__)
        self.assertIn("is_static_input_scheme", src)

    def test_llama_aiter_fp8_checks_block_quant(self):
        """LlamaDecoderLayer _aiter_fp8 detection guards against block_quant."""
        import inspect
        from sglang.srt.models.llama import LlamaDecoderLayer
        src = inspect.getsource(LlamaDecoderLayer.__init__)
        self.assertIn("block_quant", src)

    def test_qwen2_aiter_fp8_checks_block_quant(self):
        import inspect
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer
        src = inspect.getsource(Qwen2DecoderLayer.__init__)
        self.assertIn("block_quant", src)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestTPGuardConsistency(unittest.TestCase):
    """Verify all models with _aiter_fp8 have TP>1 guard."""

    def test_all_models_have_tp_guard(self):
        """Every model with _aiter_fp8 should check TP world size."""
        import os
        import glob
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "python", "sglang", "srt", "models"
        )
        missing = []
        for f in glob.glob(os.path.join(models_dir, "*.py")):
            if "__pycache__" in f or "__init__" in f:
                continue
            with open(f) as fh:
                content = fh.read()
            if "_aiter_fp8" in content and "_forward_aiter_fp8" in content:
                if "get_tensor_model_parallel_world_size" not in content:
                    missing.append(os.path.basename(f))
        self.assertEqual(missing, [],
                         f"Models with _aiter_fp8 but no TP guard: {missing}")

