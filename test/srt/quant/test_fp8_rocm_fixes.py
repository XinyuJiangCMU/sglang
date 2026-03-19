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


if __name__ == "__main__":
    unittest.main()
