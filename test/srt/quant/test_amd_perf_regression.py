"""
AMD MI300X Performance Regression Tests.

Tests kernel-level performance to catch regressions:
1. AITER flash attention TFLOPS at seq=4096
2. AITER GEMM latency for decode (M=1)
3. FP8 quantization latency
4. RMSNorm latency
"""

import time
import unittest

import torch

from sglang.srt.utils import is_hip


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestAttentionPerformance(unittest.TestCase):
    """Test AITER attention kernel performance on MI300X."""

    def test_flash_attn_varlen_tflops(self):
        """flash_attn_varlen_func should achieve > 500 TFLOPS at seq=4096."""
        from aiter import flash_attn_varlen_func

        seq_len = 4096
        H_q, H_kv, D = 32, 32, 128

        q = torch.randn(seq_len, H_q, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq_len, H_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq_len, H_kv, D, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

        # Warmup
        for _ in range(3):
            flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, causal=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, causal=True)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 10 * 1e6

        flops = 4 * 1 * H_q * seq_len * seq_len * D / 1e12
        tflops = flops / (us / 1e6)

        self.assertGreater(
            tflops,
            500.0,
            f"flash_attn_varlen at seq=4096: {tflops:.0f} TFLOPS < 500 threshold",
        )

    def test_mha_batch_prefill_works(self):
        """mha_batch_prefill_func should produce finite output."""
        from aiter import mha_batch_prefill_func

        seq_len = 512
        H_q, H_kv, D = 20, 4, 128

        q = torch.randn(seq_len, H_q, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq_len, H_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq_len, H_kv, D, device="cuda", dtype=torch.bfloat16)
        qo_indptr = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
        kv_indptr = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
        kv_indices = torch.arange(seq_len, device="cuda", dtype=torch.int32)

        out = mha_batch_prefill_func(
            q, k, v, qo_indptr, kv_indptr, kv_indices, seq_len, seq_len, causal=True
        )
        self.assertTrue(out.isfinite().all().item())
        self.assertEqual(out.shape, (seq_len, H_q, D))


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestGEMMPerformance(unittest.TestCase):
    """Test GEMM kernel performance on MI300X."""

    def test_aiter_fp8_gemm_latency(self):
        """AITER FP8 per-channel GEMM should complete in < 50us for M=1."""
        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            self.skipTest("AITER not available")

        M, N, K = 1, 2560, 2560
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        qW, ws = per_token_group_quant_fp8(W, K)
        ws = ws.t().contiguous()
        W_s = shuffle_weight(qW.contiguous(), (16, 16))
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

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

        self.assertLess(us, 50.0, f"FP8 GEMM {us:.1f}us > 50us threshold")

    def test_bf16_gemm_latency(self):
        """BF16 GEMM should complete in < 30us for M=1 N=K=2560."""
        M, N, K = 1, 2560, 2560
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        for _ in range(20):
            torch.mm(A, B)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            torch.mm(A, B)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 100 * 1e6

        self.assertLess(us, 30.0, f"BF16 GEMM {us:.1f}us > 30us threshold")


@unittest.skipIf(not is_hip(), "ROCm-only tests")
class TestRMSNormPerformance(unittest.TestCase):
    """Test RMSNorm performance on MI300X."""

    def test_fused_add_rmsnorm_latency(self):
        """AITER fused_add_rmsnorm should complete in < 15us."""
        from aiter import rmsnorm2d_fwd_with_add

        M, K = 1, 2560
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        res_out = torch.empty_like(x)

        for _ in range(20):
            rmsnorm2d_fwd_with_add(out, x, residual, res_out, w, 1e-6)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            rmsnorm2d_fwd_with_add(out, x, residual, res_out, w, 1e-6)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6

        self.assertLess(us, 15.0, f"fused_add_rmsnorm {us:.1f}us > 15us threshold")


if __name__ == "__main__":
    unittest.main()
