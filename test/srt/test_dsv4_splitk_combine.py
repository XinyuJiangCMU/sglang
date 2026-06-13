"""Correctness tests for the DSV4 split-K combine kernels (gfx950 bf16 path).

These tests cover the bf16 split-K decode optimization that is gated on
gfx950 (MI350X). The key invariant under test is that the general
``_combine_splitk_kernel_any_sk`` (used for sk16) produces the same combined
attention output as the hand-unrolled sk8 combine kernel on equivalent data,
including correct handling of the |lse| >= 1e30 empty-split sentinel.

Test A (implemented): synthetic bf16 partials + fp32 partial_lse. We build an
sk8 problem and an sk16 problem that are mathematically equivalent (the extra
8 splits in the sk16 case are all-empty / +inf-LSE) and assert the combined
outputs and LSE match.

Test B (end-to-end fused decode with synthetic FP8 KV) is left as a TODO: the
segregated per-block FP8 KV layout is fiddly to construct reliably, and a
flaky e2e test is worse than none. The microbenchmark already validates the
e2e numerics against the fp32 reference.
"""

import unittest

import torch
import triton

from sglang.srt.utils import is_gfx95_supported

from sglang.srt.layers.attention.nsa.triton_decode.triton_mla_kernels_decode_fused import (
    _combine_splitk_kernel_8_optimized,
    _combine_splitk_kernel_any_sk,
)
from sglang.srt.layers.attention.nsa.triton_decode.triton_mla_kernels_decode_common import (
    _bucket_total_tokens,
)

INF_SENTINEL = 1e35  # > 1e30 invalid-split threshold used by the kernels


def _run_combine_any_sk(partial_output, partial_lse, split_k, total_tokens, h_q, d_v):
    """Run the general any-sk combine kernel and return (output, lse)."""
    device = partial_output.device
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)
    attn_sink = lse[0, :]  # dummy (HAS_ATTN_SINK=False)

    BLOCK_H = 16
    BLOCK_D = 512
    grid = (total_tokens, triton.cdiv(h_q, BLOCK_H))
    _combine_splitk_kernel_any_sk[grid](
        partial_output,
        partial_lse,
        attn_sink,
        output,
        lse,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        d_v,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=False,
        SPLIT_K=split_k,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=1,
    )
    return output, lse


def _run_combine_sk8(partial_output, partial_lse, total_tokens, h_q, d_v):
    """Run the hand-unrolled sk8 combine kernel and return (output, lse)."""
    device = partial_output.device
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)
    attn_sink = lse[0, :]  # dummy (HAS_ATTN_SINK=False)

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
    _combine_splitk_kernel_8_optimized[grid](
        partial_output,
        partial_lse,
        attn_sink,
        output,
        lse,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        d_v,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=False,
    )
    return output, lse


@unittest.skipIf(
    not is_gfx95_supported(),
    "bf16 split-K decode path is gfx950-only; skip on gfx942/other.",
)
class TestDSV4SplitKCombine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.total_tokens = 4
        self.h_q = 64
        self.d_v = 512

    def test_a_any_sk16_matches_sk8(self):
        """Test A: sk16 combine == sk8 combine on equivalent data.

        Build 8 valid splits (bf16 partials, fp32 LSE). Run them through the
        sk8 kernel. Then build an sk16 problem whose first 8 splits are the
        same data and whose last 8 splits are all-empty (+inf LSE sentinel),
        and run it through the any-sk kernel at SPLIT_K=16. The empty splits
        must be ignored, so outputs and LSE must match.

        Also includes a token row whose first 8 splits are themselves all
        empty (all-+inf LSE) to exercise the all-empty sentinel branch.
        """
        T, H, D = self.total_tokens, self.h_q, self.d_v
        dev = self.device

        # 8 valid splits of bf16 partials + fp32 LSE.
        po8 = torch.randn(8, T, H, D, dtype=torch.float32, device=dev).to(
            torch.bfloat16
        )
        lse8 = torch.randn(8, T, H, dtype=torch.float32, device=dev) * 2.0

        # Make token row 0 an all-empty case: every split's LSE is +inf so the
        # combine must emit the empty-row result (zeros / -inf lse), not NaN.
        lse8[:, 0, :] = INF_SENTINEL

        out8, l8 = _run_combine_sk8(po8.clone(), lse8.clone(), T, H, D)

        # sk16: first 8 splits identical, last 8 splits all empty (+inf lse).
        po16 = torch.zeros(16, T, H, D, dtype=torch.bfloat16, device=dev)
        lse16 = torch.full((16, T, H), INF_SENTINEL, dtype=torch.float32, device=dev)
        po16[:8].copy_(po8)
        lse16[:8].copy_(lse8)

        out16, l16 = _run_combine_any_sk(
            po16.clone(), lse16.clone(), 16, T, H, D
        )

        # Non-empty rows: compare in fp32. Empty row 0 should be identical
        # (both produce the same sentinel output), so include it too.
        self.assertFalse(torch.isnan(out16.float()).any(), "any_sk produced NaN")
        self.assertFalse(torch.isnan(out8.float()).any(), "sk8 produced NaN")

        torch.testing.assert_close(
            out16.float(), out8.float(), atol=5e-3, rtol=5e-3
        )

        # LSE: ignore the empty row (both should be the same sentinel, but the
        # exact sentinel value is implementation detail). Compare valid rows.
        valid = torch.arange(T, device=dev) != 0
        torch.testing.assert_close(
            l16[valid], l8[valid], atol=5e-3, rtol=5e-3
        )

    def test_a_single_valid_split(self):
        """Edge case: only one valid split among 16, rest are +inf sentinels.

        The combined output must equal that one split's partial output
        (softmax weight collapses to 1.0 for the single valid split).
        """
        T, H, D = self.total_tokens, self.h_q, self.d_v
        dev = self.device

        po16 = torch.randn(16, T, H, D, dtype=torch.float32, device=dev).to(
            torch.bfloat16
        )
        lse16 = torch.full((16, T, H), INF_SENTINEL, dtype=torch.float32, device=dev)
        # Make split index 5 the only valid one.
        lse16[5] = torch.randn(T, H, dtype=torch.float32, device=dev)

        out, lse = _run_combine_any_sk(po16.clone(), lse16.clone(), 16, T, H, D)

        self.assertFalse(torch.isnan(out.float()).any())
        # With a single valid split, combined output == that split's partial.
        torch.testing.assert_close(
            out.float(), po16[5].float(), atol=5e-3, rtol=5e-3
        )


if __name__ == "__main__":
    unittest.main()
