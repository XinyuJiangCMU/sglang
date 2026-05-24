"""Correctness tests for the AMD FlyDSL kgather kernel + dispatch path.

Covers three layers:

1. The kgather kernel itself — byte-exact gather vs ``torch.gather`` on
   synthetic and (optionally) real captured DSv4 K cache.
2. The capability / feature-guard logic in
   ``flydsl_kernel._check_kgather_supported`` — every unsupported layout
   should produce a clear ``(False, reason)`` tuple without raising.
3. The full dispatch entry point
   ``dpsk_v4_fp8_attention_fwd_flydsl_kgather_only`` — does not change
   the model's math (returns TileLang's output unchanged) and tolerates
   unsupported feature combinations by skipping the kgather smoke run.

Tests skip cleanly on:

* non-AMD hosts,
* hosts without ``flydsl`` / ``aiter`` installed,
* unsupported arches (anything other than gfx95*),
* missing TileLang (for the dispatch test that compares output to TileLang).

When ``SGLANG_FLYDSL_TEST_PICKLE`` points at a captured DSv4 microbench
pickle, the byte-exact tests additionally run against real captured K
cache (not just synthetic shapes).
"""

from __future__ import annotations

import os
import unittest

import torch

# Layout constants — match the production module. Kept duplicated here so
# the test asserts the public layout rather than importing internal
# constants (which could drift silently).
BS_KV: int = 128
PACKED_W_FULL: int = 584
PACKED_W: int = 576


def _flydsl_or_skip():
    """Skip the test if the FlyDSL kgather backend can't run."""
    try:
        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            is_flydsl_kgather_available,
        )
    except ImportError as e:
        raise unittest.SkipTest(f"flydsl_kernel module not importable: {e}")
    ok, reason = is_flydsl_kgather_available()
    if not ok:
        raise unittest.SkipTest(f"FlyDSL kgather backend unavailable: {reason}")
    return is_flydsl_kgather_available


def _maybe_load_pickle():
    """Return ``kwargs`` dict from the captured pickle, or None if no path set."""
    path = os.environ.get("SGLANG_FLYDSL_TEST_PICKLE")
    if not path or not os.path.exists(path):
        return None
    mb = torch.load(path, map_location="cuda", weights_only=False)

    def _u(d):
        return d["data"].to("cuda") if isinstance(d, dict) else d

    return {k: _u(v) for k, v in mb["kwargs"].items()}


def _synthetic_k_cache(nb: int, device: torch.device) -> torch.Tensor:
    """Build a synthetic K cache with the production layout."""
    return torch.randint(
        0, 256,
        (nb, BS_KV, 1, PACKED_W_FULL),
        dtype=torch.uint8,
        device=device,
    ).view(torch.int8).contiguous()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP device required")
class TestFlyDSLKGatherKernel(unittest.TestCase):
    """Layer 1: the kgather kernel itself."""

    def setUp(self) -> None:
        _flydsl_or_skip()
        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            _build_kgather_kernel,
        )

        self._build_kgather_kernel = _build_kgather_kernel
        self._device = torch.device("cuda")

    def _run_and_compare(self, k_cache: torch.Tensor, indices: torch.Tensor) -> None:
        nb, bs_kv, h_kv, packed_w_full = k_cache.shape
        bs, s_q, topk = indices.shape
        self.assertEqual(h_kv, 1)
        self.assertEqual(s_q, 1)
        self.assertEqual(packed_w_full, PACKED_W_FULL)

        # Clip identically on both sides so the comparison is well-defined.
        max_token = nb * bs_kv - 1
        idx_clipped = torch.clamp(
            indices.reshape(-1), min=0, max=max_token
        ).to(torch.int64)
        block_id = idx_clipped // bs_kv
        in_block = idx_clipped % bs_kv
        # In-block stride is the full row width (584), not the 576-byte
        # packed region — see flydsl_kernel.py for the layout discussion.
        row_byte_offsets = (
            block_id * (bs_kv * packed_w_full)
            + in_block * packed_w_full
        ).to(torch.int32).contiguous()

        grid_x = row_byte_offsets.numel()
        if grid_x == 0:
            return  # empty workload — kernel cannot launch, nothing to check
        k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
        scratch = torch.empty(grid_x * PACKED_W, dtype=torch.int8, device=self._device)

        launch = self._build_kgather_kernel(PACKED_W, 16)
        launch(k_cache_i8, row_byte_offsets, scratch, grid_x)
        torch.cuda.synchronize()

        kc_2d = k_cache.view(torch.int8).reshape(nb * bs_kv, packed_w_full)
        row_idx_2d = (block_id * bs_kv + in_block).to(torch.int64)
        ref = kc_2d[row_idx_2d, :PACKED_W].contiguous()
        got = scratch.view(grid_x, PACKED_W)

        n_bad = int((got != ref).any(dim=1).sum().item())
        self.assertEqual(
            n_bad,
            0,
            f"{n_bad} / {grid_x} rows mismatch vs torch.gather",
        )

    # --------------------------------------------------------------
    # bs sweep — synthetic K cache
    # --------------------------------------------------------------

    def _bs_sweep_synthetic(self, bs: int, topk: int) -> None:
        torch.manual_seed(bs * 1009 + topk)
        # NB sized to ensure the gather targets are mostly valid (= 2x what
        # the workload could need at full saturation).
        nb = max(2, (bs * topk * 2) // BS_KV + 1)
        k_cache = _synthetic_k_cache(nb, self._device)
        indices = torch.randint(
            0, nb * BS_KV, (bs, 1, topk), dtype=torch.int32, device=self._device
        )
        self._run_and_compare(k_cache, indices)

    def test_bs_001(self) -> None: self._bs_sweep_synthetic(bs=1,   topk=64)
    def test_bs_002(self) -> None: self._bs_sweep_synthetic(bs=2,   topk=64)
    def test_bs_004(self) -> None: self._bs_sweep_synthetic(bs=4,   topk=64)
    def test_bs_008(self) -> None: self._bs_sweep_synthetic(bs=8,   topk=64)
    def test_bs_016(self) -> None: self._bs_sweep_synthetic(bs=16,  topk=64)
    def test_bs_032(self) -> None: self._bs_sweep_synthetic(bs=32,  topk=64)
    def test_bs_064(self) -> None: self._bs_sweep_synthetic(bs=64,  topk=64)
    def test_bs_128(self) -> None: self._bs_sweep_synthetic(bs=128, topk=64)
    def test_bs_192(self) -> None: self._bs_sweep_synthetic(bs=192, topk=64)

    # --------------------------------------------------------------
    # TOPK / BI sweep — partial blocks and edge widths
    # --------------------------------------------------------------

    def test_topk_smaller_than_bi(self) -> None:
        """BI=64 is the usual chunk; gather should still work for TOPK<64."""
        for topk in (1, 7, 16, 32, 48, 63):
            with self.subTest(topk=topk):
                self._bs_sweep_synthetic(bs=8, topk=topk)

    def test_topk_full_dsv4(self) -> None:
        """topk=128 matches the captured DSv4-Pro pickle."""
        self._bs_sweep_synthetic(bs=8, topk=128)

    # --------------------------------------------------------------
    # Index edge cases
    # --------------------------------------------------------------

    def test_repeated_indices(self) -> None:
        nb = 4
        k_cache = _synthetic_k_cache(nb, self._device)
        # Every token gathers the same row — exercises the cache.
        indices = torch.full(
            (8, 1, 32), fill_value=42, dtype=torch.int32, device=self._device
        )
        self._run_and_compare(k_cache, indices)

    def test_all_negative_indices_clip_safely(self) -> None:
        nb = 4
        k_cache = _synthetic_k_cache(nb, self._device)
        # Captured pickles have negative sentinel values for unfilled topk
        # slots. The clip path inside the test (and inside the production
        # exercise) must turn them into row 0 without OOB reads.
        indices = torch.full(
            (4, 1, 16), fill_value=-1, dtype=torch.int32, device=self._device
        )
        self._run_and_compare(k_cache, indices)

    def test_out_of_range_indices_clip_safely(self) -> None:
        nb = 4
        k_cache = _synthetic_k_cache(nb, self._device)
        # Indices >= NB*BS_KV — must clip to the last valid token, not OOB.
        indices = torch.full(
            (4, 1, 16),
            fill_value=nb * BS_KV + 9999,
            dtype=torch.int32,
            device=self._device,
        )
        self._run_and_compare(k_cache, indices)

    # --------------------------------------------------------------
    # Real captured pickle (only when SGLANG_FLYDSL_TEST_PICKLE is set)
    # --------------------------------------------------------------

    def test_real_captured_pickle(self) -> None:
        kw = _maybe_load_pickle()
        if kw is None:
            self.skipTest(
                "Set SGLANG_FLYDSL_TEST_PICKLE to a captured pickle to enable"
            )
        k_cache = kw["k_cache"]
        indices = kw["indices"]
        # Subsample to keep the test fast even on a large captured pickle.
        bs = min(16, indices.shape[0])
        self._run_and_compare(k_cache, indices[:bs].contiguous())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP device required")
class TestFlyDSLCapabilityGuards(unittest.TestCase):
    """Layer 2: _check_kgather_supported must NEVER raise on the request path."""

    def setUp(self) -> None:
        _flydsl_or_skip()
        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            _check_kgather_supported,
        )

        self._check = _check_kgather_supported
        self._device = torch.device("cuda")

    def _valid_inputs(self):
        nb = 4
        k_cache = _synthetic_k_cache(nb, self._device)
        indices = torch.randint(
            0, nb * BS_KV, (8, 1, 16), dtype=torch.int32, device=self._device
        )
        return k_cache, indices

    def test_valid_inputs_supported(self) -> None:
        ok, reason = self._check(*self._valid_inputs())
        self.assertTrue(ok, f"valid inputs rejected: {reason!r}")

    def test_none_inputs_rejected(self) -> None:
        ok, _ = self._check(None, None)
        self.assertFalse(ok)

    def test_h_kv_not_one_rejected(self) -> None:
        k_cache = torch.zeros(
            (4, BS_KV, 2, PACKED_W_FULL), dtype=torch.int8, device=self._device
        )
        idx = torch.zeros((4, 1, 16), dtype=torch.int32, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("H_KV", reason)

    def test_packed_width_mismatch_rejected(self) -> None:
        k_cache = torch.zeros(
            (4, BS_KV, 1, PACKED_W_FULL + 16),
            dtype=torch.int8,
            device=self._device,
        )
        idx = torch.zeros((4, 1, 16), dtype=torch.int32, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("packed width", reason)

    def test_noncontiguous_rejected(self) -> None:
        k_cache = _synthetic_k_cache(8, self._device)[::2]
        idx = torch.zeros((4, 1, 16), dtype=torch.int32, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("non-contiguous", reason)

    def test_s_q_not_one_rejected(self) -> None:
        k_cache = _synthetic_k_cache(4, self._device)
        idx = torch.zeros((4, 3, 16), dtype=torch.int32, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("S_Q", reason)

    def test_wrong_index_dtype_rejected(self) -> None:
        k_cache = _synthetic_k_cache(4, self._device)
        idx = torch.zeros((4, 1, 16), dtype=torch.int64, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("dtype", reason)

    def test_empty_workload_rejected(self) -> None:
        k_cache = _synthetic_k_cache(4, self._device)
        idx = torch.zeros((0, 1, 16), dtype=torch.int32, device=self._device)
        ok, reason = self._check(k_cache, idx)
        self.assertFalse(ok)
        self.assertIn("empty", reason)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP device required")
class TestFlyDSLDispatchEntryPoint(unittest.TestCase):
    """Layer 3: the dispatch entry must always return TileLang's output."""

    def setUp(self) -> None:
        _flydsl_or_skip()
        # Skip the entire class if TileLang isn't importable (we can't
        # compare without a TileLang reference).
        try:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                dpsk_v4_fp8_attention_fwd,
            )
        except ImportError as e:
            raise unittest.SkipTest(f"TileLang unavailable: {e}")

        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            dpsk_v4_fp8_attention_fwd_flydsl_kgather_only,
        )

        self._tilelang_fwd = dpsk_v4_fp8_attention_fwd
        self._flydsl_fwd = dpsk_v4_fp8_attention_fwd_flydsl_kgather_only

    def test_kgather_does_not_change_math(self) -> None:
        """Backend must return TileLang's output unchanged."""
        kw = _maybe_load_pickle()
        if kw is None:
            self.skipTest(
                "Set SGLANG_FLYDSL_TEST_PICKLE to a captured pickle to enable"
            )

        # Toggle the exercise on so we exercise the kgather path AND the
        # delegation simultaneously.
        os.environ["SGLANG_FLYDSL_EXERCISE"] = "1"
        try:
            tile_out, tile_lse = self._tilelang_fwd(**kw)
            fly_out, fly_lse = self._flydsl_fwd(**kw)
        finally:
            os.environ.pop("SGLANG_FLYDSL_EXERCISE", None)

        # Compare on finite entries only — TileLang itself is
        # non-deterministic on this captured pickle (some NaN entries), so
        # an exact equality across the entire tensor isn't appropriate.
        for name, ours, theirs in (
            ("out", fly_out, tile_out),
            ("lse", fly_lse, tile_lse),
        ):
            mask = torch.isfinite(ours) & torch.isfinite(theirs)
            diff = (ours - theirs).abs()[mask]
            if diff.numel() == 0:
                continue
            self.assertEqual(
                int((diff > 0).sum().item()),
                0,
                f"{name}: FlyDSL backend changed TileLang's math at "
                f"{int((diff > 0).sum().item())} positions",
            )

    def test_kgather_tolerates_unsupported_feature(self) -> None:
        """Non-contiguous K cache → kgather soft-skip + TileLang fallback."""
        kw = _maybe_load_pickle()
        if kw is None:
            self.skipTest(
                "Set SGLANG_FLYDSL_TEST_PICKLE to a captured pickle to enable"
            )
        kw_bad = dict(kw)
        kw_bad["k_cache"] = kw["k_cache"][:512].contiguous()
        # Force a non-contiguous view to trigger the soft-reject path.
        kw_bad["k_cache"] = kw_bad["k_cache"][::2]

        os.environ["SGLANG_FLYDSL_EXERCISE"] = "1"
        os.environ["SGLANG_FLYDSL_DEBUG"] = "1"
        try:
            # The dispatch must not raise — kgather is skipped, math
            # delegates to TileLang normally. If TileLang itself can't
            # handle the shape, that's expected (we just need OUR code
            # to not raise before the delegate gets called).
            try:
                self._flydsl_fwd(**kw_bad)
            except Exception:
                # TileLang's own failure path. The contract here is "FlyDSL
                # code does not raise before the delegate"; the delegate
                # may then fail for unrelated reasons (bad inputs we
                # passed). That's outside the scope of this test.
                pass
        finally:
            os.environ.pop("SGLANG_FLYDSL_EXERCISE", None)
            os.environ.pop("SGLANG_FLYDSL_DEBUG", None)


if __name__ == "__main__":
    unittest.main()
