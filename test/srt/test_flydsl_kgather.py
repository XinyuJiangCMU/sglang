"""Correctness test for the AMD FlyDSL kgather kernel.

Validates the FlyDSL weapon-1 K-gather kernel byte-exactly against
``torch.gather`` on a real DeepSeek-V4 sparse FP8 K cache shape.

The test is gated on environment availability:

* Skips cleanly if FlyDSL / AITER are not installed.
* Skips cleanly if not running on AMD gfx95*.
* If ``SGLANG_FLYDSL_TEST_PICKLE`` points at a captured DSv4 microbench
  pickle (must contain ``kwargs.k_cache`` with shape
  ``(NB, BS_KV, 1, 584)`` FP8 and ``kwargs.indices`` with shape
  ``(BS, 1, TOPK)`` int32), the test uses the live captured tensors.
* Otherwise the test falls back to synthetic shapes that match the
  layout expected by the production kernel.

The kernel under test only gathers the 576-byte packed FP8 region per
token (not the 8-byte per-NOPE_TILE scale region — see
``benchmark/sparse_mla_decode_flydsl/bench_kgather.py`` for the full
two-region kernel).
"""

from __future__ import annotations

import os
import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA/HIP device required"
)
class TestFlyDSLKGather(unittest.TestCase):
    """End-to-end correctness for the FlyDSL kgather kernel."""

    BS_KV: int = 128
    PACKED_W_FULL: int = 584
    PACKED_W: int = 576

    def setUp(self) -> None:
        # Use the in-tree capability check so the skip reason matches what
        # the production dispatch would log.
        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            is_flydsl_kgather_available,
        )

        ok, reason = is_flydsl_kgather_available()
        if not ok:
            self.skipTest(f"FlyDSL kgather backend unavailable: {reason}")

        # Lazy import so the module-level import doesn't break non-AMD hosts.
        from sglang.srt.layers.attention.nsa.flydsl_kernel import (
            _build_kgather_kernel,
        )

        self._build_kgather_kernel = _build_kgather_kernel
        self._device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(k_cache, indices)`` from pickle or synthetic data."""
        pkl_path = os.environ.get("SGLANG_FLYDSL_TEST_PICKLE")
        if pkl_path and os.path.exists(pkl_path):
            mb = torch.load(pkl_path, map_location=self._device, weights_only=False)
            kw = mb["kwargs"]

            def _u(d):
                return d["data"].to(self._device) if isinstance(d, dict) else d

            k_cache = _u(kw["k_cache"])
            indices = _u(kw["indices"])
            # Use a subset of batches to keep the test fast.
            bs = min(8, indices.shape[0])
            indices = indices[:bs].contiguous()
            return k_cache, indices

        # Synthetic fallback that matches the production layout.
        nb = 32
        bs = 4
        topk = 16
        k_cache = torch.randint(
            0, 256,
            (nb, self.BS_KV, 1, self.PACKED_W_FULL),
            dtype=torch.uint8,
            device=self._device,
        ).view(torch.int8).contiguous()
        # Make sure indices reference valid tokens.
        max_token = nb * self.BS_KV
        indices = torch.randint(
            0, max_token, (bs, 1, topk), dtype=torch.int32, device=self._device
        ).contiguous()
        return k_cache, indices

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_kgather_byte_exact_vs_torch_gather(self) -> None:
        k_cache, indices = self._load_inputs()
        nb, bs_kv, h_kv, packed_w_full = k_cache.shape
        bs, s_q, topk = indices.shape
        self.assertEqual(h_kv, 1, "DSv4 expects H_KV=1 (MQA)")
        self.assertEqual(s_q, 1, "decode expects S_Q=1")
        self.assertEqual(packed_w_full, self.PACKED_W_FULL)

        # Clip indices to the valid token range. Captured pickles contain
        # sentinel values (negative or >= NB*BS_KV) for padding slots; we
        # clip both kernel input and torch reference identically so the
        # byte-exact comparison is well-defined.
        max_token = nb * bs_kv - 1
        idx_clipped = (
            torch.clamp(indices.reshape(-1), min=0, max=max_token).to(torch.int64)
        )
        block_id = idx_clipped // bs_kv
        in_block = idx_clipped % bs_kv
        # IMPORTANT: in-block stride is *packed_w_full* (584), not
        # PACKED_W (576). Each token's storage = packed FP8 (576) + scale
        # region (8). Using PACKED_W for the in-block stride skips the
        # scale region of the previous token and reads shifted data.
        row_byte_offsets = (
            block_id * (bs_kv * packed_w_full)
            + in_block * packed_w_full
        ).to(torch.int32).contiguous()

        grid_x = row_byte_offsets.numel()
        k_cache_i8 = k_cache.view(torch.int8).reshape(-1).contiguous()
        scratch = torch.empty(
            grid_x * self.PACKED_W,
            dtype=torch.int8,
            device=self._device,
        )

        launch = self._build_kgather_kernel(self.PACKED_W, 16)
        launch(k_cache_i8, row_byte_offsets, scratch, grid_x)
        torch.cuda.synchronize()

        # Reference: gather the same rows with torch advanced indexing.
        kc_2d_i8 = k_cache.view(torch.int8).reshape(nb * bs_kv, packed_w_full)
        row_idx_2d = (block_id * bs_kv + in_block).to(torch.int64)
        ref = kc_2d_i8[row_idx_2d, : self.PACKED_W].contiguous()
        got = scratch.view(grid_x, self.PACKED_W)

        diff_rows = (got != ref).any(dim=1)
        n_bad = int(diff_rows.sum().item())
        self.assertEqual(
            n_bad,
            0,
            f"{n_bad} / {grid_x} rows mismatch vs torch.gather",
        )


if __name__ == "__main__":
    unittest.main()
