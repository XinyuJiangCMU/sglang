# SPDX-License-Identifier: Apache-2.0
"""Utility to tune missing AITER FP8 GEMM configs for AMD MI300X.

This script identifies and tunes missing kernel configurations in AITER's
a8w8_bpreshuffle_tuned_gemm.csv.  Shapes with K not divisible by 512
(e.g. K=9472 for Qwen2.5-7B down_proj, K=7392/3696 for Qwen2.5-72B at TP=4/8)
cannot use the default CK heuristic path and must have explicit tuned configs
or they run ~3x slower.

Usage:
    python -m sglang.srt.layers.quantization.aiter_gemm_tune [--dry-run] [--mp N]

After tuning, set AITER_REBUILD=1 and restart SGLang to rebuild the kernel .so
with the new lookup table entries.

Example:
    python -m sglang.srt.layers.quantization.aiter_gemm_tune --mp 8
    AITER_REBUILD=1 python -m sglang.launch_server ...
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Shapes known to be slow without explicit AITER tuning.
# Format: (N, K, description)
# These shapes have K that is not divisible by 512, so the CK heuristic
# dispatcher picks a sub-optimal kernel.  A tuned entry with KPerThread=256
# (or other K%256==0 kernel) must be present in a8w8_bpreshuffle_tuned_gemm.csv.
SHAPES_TO_TUNE: List[Tuple[int, int, str]] = [
    # Qwen2.5-7B/14B: intermediate_size=18944 -> K=9472 at TP=2
    (3584, 9472, "Qwen2.5-7B/14B down_proj TP=2"),
    (9472, 3584, "Qwen2.5-7B/14B gate_up TP=2"),
    # Llama3-8B: intermediate_size=14336 -- K divisible by 512, included for
    # benchmarking since shape is large and default kernel may not be optimal.
    (28672, 4096, "Llama3-8B fused gate_up TP=1"),
    (4096, 28672, "Llama3-8B down_proj TP=1"),
    # Qwen2.5-72B: intermediate_size=29568, hidden_size=8192.
    # down_proj K=29568/tp is never divisible by 512 at any TP level.
    (8192, 29568, "Qwen2.5-72B down_proj TP=1"),
    (8192, 14784, "Qwen2.5-72B down_proj TP=2"),
    (8192, 7392, "Qwen2.5-72B down_proj TP=4"),
    (8192, 3696, "Qwen2.5-72B down_proj TP=8"),
]

# Batch sizes (M) to tune for each shape.
# Covers: decode (M=1..32), chunked-prefill (M=64..128), prefill (M=512..1024).
TUNE_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024]


def find_missing_shapes() -> List[Tuple[int, int, str]]:
    """Return shapes from SHAPES_TO_TUNE that are missing from AITER config."""
    import pandas as pd
    from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE

    df = pd.read_csv(AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE)
    missing = []
    for N, K, desc in SHAPES_TO_TUNE:
        if len(df[(df["N"] == N) & (df["K"] == K)]) == 0:
            missing.append((N, K, desc))
    return missing


def build_untuned_csv(
    shapes: List[Tuple[int, int, str]], m_values: List[int]
) -> str:
    """Write an untuned CSV with the given shapes and return its path."""
    import pandas as pd

    rows = []
    for N, K, _ in shapes:
        for M in m_values:
            rows.append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "q_dtype_w": "torch.float8_e4m3fnuz",
                }
            )
    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(
        suffix="_untuned.csv", delete=False, mode="w"
    )
    df.to_csv(tmp.name, index=False)
    return tmp.name


def merge_tuned_results(tuned_csv: str) -> None:
    """Merge newly tuned results into AITER's main tuned GEMM config."""
    import pandas as pd
    from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE

    main = pd.read_csv(AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE)
    new_shapes = pd.read_csv(tuned_csv)

    # Check for existing entries that would be overwritten (shouldn't happen
    # for truly missing shapes, but be safe).
    merged = pd.concat([main, new_shapes], ignore_index=True)
    merged = merged.sort_values(["cu_num", "M", "N", "K"]).reset_index(drop=True)
    merged.to_csv(AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE, index=False)
    logger.info(
        f"Merged {len(new_shapes)} new entries into {AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE}"
    )


def tune(mp: int = 1, dry_run: bool = False) -> None:
    """Main entry point: find missing shapes, tune them, merge results."""
    try:
        from aiter.jit.core import AITER_CSRC_DIR
    except ImportError:
        logger.error("AITER is not installed.  Install aiter to use this utility.")
        sys.exit(1)

    missing = find_missing_shapes()
    if not missing:
        logger.info("All expected AITER FP8 GEMM shapes are already tuned.")
        return

    logger.info("Missing AITER FP8 GEMM configs:")
    for N, K, desc in missing:
        logger.info(f"  N={N} K={K}  ({desc})")

    if dry_run:
        logger.info("Dry-run mode: not tuning.  Run without --dry-run to tune.")
        return

    untuned_csv = build_untuned_csv(missing, TUNE_M_VALUES)
    logger.info(f"Wrote untuned shapes to {untuned_csv}")

    tune_script = os.path.join(
        AITER_CSRC_DIR,
        "ck_gemm_a8w8_bpreshuffle",
        "gemm_a8w8_bpreshuffle_tune.py",
    )
    if not os.path.exists(tune_script):
        logger.error(f"Tune script not found: {tune_script}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix="_tuned.csv", delete=False) as f:
        tuned_csv = f.name

    cmd = (
        f"cd {os.path.dirname(tune_script)} && "
        f"python3 {tune_script} "
        f"  -i {untuned_csv} "
        f"  -o {tuned_csv} "
        f"  --mp {mp} "
        f"  --iters 50 "
        f"  --warmup 10 "
        f"  --libtype ck,cktile "
        f"  --timeout 600"
    )
    logger.info(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        logger.error("Tune script failed.")
        sys.exit(1)

    merge_tuned_results(tuned_csv)
    os.unlink(untuned_csv)
    os.unlink(tuned_csv)

    logger.info(
        "Tuning complete.  Set AITER_REBUILD=1 and restart SGLang to rebuild "
        "the kernel .so with the new lookup table entries."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Tune missing AITER FP8 GEMM configs for AMD MI300X"
    )
    parser.add_argument(
        "--mp",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel tuning (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing shapes without running the tuner",
    )
    args = parser.parse_args()
    tune(mp=args.mp, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
