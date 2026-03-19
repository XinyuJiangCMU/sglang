# SPDX-License-Identifier: Apache-2.0
"""Utility to tune missing AITER FP8 GEMM configs for AMD MI300X.

This script identifies and tunes missing kernel configurations in AITER's
a8w8_bpreshuffle_tuned_gemm.csv.  Shapes with K not divisible by 512
(e.g. K=9472 for Qwen2.5-7B down_proj, K=14784/29568 for Qwen2.5-72B at TP=1/2)
cannot use the default CK heuristic path and must have explicit tuned configs
or they run ~3x slower.

NOTE: K values that are not divisible by 64 (e.g. K=7392 for Qwen2.5-72B TP=4,
K=3696 for TP=8) cannot be tuned by any available CK bpreshuffle kernel and are
excluded from SHAPES_TO_TUNE.  Those shapes will always use the slower fallback.

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
# dispatcher picks a sub-optimal kernel.  A tuned entry must be present in
# a8w8_bpreshuffle_tuned_gemm.csv.
#
# Shapes NOT listed here (cannot be tuned -- no compatible CK kernel):
#   K=7392  (Qwen2.5-72B TP=4):  7392 % 64 != 0
#   K=3696  (Qwen2.5-72B TP=8):  3696 % 32 != 0
SHAPES_TO_TUNE: List[Tuple[int, int, str]] = [
    # Qwen2.5-7B/14B: intermediate_size=18944 -> K=9472 at TP=2
    (3584, 9472, "Qwen2.5-7B/14B down_proj TP=2"),
    (9472, 3584, "Qwen2.5-7B/14B gate_up TP=2"),
    # Llama3-8B: intermediate_size=14336 -- K divisible by 512, included for
    # benchmarking since shape is large and default kernel may not be optimal.
    (28672, 4096, "Llama3-8B fused gate_up TP=1"),
    (4096, 28672, "Llama3-8B down_proj TP=1"),
    # Qwen2.5-72B: intermediate_size=29568, hidden_size=8192.
    # down_proj K=29568/tp.  TP=4 (K=7392) and TP=8 (K=3696) have no valid
    # CK bpreshuffle kernel (K % 64 != 0) so only TP=1 and TP=2 are tunable.
    (8192, 29568, "Qwen2.5-72B down_proj TP=1"),
    (8192, 14784, "Qwen2.5-72B down_proj TP=2"),
    # --- Expanded coverage added 2026-03-19 ---
    # Qwen2.5-7B TP=4: down_proj K=4736 (18944/4)
    (3584, 4736, "Qwen2.5-7B down_proj TP=4"),
    # Llama3-8B TP=4: down_proj K=3584 (14336/4)
    (4096, 3584, "Llama3-8B down_proj TP=4"),
    # Qwen2.5-14B: hidden=5120, intermediate=13696
    (5120, 13696, "Qwen2.5-14B down_proj TP=1"),
    (5120, 6848, "Qwen2.5-14B down_proj TP=2"),
    (27392, 5120, "Qwen2.5-14B gate_up TP=1"),
    (13696, 5120, "Qwen2.5-14B gate_up TP=2"),
    # Qwen2.5-32B: hidden=5120, intermediate=27648
    # TP=2 down_proj K=13824 is already present; TP=1 gate_up K=5120 may be missing
    (55296, 5120, "Qwen2.5-32B gate_up TP=1"),
    (5120, 27648, "Qwen2.5-32B down_proj TP=1"),
    # Qwen2.5-3B: hidden=2048, intermediate=11008
    # Both K=11008 (down_proj) and K=2048 (gate_up) are divisible by 512 and tunable.
    (2048, 11008, "Qwen2.5-3B down_proj TP=1"),
    (22016, 2048, "Qwen2.5-3B gate_up TP=1"),
    # DeepSeek-V2-Lite dense: hidden=2048, intermediate=11264
    (2048, 11264, "DeepSeek-V2-Lite down_proj TP=1"),
    # Gemma3-4B: hidden=2560, intermediate=10240
    (2560, 10240, "Gemma3-4B down_proj TP=1"),
    (2560, 5120, "Gemma3-4B down_proj TP=2"),
    # Gemma3-12B: hidden=3840, intermediate=15360
    (3840, 15360, "Gemma3-12B down_proj TP=1"),
    (3840, 7680, "Gemma3-12B down_proj TP=2"),
    # --- Qwen2.5-72B N>=K shapes (stay on bpreshuffle with hybrid dispatch) ---
    # QKV: num_heads=64, kv_heads=8, head_dim=128 -> N=(64+2*8)*128=10240
    (10240, 8192, "Qwen2.5-72B QKV proj TP=1"),
    # GateUp: N=intermediate*2/tp = 29568*2/1 = 59136
    (59136, 8192, "Qwen2.5-72B gate_up TP=1"),
    # O proj: N=hidden=8192, K=hidden=8192
    (8192, 8192, "Qwen2.5-72B/Llama3-70B o_proj TP=1"),
    # Llama3-8B/70B: QKV and GateUp shapes
    (6144, 4096, "Llama3-8B QKV proj TP=1"),
    (8192, 4096, "Llama3-70B QKV proj TP=1"),
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

    # Use a path that does not yet exist so the tuner creates it fresh
    # (some tuner versions fail when given a pre-existing empty file).
    tuned_csv = tempfile.mktemp(suffix="_tuned.csv")

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

    # The tuner exits 1 if any shape fails, even when others succeed and were
    # written to the output file.  Merge whatever was successfully tuned.
    if not os.path.exists(tuned_csv):
        logger.error("Tune script produced no output file.")
        sys.exit(1)

    import pandas as pd

    tuned_df = pd.read_csv(tuned_csv) if os.path.getsize(tuned_csv) > 0 else None
    if tuned_df is None or len(tuned_df) == 0:
        logger.error("Tune script produced an empty output file; nothing to merge.")
        sys.exit(1)

    if ret != 0:
        logger.warning(
            f"Tune script exited with code {ret}; some shapes may not have been tuned. "
            f"Merging {len(tuned_df)} successfully tuned entries."
        )

    merge_tuned_results(tuned_csv)
    os.unlink(untuned_csv)
    if os.path.exists(tuned_csv):
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
