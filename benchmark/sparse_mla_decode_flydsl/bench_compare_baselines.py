"""Side-by-side timing of TileLang and Triton sparse-FP8 MLA decode.

This script runs the existing production backends on a captured DSv4
microbench pickle so the numbers can be compared apples-to-apples
against FlyDSL standalone benchmarks. It does **not** call any FlyDSL
code itself.

Usage:
    SGLANG_FLYDSL_TEST_PICKLE=/path/to/microbench_bs192.pkl \\
        python3 benchmark/sparse_mla_decode_flydsl/bench_compare_baselines.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any, Callable, Dict

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pickle", type=str,
        default=os.environ.get("SGLANG_FLYDSL_TEST_PICKLE"),
        help="path to captured microbench pickle "
             "(env: SGLANG_FLYDSL_TEST_PICKLE)",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--samples", type=int, default=5)
    return p.parse_args()


def _unwrap(d: Any) -> Any:
    if isinstance(d, dict) and "data" in d:
        return d["data"].to("cuda")
    return d


def _load_kwargs(pickle_path: str) -> Dict[str, Any]:
    mb = torch.load(pickle_path, map_location="cuda", weights_only=False)
    return {k: _unwrap(v) for k, v in mb["kwargs"].items()}


def _bench(
    name: str,
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    warmup: int,
    iters: int,
    samples: int,
) -> None:
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    sample_us = []
    for _ in range(samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(**kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sample_us.append((t1 - t0) / iters * 1e6)

    bs = kwargs["q"].shape[0]
    sample_us.sort()
    median = statistics.median(sample_us)
    p90 = sample_us[int(0.9 * (len(sample_us) - 1))]
    print(
        f"{name:>16}: median={median:7.2f} µs/call  "
        f"({median / bs:6.3f} µs/batch)   p90={p90:7.2f} µs"
    )


def main() -> None:
    args = _parse_args()
    if not args.pickle or not os.path.exists(args.pickle):
        raise SystemExit(
            "Need a captured pickle: pass --pickle or set "
            "SGLANG_FLYDSL_TEST_PICKLE."
        )

    print(f"loading {args.pickle}")
    kwargs = _load_kwargs(args.pickle)
    bs = kwargs["q"].shape[0]
    print(
        f"shapes: q={tuple(kwargs['q'].shape)}, "
        f"k_cache={tuple(kwargs['k_cache'].shape)}, "
        f"indices={tuple(kwargs['indices'].shape)}, BS={bs}"
    )
    print(
        f"timing config: warmup={args.warmup} iters, "
        f"samples={args.samples} × {args.iters} iters\n"
    )

    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        dpsk_v4_fp8_attention_fwd,
    )
    from sglang.srt.layers.attention.nsa.triton_decode import (
        triton_fp8_attention_fwd,
    )

    _bench("tilelang", dpsk_v4_fp8_attention_fwd, kwargs,
           args.warmup, args.iters, args.samples)
    _bench("triton", triton_fp8_attention_fwd, kwargs,
           args.warmup, args.iters, args.samples)


if __name__ == "__main__":
    main()
