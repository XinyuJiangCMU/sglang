"""Benchmark: FlyDSL weapon-1 K-gather kernel (AMD gfx950).

Measures the same kernel that the production ``flydsl_kgather_only``
backend uses. Reports median per-call latency and effective HBM
bandwidth, excluding one-time compile time. No request-path-style
synchronization happens between iterations; we sync once at the start
and once at the end of the timed loop.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nb", type=int, default=2897,
        help="number of K-cache blocks (default matches captured bs=192 pickle)",
    )
    p.add_argument(
        "--bs-kv", type=int, default=128,
        help="K cache block size in tokens",
    )
    p.add_argument(
        "--packed-w-full", type=int, default=584,
        help="packed bytes per K-cache row (packed FP8 + scale region)",
    )
    p.add_argument(
        "--packed-w", type=int, default=576,
        help="bytes the kgather kernel reads per row (skip scale region here)",
    )
    p.add_argument(
        "--bs", type=int, default=159,
        help="decode batch size (number of (batch, k) gather rows per call)",
    )
    p.add_argument(
        "--topk", type=int, default=128,
        help="number of K rows per batch to gather",
    )
    p.add_argument(
        "--warmup", type=int, default=20,
    )
    p.add_argument(
        "--iters", type=int, default=200,
    )
    p.add_argument(
        "--samples", type=int, default=10,
        help="number of timing samples (each averages `iters` calls)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from sglang.srt.layers.attention.nsa.flydsl_kernel import (
        _build_kgather_kernel,
        is_flydsl_kgather_available,
    )

    ok, reason = is_flydsl_kgather_available()
    if not ok:
        raise SystemExit(f"FlyDSL kgather backend unavailable: {reason}")

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Allocate a representative K cache + indices on device.
    k_cache_i8 = torch.randint(
        0, 256,
        (args.nb * args.bs_kv * args.packed_w_full,),
        dtype=torch.uint8,
        device=device,
    ).view(torch.int8).contiguous()

    indices = torch.randint(
        0,
        args.nb * args.bs_kv,
        (args.bs * args.topk,),
        dtype=torch.int32,
        device=device,
    )
    block_id = (indices // args.bs_kv).to(torch.int64)
    in_block = (indices % args.bs_kv).to(torch.int64)
    # Correct in-block stride is packed_w_full bytes per token.
    row_byte_offsets = (
        block_id * (args.bs_kv * args.packed_w_full)
        + in_block * args.packed_w_full
    ).to(torch.int32).contiguous()
    grid_x = row_byte_offsets.numel()
    scratch = torch.empty(grid_x * args.packed_w, dtype=torch.int8, device=device)

    # Compile + warmup (compile cost excluded from timing).
    launch = _build_kgather_kernel(args.packed_w, 16)
    for _ in range(args.warmup):
        launch(k_cache_i8, row_byte_offsets, scratch, grid_x)
    torch.cuda.synchronize()

    # Timed loop: each sample is `iters` calls back-to-back.
    samples_us = []
    for _ in range(args.samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            launch(k_cache_i8, row_byte_offsets, scratch, grid_x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_us.append((t1 - t0) / args.iters * 1e6)

    samples_us.sort()
    median_us = statistics.median(samples_us)
    p90_us = samples_us[int(0.9 * (len(samples_us) - 1))]

    bytes_loaded = grid_x * args.packed_w
    eff_bw_gbs = bytes_loaded / (median_us * 1e-6) / 1e9

    print(f"K cache: NB={args.nb}, BS_KV={args.bs_kv}, packed_w_full={args.packed_w_full}")
    print(f"workload: BS={args.bs}, TOPK={args.topk}, grid_x={grid_x}, "
          f"per-call bytes={bytes_loaded:,}")
    print(f"latency: median={median_us:.3f} µs, p90={p90_us:.3f} µs "
          f"(over {args.samples} samples × {args.iters} iters)")
    print(f"effective HBM BW: {eff_bw_gbs:.1f} GB/s")


if __name__ == "__main__":
    main()
