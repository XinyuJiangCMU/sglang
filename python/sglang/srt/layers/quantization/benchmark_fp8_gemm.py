#!/usr/bin/env python3
"""Benchmark FP8 GEMM dispatch paths on AMD MI300X.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 -m sglang.srt.layers.quantization.benchmark_fp8_gemm

Tests the hybrid bpreshuffle / torch._scaled_mm dispatch for common model shapes
and reports per-layer and per-model decode latency.
"""

import os
import time

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")


def bench(fn, iters=500, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    from aiter import gemm_a8w8_bpreshuffle, per_token_quant_hip, silu_and_mul
    from aiter import rmsnorm2d_fwd_with_add_dynamicquant
    from aiter.ops.shuffle import shuffle_weight
    import aiter

    fp8 = torch.float8_e4m3fnuz

    models = [
        {
            "name": "Qwen2.5-72B",
            "hidden": 8192,
            "intermediate": 29568,
            "qkv_out": 10240,
            "n_layers": 80,
        },
        {
            "name": "Llama3-70B",
            "hidden": 8192,
            "intermediate": 14336,
            "qkv_out": 10240,
            "n_layers": 80,
        },
        {
            "name": "Llama3-8B",
            "hidden": 4096,
            "intermediate": 14336,
            "qkv_out": 6144,
            "n_layers": 32,
        },
        {
            "name": "Qwen2.5-7B",
            "hidden": 3584,
            "intermediate": 9472,
            "qkv_out": 4608,
            "n_layers": 28,
        },
    ]

    for M in [1, 32, 128]:
        print(f"\n{'='*70}")
        print(f"Batch size = {M}")
        print(f"{'='*70}")
        header = f"{'Model':<16} {'QKV':>8} {'O':>8} {'GateUp':>8} {'Down':>8} {'Norm':>8} {'SiLU':>8} {'Quant':>8} {'Total':>10}"
        print(header)
        print("-" * len(header))

        for model in models:
            hidden = model["hidden"]
            intermediate = model["intermediate"]
            qkv_out = model["qkv_out"]
            n_layers = model["n_layers"]

            # Prepare tensors
            x_fp8 = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16).to(fp8)
            xs = torch.ones(M, 1, device="cuda", dtype=torch.float32)

            # Weights (decide dispatch per shape)
            shapes = {
                "qkv": (qkv_out, hidden),
                "o": (hidden, hidden),
                "gate_up": (intermediate * 2, hidden),
                "down": (hidden, intermediate),
            }

            times = {}
            for name, (N, K) in shapes.items():
                w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(fp8)
                use_smm = K > N or N * K > 200_000_000

                if use_smm:
                    w_t = w.t()
                    ws_r = torch.ones(1, N, device="cuda", dtype=torch.float32)
                    if name == "down":
                        x_in = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(fp8)
                        xs_in = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                    else:
                        x_in, xs_in = x_fp8, xs
                    times[name] = bench(
                        lambda x=x_in, wt=w_t, sa=xs_in, sb=ws_r: torch._scaled_mm(
                            x, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
                        )
                    )
                else:
                    w_s = shuffle_weight(w, (16, 16))
                    ws = torch.ones(N, device="cuda", dtype=torch.float32)
                    if name == "down":
                        x_in = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(fp8)
                        xs_in = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                    else:
                        x_in, xs_in = x_fp8, xs
                    times[name] = bench(
                        lambda x=x_in, w=w_s, sa=xs_in, sb=ws: gemm_a8w8_bpreshuffle(
                            x, w, sa, sb, None, torch.bfloat16
                        )
                    )

            # Non-GEMM
            h = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16)
            res = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16)
            wn = torch.randn(hidden, device="cuda", dtype=torch.float32)
            fo = torch.empty(M, hidden, device="cuda", dtype=fp8)
            fs = torch.empty(M, 1, device="cuda", dtype=torch.float32)
            ro = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16)
            norm_us = bench(
                lambda: rmsnorm2d_fwd_with_add_dynamicquant(fo, h, res, ro, fs, wn, 1e-6)
            )
            gu = torch.randn(M, intermediate * 2, device="cuda", dtype=torch.bfloat16)
            so = torch.empty(M, intermediate, device="cuda", dtype=torch.bfloat16)
            silu_us = bench(lambda: silu_and_mul(so, gu))
            di = torch.randn(M, intermediate, device="cuda", dtype=torch.bfloat16)
            quant_us = bench(
                lambda: per_token_quant_hip(di, quant_dtype=aiter.dtypes.fp8)
            )

            per_layer = (
                times["qkv"]
                + times["o"]
                + times["gate_up"]
                + times["down"]
                + 2 * norm_us
                + silu_us
                + quant_us
                + 15  # attention estimate
            )
            total = per_layer * n_layers / 1000  # ms

            print(
                f"{model['name']:<16} "
                f"{times['qkv']:>7.1f}{'*' if (hidden > qkv_out) else ' '}"
                f"{times['o']:>7.1f}{'*' if (hidden > hidden) else ' '}"
                f"{times['gate_up']:>7.1f}{'*' if (hidden > intermediate*2) or (intermediate*2*hidden > 200e6) else ' '}"
                f"{times['down']:>7.1f}{'*' if (intermediate > hidden) else ' '}"
                f"{norm_us:>8.1f}"
                f"{silu_us:>8.1f}"
                f"{quant_us:>8.1f}"
                f"{total:>9.1f}ms"
            )

        print("\n* = uses torch._scaled_mm (hipBLASLt)")


if __name__ == "__main__":
    main()
