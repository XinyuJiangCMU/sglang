# FlyDSL K-gather benchmark (AMD gfx950)

Standalone microbenchmark for the FlyDSL weapon-1 K-gather kernel used by
the production-integrated `flydsl_kgather_only` backend.

This directory contains a **single** benchmark script. It measures the
kgather kernel only — the same code path that runs inside the production
backend when `SGLANG_FLYDSL_EXERCISE=1`.

| Script | Measures |
|---|---|
| `bench_kgather.py` | Median + p90 per-call latency of the FlyDSL weapon-1 K-gather kernel; effective HBM bandwidth. |

There are **no end-to-end FlyDSL attention numbers in this PR**. An
earlier draft contained a prototype standalone full sub-kernel
benchmark, but its V dequant path was structurally wrong for DSv4 (it
read from a synthetic separate V cache with a hardcoded scale byte,
whereas real DSv4 uses `V == K` with the same per-NOPE_TILE scale —
see `tilelang_kernel.py:1849-1860`). That prototype was removed
because its µs/batch number is not meaningful as a production comparison.

## Hardware

- AMD MI355X / **gfx950**
- ROCm 6.x
- `flydsl` + `aiter` Python packages installed

## Run

```bash
python3 benchmark/sparse_mla_decode_flydsl/bench_kgather.py
```

CLI flags (all optional, see `--help`):

- `--nb`, `--bs-kv`, `--packed-w-full`, `--packed-w` — K cache shape
  (defaults match captured DSv4-Pro `bs=192` decode workload)
- `--bs`, `--topk` — workload size
- `--warmup` (default 20), `--iters` (default 200), `--samples`
  (default 10) — timing methodology

## What "perf" means here

The kgather kernel does **not** change attention math. It only gathers
sparse K-cache rows into a scratch buffer and is run in production by
the `flydsl_kgather_only` backend as a smoke test that the FlyDSL
toolchain functions under real server load (CUDA graph capture/replay,
concurrent decode requests, real HBM pressure).

A "fast" kgather kernel does not imply a faster end-to-end decode
kernel — the kgather output is discarded and the actual attention math
is delegated to TileLang. See
[`docs/developer_guide/amd_flydsl_sparse_mla.md`](
../../docs/developer_guide/amd_flydsl_sparse_mla.md) for the full
production-vs-prototype boundary and roadmap.
