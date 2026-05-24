# AMD FlyDSL kgather backend for DSv4 sparse FP8 MLA decode

**Status: opt-in diagnostic backend. Math behavior unchanged vs TileLang.**

This document describes a single, narrowly-scoped backend that exercises
a FlyDSL [weapon-1](https://gpuopen.com/learn/) (HBM → LDS direct)
K-cache gather kernel under real server load and then delegates all
attention math to the production TileLang backend
(`dpsk_v4_fp8_attention_fwd`). The model's numerical output is identical
to running with `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`.

There is **no FlyDSL attention math** in this PR. An earlier draft
contained a standalone prototype FlyDSL FP8 sparse attention kernel
together with a "0.4 µs/batch" microbenchmark number. That prototype was
removed from the PR because its V dequant path was structurally wrong
for DSv4 (it loaded from a synthetic separate V cache with a hardcoded
scale byte, while real DSv4 reuses the same dequant'd K for V with the
same per-NOPE_TILE scale — see `tilelang_kernel.py:1849` `T.gemm(S_shared,
KV_shared, acc_o, ...)`). The microbench number was therefore not
meaningful as a production comparison.

## Why this backend exists

The DSv4 sparse MLA decode kernel is K-cache-bandwidth-bound on AMD
gfx950. TileLang's HIP backend currently emits the K-cache load as
two-step `global_load + ds_write` for N=8 / N=16 byte chunks (TileLang's
`cp_async_gs<N>` specializes only N=4 for the LDS-direct path on AMD;
see the upstream `tl_templates/hip/copy.h`). FlyDSL exposes
`rocdl.buffer_load_to_lds` directly and can emit `buffer_load_dwordx4 v0,
s[0:3], 0 offen lds` cleanly on gfx950.

This PR validates the *infrastructure* needed to call a FlyDSL kernel from
inside the SGLang dispatch path. It does not yet validate that a full
FlyDSL attention kernel beats TileLang end-to-end on real workloads —
that requires writing a feature-complete FlyDSL kernel (see "Roadmap").

## Hardware / software requirements

- AMD MI355X / **gfx950**
- ROCm 6.x
- `flydsl` Python package
- `aiter` Python package (provides `aiter.ops.flydsl.kernels.tensor_shim.GTensor`)

If any of these are missing, the capability check
([`is_flydsl_kgather_available`](../../python/sglang/srt/layers/attention/nsa/flydsl_kernel.py))
returns `(False, reason)` and the dispatch layer falls back to TileLang
with a one-time `RuntimeWarning`. Production hosts without FlyDSL/aiter
are safe.

## Enabling

```bash
# 1. Route the dispatch to the FlyDSL kgather-only backend.
export SGLANG_HACK_FLASHMLA_BACKEND=flydsl_kgather_only

# 2. Tell that backend to actually run the kgather kernel. The default
#    (no env) means selecting the backend has zero math impact even when
#    SGLANG_HACK_FLASHMLA_BACKEND accidentally leaks into a deploy.
export SGLANG_FLYDSL_EXERCISE=1

# Optional:
export SGLANG_FLYDSL_DEBUG=1         # log first kgather failure / feature reject
export SGLANG_FLYDSL_DEBUG_SYNC=1    # cuda.synchronize() after kgather — profiling only
```

The kgather kernel adds a per-layer-per-decode-step kernel launch
(~14 µs on MI355X) when enabled. **Do not enable in production traffic
until perf is measured end-to-end.**

### Interaction with the bs-aware dispatch override

`debug_flash_mla_adapter.py` contains a perf heuristic that overrides the
selected backend to `tilelang` for `bs ∈ [40, 248]` (see the comment at
the top of `flash_mla_with_kvcache_entrypoint`). The
`flydsl_kgather_only` backend is exempt from this override — selecting
it via `SGLANG_HACK_FLASHMLA_BACKEND=flydsl_kgather_only` takes effect
across all bs ranges (because it's a debug/diagnostic backend the user
opted into explicitly, not a perf-pickable production backend).

## Capability + feature checks

Before running the kgather kernel,
[`_check_kgather_supported`](../../python/sglang/srt/layers/attention/nsa/flydsl_kernel.py)
soft-checks the K cache and indices layout:

| Reject reason | Source |
|---|---|
| `k_cache or indices is None` | nil inputs |
| `k_cache dtype is not 1-byte (FP8)` | wrong dtype |
| `k_cache must be 4D` | wrong rank |
| `H_KV != 1` | non-MQA layout |
| `K cache packed width != 584` | layout mismatch (real DSv4 = 576 packed FP8 + 8 scale) |
| `k_cache is non-contiguous` | refuse a 200+ MB copy |
| `indices must be 3D` | wrong rank |
| `S_Q != 1` | not a decode shape |
| `indices dtype must be int32` | wrong dtype |
| `empty workload (BS=0 or TOPK=0)` | nothing to gather |

Rejections are *soft*: the function returns `(False, reason)` instead of
raising. The dispatch wrapper logs the reason once per unique reason and
skips that kgather call, then proceeds to the TileLang delegate. No
crashes, no math change.

The `extra_k_cache` / `extra_indices_in_kvcache` (dual cache) path is
checked independently with the same predicate, so a dual-cache request
will exercise both kgather calls when both pass the check.

## K cache layout (asserted, do not change without updating both
ends)

Per-token row layout in the FP8 K cache:

```
bytes [  0, 576)   : packed FP8 data (D + 2 * tail_dim = 448 + 128 = 576)
bytes [576, 584)   : per-NOPE_TILE scale bytes (8 bytes total; D / NOPE_TILE = 7
                     scales + 1 pad byte)
```

In-block stride between adjacent tokens is the full **584 bytes**
(`packed_w_full`), not the 576-byte packed region. An earlier iteration
of this code used 576 as the in-block stride; that matched a torch
reference that had the same bug (read shifted bytes for `in_block > 0`),
so test "byte-exact" passes were against a wrong reference. The current
code asserts the layout and matches TileLang.

## Reproducing the kgather benchmark

```bash
python3 benchmark/sparse_mla_decode_flydsl/bench_kgather.py
```

Default settings on MI355X: gather 20352 K rows of 576 bytes each
(workload = BS=159 batches × topk=128 indices per batch, matching the
captured DSv4-Pro `bs=192` decode workload).

Expected output:

```
K cache: NB=2897, BS_KV=128, packed_w_full=584
workload: BS=159, TOPK=128, grid_x=20352, per-call bytes=11,722,752
latency: median=14.245 µs, p90=14.245 µs (over 10 samples × 200 iters)
effective HBM BW: 823.0 GB/s
```

For context: the MI355X HBM peak is ~5.3 TB/s. The kernel hits ~15% of
peak in this microbench, primarily because the workload is small
(11.7 MB / call) and dispatch / launch overhead dominates. The kgather
result is reported as a kernel-level standalone microbench number; it is
**not** an end-to-end server speedup claim.

## Reproducing the correctness tests

```bash
# Synthetic K cache (no captured data needed).
pytest test/srt/test_flydsl_kgather.py -v

# With a captured DSv4 sparse FP8 K cache from a real workload.
SGLANG_FLYDSL_TEST_PICKLE=/path/to/microbench_bs192.pkl \
  pytest test/srt/test_flydsl_kgather.py -v
```

The test suite covers:

- **Kernel layer** (`TestFlyDSLKGatherKernel`):
  byte-exact gather vs `torch.gather` on synthetic shapes,
  bs ∈ {1, 2, 4, 8, 16, 32, 64, 128, 192},
  topk ∈ {1, 7, 16, 32, 48, 63, 64, 128},
  repeated indices, all-negative-sentinel indices, out-of-range indices,
  real captured K cache (when pickle env is set).
- **Capability layer** (`TestFlyDSLCapabilityGuards`):
  every documented reject reason fires a `(False, reason)` tuple
  without raising — H_KV != 1, packed width mismatch, non-contiguous,
  S_Q != 1, wrong index dtype, empty workload.
- **Dispatch layer** (`TestFlyDSLDispatchEntryPoint`):
  the FlyDSL backend returns TileLang's `(output, lse)` unchanged on
  finite entries; unsupported feature combos (e.g., non-contiguous K
  cache) do not raise inside the FlyDSL code path.

All tests skip cleanly on non-AMD / no-flydsl hosts. Tests that require
the captured pickle skip cleanly when `SGLANG_FLYDSL_TEST_PICKLE` is not
set; they do not gate on private paths.

## What this PR explicitly does **not** do

- **No FlyDSL attention math.** The dispatch returns TileLang's output
  for every selected backend in this PR.
- **No end-to-end perf claim.** The kgather µs/call number is a
  kernel-level microbench, not a server speedup.
- **No dual-cache / attn_sink / D_tail / online-softmax-across-iter
  implementation in FlyDSL.** TileLang handles all of these today and
  will continue to handle them when this backend is selected.
- **No vendoring of `aiter` internals.** The kernel imports
  `aiter.ops.flydsl.kernels.tensor_shim.GTensor`; if `aiter` is missing
  the capability check returns False and dispatch falls back.

## Roadmap (out of scope for this PR)

Production-grade FlyDSL attention requires, in roughly increasing
difficulty:

1. **A correct standalone full attention kernel.** The prior prototype
   was structurally wrong for DSv4 (separate V cache, fixed V scale).
   The replacement must use `V == K` with the same per-NOPE_TILE scale
   handling as TileLang.
2. **`D_tail` (64 BF16 elements per K row → `acc_o_tail`).** Adds ~14%
   compute on QK and SV vs the current sub-kernel scope.
3. **Dual cache (`extra_k_cache` + `extra_indices_in_kvcache`).** Real
   DSv4 workloads can have ~2× more KV traffic than a single chunk.
4. **Online softmax across multiple BI chunks** (`m_i / sumexp` carry).
5. **`Partial_O` / `Partial_LSE` emission matching the TileLang combine
   kernel input contract**, OR a fused combine.
6. **`attn_sink` folding** (currently handled by the combine kernel).
7. **Server-side end-to-end decode benchmark vs TileLang** at
   `bs ∈ {64, 128, 192, 256, 384, 512}` on real DSv4-Pro traffic. Only
   after this does any speedup claim become meaningful.
