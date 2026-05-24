# AMD FlyDSL backend for DSv4 sparse FP8 MLA decode

**Status: WIP / prototype. Production-safe by default.**

This document describes the FlyDSL backend for AMD MI355X (gfx950) sparse
FP8 MLA decode attention in DeepSeek-V4-Pro. Two pieces exist today:

1. **Production-integrated `flydsl_kgather_only` backend**
   ([`python/sglang/srt/layers/attention/nsa/flydsl_kernel.py`](
   ../../python/sglang/srt/layers/attention/nsa/flydsl_kernel.py)) —
   exercises a validated FlyDSL weapon-1 K-gather kernel against the
   live K cache and then **delegates the actual attention math to the
   production TileLang backend**. The model's numerical output is
   unchanged.

2. **Standalone prototype full attention sub-kernel**
   ([`benchmark/sparse_mla_decode_flydsl/bench_subkernel_fp8.py`](
   ../../benchmark/sparse_mla_decode_flydsl/bench_subkernel_fp8.py))
   that runs end-to-end Q@K^T → softmax → S@V in FlyDSL but covers
   only a sub-kernel slice (see "Scope" below). **Not** integrated
   into the request path.

## When to enable

The production-integrated backend (#1) is off by default and adds latency
on enable. Only enable for:

- Smoke-testing FlyDSL on a real production workload
- Profiling that the FlyDSL pipeline survives CUDA graph capture / replay
  and concurrent decode requests on real K cache pressure

It is **not** expected to improve perf in this form. See "Roadmap"
below for what's needed before the full kernel can replace TileLang.

## Hardware / software requirements

- AMD MI355X / **gfx950**
- ROCm 6.x
- `flydsl` Python package
- `aiter` Python package (provides `tensor_shim.GTensor`)

The capability check
([`is_flydsl_kgather_available`](../../python/sglang/srt/layers/attention/nsa/flydsl_kernel.py))
returns `(False, reason)` cleanly on unsupported systems; the dispatch
layer falls back to TileLang with a one-time warning.

## Enabling the backend

```bash
# 1. Route the sparse FP8 MLA dispatch to the FlyDSL kgather-only backend.
export SGLANG_HACK_FLASHMLA_BACKEND=flydsl_kgather_only

# 2. Tell that backend to actually run the kgather kernel (default off so
#    selecting the backend has zero math impact even when the env flag
#    accidentally leaks into a production deploy).
export SGLANG_FLYDSL_EXERCISE=1

# Optional:
export SGLANG_FLYDSL_DEBUG=1         # log the first kgather failure
export SGLANG_FLYDSL_DEBUG_SYNC=1    # add torch.cuda.synchronize() after
                                      # the kgather kernel (deterministic
                                      # timing — only enable for profiling)
```

If `flydsl` or `aiter.ops.flydsl.kernels.tensor_shim` are not importable,
or the GPU is not gfx95*, the backend logs a `RuntimeWarning` and falls
back to TileLang automatically.

## Scope of the prototype sub-kernel

The standalone prototype kernel covers:

| Piece | Status |
|---|---|
| Sparse K/V gather via `indices[b, n]` → block/in_block decomposition | implemented |
| FP8 e4m3 inline dequant with real per-NOPE_TILE scale (K and V) | implemented |
| `softmax_scale` multiply (DSv4 default 1/√(D+D_tail)) | implemented |
| Q @ K^T via `mfma_f32_16x16x32_bf16` | implemented |
| Row-wise softmax over BI=64 via LDS cross-lane reduction | implemented |
| S @ V via `mfma_f32_16x16x16bf16_1k` | implemented |
| Bit-identical correctness vs PyTorch reference | implemented |

The sub-kernel does **not** cover:

| Piece | Why it matters |
|---|---|
| `D_tail` (64 BF16 elements per K row) | TileLang has D+D_tail=512 worth of compute on QK and SV; sub-kernel has only D=448 (~14% less) |
| `extra_k_cache` / `extra_indices_in_kvcache` (dual cache) | TileLang processes both caches; ~2× less KV traffic in sub-kernel on real workloads |
| Online softmax across multiple BI chunks (m_i / sumexp carry) | TileLang carries running max/sum across many BI-sized chunks; sub-kernel is single-pass over one BI=64 chunk |
| Real K cache row stride (584 bytes/row vs sub-kernel's synthetic 456) | ~22% less HBM per K row in the sub-kernel |
| `attn_sink` folding | Handled by TileLang's combine kernel |
| Partial_O / Partial_LSE emission + combine kernel | Sub-kernel writes directly to `O` |

A sub-kernel µs-per-batch number is therefore **not** directly comparable
to TileLang's `dpsk_v4_fp8_attention_fwd` µs-per-batch number.

## Reproducing benchmarks

```bash
# K-gather kernel only (matches the production-integrated backend)
python3 benchmark/sparse_mla_decode_flydsl/bench_kgather.py

# Prototype sub-kernel (NOT integrated into dispatch)
python3 benchmark/sparse_mla_decode_flydsl/bench_subkernel_fp8.py

# TileLang + Triton baselines on a captured DSv4 pickle (apples-to-apples
# comparison: side-by-side both run the full TileLang/Triton kernel).
SGLANG_FLYDSL_TEST_PICKLE=/path/to/microbench_bs192.pkl \
  python3 benchmark/sparse_mla_decode_flydsl/bench_compare_baselines.py
```

## Reproducing correctness tests

```bash
# Synthetic K cache (no pickle needed)
pytest test/srt/test_flydsl_kgather.py -v

# Against captured DSv4 sparse FP8 K cache
SGLANG_FLYDSL_TEST_PICKLE=/path/to/microbench_bs192.pkl \
  pytest test/srt/test_flydsl_kgather.py -v
```

The test gathers up to 1024 K rows via the FlyDSL kernel and compares
byte-exactly against `torch.gather`. It skips cleanly on non-AMD hosts
or when `flydsl`/`aiter` are missing.

## Roadmap (what needs to happen before production)

1. Wire the standalone sub-kernel ([`bench_subkernel_fp8.py`](
   ../../benchmark/sparse_mla_decode_flydsl/bench_subkernel_fp8.py))
   into the dispatch path with a capability check that bails out to
   TileLang for unsupported cases (currently essentially every case
   the sub-kernel doesn't yet handle).
2. Add `D_tail` accumulation (a second mfma path per K_v tile).
3. Add dual cache (extra_k_cache loop, separate Partial_O/Partial_LSE write).
4. Add online softmax across multiple BI chunks (m_i / sumexp carry).
5. Emit `Partial_O` and `Partial_LSE` matching TileLang's combine
   kernel's input contract, OR fuse combine.
6. Add `attn_sink` folding.
7. Server-side end-to-end decode benchmark vs TileLang at
   bs ∈ {64, 128, 192, 256, 384, 512}.
8. Correctness against real DSv4 pickle data (not just the bit-identical
   torch reference at the same sub-kernel scope).

## Internal contract (layout asserted by the kgather kernel)

Asserted in
[`flydsl_kernel._exercise_kgather_on_real`](
../../python/sglang/srt/layers/attention/nsa/flydsl_kernel.py):

- `k_cache.dtype.itemsize == 1` (FP8)
- `k_cache.shape == (NB, BS_KV, 1, 584)` — H_KV=1 (MQA), 576-byte FP8 +
  8-byte per-NOPE_TILE scale region
- `indices.shape == (BS, 1, TOPK)`, `dtype == int32`
- `k_cache.is_contiguous()` — non-contiguous K cache surfaces as a
  RuntimeError rather than silently triggering a multi-hundred-MB copy
- In-block row stride is the full **584-byte** `packed_w_full`, **not**
  the 576-byte packed region — every byte offset calculation must use
  `packed_w_full` so the kernel sees the same bytes as TileLang would

A prior iteration of this code used 576 as the in-block stride. That
matched a torch reference that had the same bug (read shifted bytes),
so test "byte-exact" passes were against a wrong reference. The current
code asserts the layout and matches TileLang.
