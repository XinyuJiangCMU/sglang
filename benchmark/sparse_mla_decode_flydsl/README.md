# FlyDSL DSv4 sparse MLA decode — benchmarks (AMD gfx950)

WIP benchmarks for the FlyDSL backend of `dpsk_v4_fp8_attention_fwd`.

**Scope:** these scripts measure the standalone FlyDSL kernels. They do
**not** measure end-to-end server latency. The kernel that beats TileLang
on the standalone benchmark is **not** yet wired into the production
dispatch path — see [`docs/developer_guide/amd_flydsl_sparse_mla.md`](
../../docs/developer_guide/amd_flydsl_sparse_mla.md) for what's
production-integrated today vs. what's prototype-only.

| Script | What it measures |
|---|---|
| `bench_kgather.py` | FlyDSL weapon-1 K-gather kernel (standalone). Production-integrated as `flydsl_kgather_only` backend. |
| `bench_subkernel_fp8.py` | Prototype full FP8 sparse attention sub-kernel (NOT yet integrated). Reports the "0.4 µs/batch" headline number — sub-kernel scope, not feature parity with TileLang. |
| `bench_compare_baselines.py` | Times TileLang + Triton on the same captured pickle for side-by-side comparison. |

## Hardware requirements

- AMD MI355X / gfx950
- `flydsl` Python package
- `aiter` Python package (provides `tensor_shim.GTensor`)
- ROCm 6.x

## Running

```bash
# Standalone kgather perf (cheap, no pickle needed)
python3 benchmark/sparse_mla_decode_flydsl/bench_kgather.py

# Prototype FP8 sub-kernel perf (no correctness vs TileLang — scope mismatch)
python3 benchmark/sparse_mla_decode_flydsl/bench_subkernel_fp8.py

# Side-by-side TileLang + Triton baselines on a captured pickle
SGLANG_FLYDSL_TEST_PICKLE=/path/to/microbench_bs192.pkl \
  python3 benchmark/sparse_mla_decode_flydsl/bench_compare_baselines.py
```

## Honest scope

The prototype sub-kernel does **not** include:

- `D_tail` (64 BF16 elements per K row; -14% compute)
- `extra_k_cache` / `extra_indices_in_kvcache` (dual cache; ~2× more KV traffic on real workload)
- Online softmax across multiple BI chunks (m_i / sumexp carry)
- `attn_sink` folding (handled by TileLang's combine kernel)
- Partial_O / Partial_LSE emission + combine kernel

So a sub-kernel µs-per-batch number is not directly comparable to
TileLang's full `dpsk_v4_fp8_attention_fwd` µs-per-batch number. Treat
them as evidence the FlyDSL toolchain can hit competitive perf at the
sub-kernel level, not as a production speedup claim.

The kernel that IS production-integrated today (`flydsl_kgather_only`
backend) runs the kgather kernel as a smoke exercise and then **delegates
all attention math to TileLang**. The model's numerical output is
unchanged when this backend is selected.
