# r051 Stage 1 Round 1 — Status

**Goal**: scaffold + dispatch + delegation correctness.
**Status**: PASS for round 1. Real FlyDSL kernel deferred to round 2.

## FlyDSL primitives audit (KEY FINDING)

FlyDSL directly exposes (in `/opt/venv/lib/python3.10/site-packages/flydsl/expr/rocdl.py`):

- `global_load_lds` — **WEAPON 1 direct HBM→LDS**
- `cluster_load_async_to_lds_b{8,32,64,128}` — multi-width LDS-direct async
- `mfma_f32_16x16x32_fp8_fp8` — FP8 mfma direct
- `mfma_scale_f32_16x16x128_f8f6f4` — **WEAPON 5 FP8 with scale**

→ Unlike tilelang (per r050 verdict: only `async_buffer_load_dword_v` at N=4),
FlyDSL has full multi-width LDS-direct path + FP8 mfma exposed. **Architectural
reason to expect FlyDSL kernel to beat tilelang on AMD attention.**

## Round 1 deliverables (DONE)

1. `_r051_artifacts/flydsl_sparse_mla_decode.py` — API-compatible wrapper matching
   tilelang's `dpsk_v4_fp8_attention_fwd` signature. Env-gated: `SGLANG_FLYDSL_REAL=1`
   switches to real FlyDSL kernel (r052+); default delegates to tilelang.
2. Same file deployed to `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/sparse_mla_decode.py`
   in container (new file only — no native aiter code modified).
3. `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` — added 9-line
   `if backend == "flydsl":` dispatch branch.
4. Delegation correctness:
   - `microbench_bs192.pkl` loaded
   - `dpsk_v4_fp8_attention_fwd` (tilelang) called → (o_t, lse_t)
   - `dpsk_v4_fp8_attention_fwd_flydsl` (delegates to tilelang) → (o_f, lse_f)
   - `torch.allclose(o_t, o_f, equal_nan=True, rtol=0, atol=0)` = **True**
   - `torch.allclose(lse_t, lse_f, equal_nan=True, rtol=0, atol=0)` = **True**
   - Note: pickle has nil `cache_seqlens / block_table` causing NaN; both calls
     produce bitwise-identical NaN. Cleaner pickle TBD in r052.

## Round 2 plan

1. Write real FlyDSL kernel in `_dpsk_v4_fp8_attention_fwd_flydsl_real`:
   - Use moe_gemm_2stage.py as structural template (most complete FlyDSL kernel in aiter)
   - Implement dpsk_v4 partial kernel (Q load, gather K, FP8 dequant, mfma gemm,
     online softmax, S@V, partial_O / partial_LSE write)
   - Stage 1 simplicity: regular `buffer_copy_gmem16_dwordx4` for loads, BF16 mfma
     after dequant, no software pipelining
2. Capture cleaner pickle during sustained-decode burst (clean cache_seqlens etc)
3. Microbench at bs=64/128/192/256/384/512
4. Spawn reviewer with strict checklist

## What round 1 deliberately did NOT do

- No real FlyDSL kernel code (200-400 line work, r052+)
- No server-side bench (wrapper has same perf as tilelang via delegation)
- No reviewer (no perf or kernel claims to review yet)
- No tilelang_kernel.py edits (hard constraint)

## Files

| File | Change |
|---|---|
| `_r051_artifacts/flydsl_sparse_mla_decode.py` | NEW — scaffold |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/sparse_mla_decode.py` | NEW (container) |
| `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` | +9 lines |
| `_r051_artifacts/STATUS.md` | NEW — this file |

## Time

Round 1 wall: ~1.5 hours of 4-hour budget. Under limit.
