# [AMD/gfx950] FlyDSL DSv4 sparse FP8 MLA decode — sub-kernel 4.5× faster than tilelang

## TL;DR (honest, post-review)

A FlyDSL sub-kernel covering ~70-80% of `dpsk_v4_fp8_partial_kernel`'s work
runs at **0.407 µs/batch** on AMD MI355X (gfx950), vs tilelang's full
`dpsk_v4_fp8_attention_fwd` at **1.825 µs/batch** and triton's
`triton_fp8_attention_fwd` at **0.868 µs/batch**. Correctness validated
bit-identically against PyTorch reference using the same FP8 dequant
formula.

The ratio (~4.5× faster than tilelang) is **NOT apples-to-apples** —
this PR explicitly documents what the FlyDSL sub-kernel does and does
not do vs tilelang full feature parity. See **"Scope honesty"** below.

## Numbers (DSv4 partial-kernel shape, BS=159, M_HEADS=128, BI=64, D=448, D_V=448)

| Backend | Format | µs/batch | Scope |
|---|---|---|---|
| tilelang | FP8 sparse | 1.825 | full (D=512 incl. D_tail, dual cache, online softmax, attn_sink, combine) |
| triton | FP8 sparse | 0.868 | full |
| **FlyDSL** | **FP8 sparse** | **0.407** | **sub-kernel** |

Baselines re-bench'd by reviewer subagent and reproduced within 2-15%.

## Scope honesty: what FlyDSL is NOT yet doing vs tilelang

- **D_tail** (64 BF16 elements per K row) — FlyDSL has `D = 448`; tilelang
  has `D + D_tail = 512` (~14% less compute on QK and SV)
- **Dual cache** — tilelang processes `extra_k_cache` (~26 extra tokens
  per batch on the pickle); FlyDSL processes only the primary 64-token
  chunk (~2× less KV traffic on real workload)
- **Online softmax** across multiple BI chunks (m_i/sumexp carry) — FlyDSL
  is single-pass; tilelang carries across many BI-sized chunks
- **K row stride** — FlyDSL uses 456 bytes/row; real DSv4 cache is 584
  bytes/row (-22% HBM per K row)
- **attn_sink** folding — handled by tilelang's combine kernel

These get closed in r052+ before any production claim. Until then, the
sub-kernel result is evidence the architectural direction (FlyDSL +
weapon-1 + clean mfma usage) is viable on AMD.

## What's validated

### Foundational primitives (rounds 1-12)
- weapon-1 emission on synthetic + real 216 MB DSv4 FP8 K cache
  (`buffer_load_dwordx4 v0, s[0:3], 0 offen lds` ISA verified)
- K-gather kernel byte-exact vs `torch.gather` on real K cache
  (4096/4096 rows; packed 576 B + scale 8 B regions)
- `mfma_f32_16x16x32_bf16` validated standalone (32.0×256 correct broadcast
  output) and on HBM data with full lane layout (256/256 byte-exact at
  16x16x32, 8192/8192 within 0.05 tol at full D=448 multi-tile)
- FP8 e4m3 dequant bit-formula correctness (round 10) — note this only
  proves the kernel implements the same bit formula as torch, not vs
  `torch.float8_e4m3fn` semantics; tilelang uses the same formula
- LDS cross-lane softmax (256/256 within 1e-4)
- Composed micro-attention 16×16 (Q@K → softmax → S@V, 256/256 within 5e-3)

### Real-shape kernel (rounds 13-16b)
- BF16 dense @ DSv4 shape — `57344/57344` within tol (round 14)
- Multi-batch BS=159, BF16 dense — 0.216 µs/batch (round 14b)
- BF16 sparse K/V gather — 0.234 µs/batch (round 15)
- **FP8 sparse + per-NOPE_TILE V scale + softmax_scale — 0.407 µs/batch
  with bit-identical PyTorch reference passing 9,117,696/9,117,696 finite
  positions** (round 16b)

## Dispatch wiring

```bash
SGLANG_HACK_FLASHMLA_BACKEND=flydsl     # route to FlyDSL backend
SGLANG_FLYDSL_REAL=1                    # use _real entry (not delegation)
SGLANG_FLYDSL_EXERCISE=1                # actually run FlyDSL kernel
```

Currently `_real` runs the validated weapon-1 kgather kernel and **delegates
math to tilelang** (round 6). The perf-winning round-16b kernel is a
standalone microbench file; wiring it into the dispatch (replacing the
tilelang delegation) + the missing-feature work above is r052+.

## Container deploy

```bash
docker cp _r051_artifacts/flydsl_sparse_mla_decode.py \
  hai-1:/sgl-workspace/aiter/aiter/ops/flydsl/kernels/sparse_mla_decode.py
```

## Test plan

```bash
# Primitive byte-exact checks
docker exec hai-1 python3 /mnt/.../test_weapon1_emission.py            # PASS
docker exec hai-1 python3 /mnt/.../test_kgather_full.py                # 4096/4096
docker exec hai-1 python3 /mnt/.../test_mfma_qk_full_d448.py           # 8192/8192 (tol=0.05)
docker exec hai-1 python3 /mnt/.../test_fp8_dequant.py                 # 256/256 bit-formula
docker exec hai-1 python3 /mnt/.../test_micro_attention.py             # 256/256 within 5e-3

# Real-shape correctness + perf
docker exec hai-1 python3 /mnt/.../test_attention_realsize_bf16.py     # 9117696/9117696, 0.216 µs/batch
docker exec hai-1 python3 /mnt/.../test_attention_sparse_bf16.py       # 9117696/9117696, 0.234 µs/batch
docker exec hai-1 python3 /mnt/.../test_attention_fp8_sparse.py        # 9117696/9117696 finite, 0.407 µs/batch
```

## Reviewer notes addressed (this revision)

A subagent code review on the previous PR draft caught issues:

1. ✅ V dequant hardcoded `scale=7` — **fixed**: real per-NOPE_TILE V scale load (round 16b)
2. ✅ No `softmax_scale` multiply — **fixed**: DSv4 default 1/√(D+D_tail) baked in (round 16b)
3. ✅ No correctness validation on FP8 kernel — **fixed**: bit-identical
   PyTorch reference, 9,117,696/9,117,696 finite positions pass (round 16b)
4. ✅ "3.89× full feature parity" headline — **retracted**: now reads
   "sub-kernel 4.5× faster" with explicit scope-honesty section
5. ✅ "byte-exact" misuse for round 9 — **fixed**: STATUS.md now reads
   "byte-exact / within bf16 tol" and notes round 9 uses tol=0.05

Still on the to-do list (r052+):
- D_tail handling, dual cache, online softmax across chunks, real K row
  stride (584 vs 456), attn_sink folding
- Wire round-16b kernel into dispatch path (currently `_real` still
  delegates math to tilelang after running kgather as exercise)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
