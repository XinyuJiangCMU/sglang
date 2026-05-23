# r051 — Status

**Goal**: FlyDSL implementation of DSv4 sparse FP8 MLA decode attention that
beats both tilelang AND triton across all batch sizes.

## Round 1: scaffold + dispatch + delegation correctness  ✅

1. `_r051_artifacts/flydsl_sparse_mla_decode.py` — API-compatible wrapper
   matching tilelang's `dpsk_v4_fp8_attention_fwd` signature.
2. Deployed to `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/sparse_mla_decode.py`
   in container (new file only — no native aiter code modified).
3. `debug_flash_mla_adapter.py` — added 9-line `if backend == "flydsl":` branch.
4. Env-gated `SGLANG_FLYDSL_REAL=0` (default) delegates to tilelang for
   byte-equal correctness baseline.

## Round 3 (incremental): weapon-1 on REAL DSv4 K cache ✅

`_r051_artifacts/test_weapon1_on_real_kcache.py` — same verified
weapon-1 kernel pattern, but with source tensor = real 216 MB DSv4 FP8
K cache (shape `(2897, 128, 1, 584)`, captured from microbench pickle).

Same ISA emitted: `buffer_load_dwordx4 v0, s[0:3], 0 offen lds`.
Buffer resource descriptor handles real-tensor strides correctly:
```asm
s_mov_b32 s3, 0x27000     ; rsrc format bits
s_mov_b32 s2, -1          ; rsrc num_records
buffer_load_dwordx4 v0, s[0:3], 0 offen lds
```

ISA saved to `.humanize/round-051-flydsl-attention/w1real_test_kernel.isa.s`.

This unblocks the real kernel implementation in round 4+: weapon-1 K
cache gather will work on the actual DSv4 layout without further infra
investigation.

## Round 2: WEAPON 1 emission verification  ✅

**Decisive evidence that FlyDSL can emit `buffer_load_dwordx4 ... lds` on gfx950.**

- Test: `_r051_artifacts/test_weapon1_emission.py`
- ISA: `.humanize/round-051-flydsl-attention/weapon1_test_kernel.isa.s`
- Verification doc: `.humanize/round-051-flydsl-attention/WEAPON1_VERIFICATION.md`

Final ISA excerpt:
```asm
.amdhsa_kernel weapon1_test_kernel
  buffer_load_dwordx4 v0, s[0:3], 0 offen lds      ← WEAPON 1
.end_amdhsa_kernel
```

Counts: `buffer_load_*lds` = 1, `ds_write_*` = 0 (vs tilelang per r050:
weapon 1 = 0, ds_write = 88).

**This is the FIRST aiter FlyDSL kernel to use `rocdl.buffer_load_to_lds`.**
All existing aiter FlyDSL kernels (moe_gemm, mfma_preshuffle, chunk_gated_delta_h)
use the 2-step `buffer_load` → `vector.store` pattern (same as tilelang's broken
path). Architectural premise from r050 confirmed: FlyDSL has a real advantage
over tilelang on AMD attention.

API quirks documented in WEAPON1_VERIFICATION.md.

## Round 3+: real FlyDSL kernel implementation  🚧 in progress

Strategy: incremental milestones, each commitable + validatable.

### Milestone 3A — kernel signature skeleton

`_dpsk_v4_fp8_attention_fwd_flydsl_real` matches tilelang signature, returns
correct-shape zero tensors. Validates dispatch path with `SGLANG_FLYDSL_REAL=1`
without breaking server.

### Milestone 3B — weapon 1 Q+K load demo

FlyDSL kernel emits weapon-1 loads for Q + first K block, writes loaded bytes
to scratch buffer. Math still ref-computed in torch. Validates real-tensor
weapon-1 path inside the dispatch.

### Milestone 3C — single-head QK gemm

FlyDSL kernel does Q@K^T for one attention head, online-softmax in torch.
Validates `mfma_f32_16x16x32_bf16` from FlyDSL on real shapes.

### Milestone 3D — full single-cache decode

Full kernel for single-cache decode (topk_1 only, skip extra_cache). Online
softmax + S@V in FlyDSL. Output matches tilelang within atol=5e-2.

### Milestone 3E — dual cache + correctness final

Add `extra_k_cache` / `extra_indices_in_kvcache` second pass. Full correctness
validation vs tilelang on captured microbench pickle.

### Stage 2: weapons 2+5

W2 (pipelining), W5 (fp8 mfma direct, skip dequant if `mfma_scale_f32_16x16x128_f8f6f4`
maps cleanly).

### Stage 4: validate + writeup

Per-bs server bench. `kernel-wiki/09_flydsl_attention.md` writeup.

## Files

| File | Status |
|---|---|
| `_r051_artifacts/flydsl_sparse_mla_decode.py` | round 1 scaffold (delegates) |
| `_r051_artifacts/test_weapon1_emission.py` | round 2 weapon-1 verification |
| `_r051_artifacts/STATUS.md` | this file |
| `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` | +9 line flydsl branch |
| `.humanize/round-051-flydsl-attention/WEAPON1_VERIFICATION.md` | not in git |
| `.humanize/round-051-flydsl-attention/weapon1_test_kernel.isa.s` | not in git |

## Constraint reminders (from user)

- No upstream PR — fork only (XinyuJiangCMU/sglang).
- No edits to tilelang_kernel.py (Thomas Wang's r033/r047 baseline).
- No edits to aiter native code — only NEW files in
  `aiter/ops/flydsl/kernels/`.
- Correctness before perf. Bench numbers measured, never estimated.
- Autonomous mode, no time limit.
