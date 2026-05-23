# AMD gfx950 (MI350X / MI355X / CDNA4) — TileLang optimization weapons

Notes for engineers writing or optimizing TileLang kernels targeting the AMD
CDNA4 architecture (MI350X / MI355X), in the context of `sglang`'s sparse MLA
attention for DeepSeek-V4. These map NVIDIA-specific primitives (TMA, WGMMA,
named barriers, cluster, TMEM) to the closest AMD equivalents and call out
which TileLang patterns trigger which AMD instruction.

Anchors used as concrete examples below:
- AMD kernel: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` —
  `dpsk_v4_fp8_partial_kernel` (~line 1582) and `dpsk_v4_fp8_attention_fwd`
  (~line 2474) on the `amd/deepseek_v4` branch.
- NV reference: `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` in
  [DeepSeek's FlashMLA repo](https://github.com/deepseek-ai/FlashMLA).

These are working notes, not authoritative documentation. Verify against
[ROCm CDNA architecture whitepapers](https://rocm.docs.amd.com/) and
[TileLang source](https://github.com/tile-ai/tilelang) before relying on
specific behaviour.

---

## Why doesn't AMD have NV's exact features?

Three reasons, briefly.

**Design philosophy.** NV invests in many small SMs (132 on H100, 192 on B200)
each with complex per-SM hardware (TMA, TMEM, WGMMA). AMD CDNA goes the
other way: fewer but fatter CUs (256 on MI350X) with more matrix engines per
CU and more LDS/L1 capacity. The bet is "more parallel, simpler per-unit"
versus NV's "fewer, smarter per-unit".

**History and target market.** AMD CDNA was originally HPC-first (Frontier,
El Capitan) before AI features were grafted on. NV designed each tensor-core
generation explicitly for AI workloads. The roadmap gap is roughly 2-3 generations.

**AMD substitutes via different mechanisms.** Several NV features have
functional-but-not-syntactic AMD equivalents:

| NV feature | gfx950 equivalent | Equivalence |
|---|---|---|
| TMA (async block copy with transaction barrier) | `global_load_lds_*` + `s_waitcnt` | ~80% (no automatic transaction-barrier signal) |
| WGMMA (async warp-group MMA) | MFMA + dual-issue (new in CDNA4) | ~50% (dual-issue overlaps MFMA with VMEM, but no async MFMA chain) |
| `cp.async` commit/wait_group | `s_waitcnt vmcnt(N)` | ~90% |
| 16 named barriers | 1 `s_barrier` + LDS atomics | ~30% (much weaker) |
| TMEM (separate accumulator memory) | VGPR | None (different design point) |
| Threadblock cluster (CGA) | — | None |
| `cvt_fp8x8_bf16x8` HW dequant | `mfma_scale_f32_*_f8f6f4` (skip dequant entirely) | Better (AMD can multiply FP8 directly) |

The real gaps that hurt the DSv4 sparse MLA decode kernel today are
async-MFMA, named barriers (for warp specialization), and threadblock cluster.
The rest are workable with different syntax.

---

## The 8 weapons

Ordered by likely return-on-effort for kernel optimization on gfx950, based
on profiling the DSv4-Pro sparse MLA decode path on MI350X.

### Weapon 1 — `global_load_lds_*` (HBM → LDS, bypass VGPR)

**What it is.** A class of gfx950 instructions
(`global_load_lds_b32 / b64 / b128`, also `buffer_load_lds_*`) that copies
from HBM straight into LDS without staging through vector registers.

**NV equivalent.** `cp.async` (Ampere onwards). The functional intent is the
same: free VGPRs and hide load latency behind compute.

**Why it matters.** The 2-step path (`global_load` → register → `ds_write`)
occupies VGPRs during the load, capping wave occupancy. The 1-step path
keeps VGPRs available for compute.

**In TileLang.** A well-formed `T.copy(global_tensor, lds_buffer)` is lowered
to `global_load_lds_*` when the operand shape, alignment, and dtype line up.
The compiler may fall back to the 2-step path if the destination access is
expressed element-wise rather than as a contiguous tile copy.

To verify, dump the generated amdgcn and grep for `global_load_lds_b128`
near the K load. If you see `global_load_b128` followed shortly by
`ds_write_b128` instead, the fast path was not taken; the fix is usually to
express the LDS-side write as a `T.copy(...)` slice instead of an element-wise
loop, and to confirm the source is 16-byte aligned.

### Weapon 2 — Software pipelining via `T.Pipelined(num_stages=K)`

**What it is.** With `K > 0`, the TileLang compiler emits double or
ping-pong LDS buffers, schedules load(N+K) concurrently with compute(N), and
inserts `s_waitcnt vmcnt(K-1)` to overlap memory with compute.

**NV equivalent.** A producer warpgroup loading K tiles ahead of consumer
warpgroups in flash_mla. On AMD this happens within a single CTA, not across
warp groups.

**Why it matters.** This is the AMD path to overlap dequant and other LDS
work with the next K-tile's load — the same throughput goal that NV's
warp-specialization achieves architecturally.

**Why it sometimes fails to compile.** The compiler's `PipelineInjector`
rejects the loop body if independent stages cannot be reordered without
violating buffer access ordering. A representative error:

```
PipelineInjector ValidatePipelineBody: two statements with buffer access
dependency in the same stage of the software pipeline cannot be reordered
```

The remedy is to split the loop body so each conceptual stage writes a
distinct LDS buffer that subsequent stages do not also write. A sketch:

```python
K_packed_buf = T.alloc_shared([2, BI, PACKED_W4], "uint32")  # ping-pong
KV_buf       = T.alloc_shared([2, BI, D], "bfloat16")

for k_i in T.Pipelined(inner_iter_1, num_stages=2):
    bi = k_i % 2

    # Stage 0: load only (no consumers other than stage 1)
    T.copy(K_combined_1[...], K_packed_buf[bi])

    # Stage 1: dequant of the previous iteration
    dequant(K_packed_buf[1 - bi], KV_buf[1 - bi])

    # Stage 2: gemm on the iteration before that
    T.gemm(Q_shared, KV_buf[1 - bi], acc_s)
```

LDS budget approximately doubles, so verify against the 160 KB/CU limit
before claiming a win.

### Weapon 3 — Dual-issue (new in CDNA4)

**What it is.** gfx950 hardware can issue VALU and VMEM in the same cycle.
CDNA3 cannot.

**NV equivalent.** Long-standing ILP across functional units.

**How to trigger.** Compiler-automatic. The kernel author's job is to keep
the instruction stream from being long stretches of a single class. Interleave
VALU work (dequant, softmax) with VMEM (next K load) within the same
`T.Pipelined` stage.

**How to verify.** In the amdgcn dump, look for `v_mfma_*` and `ds_read_b128`
appearing on adjacent lines without an intervening `s_waitcnt`. That is
dual-issue happening.

### Weapon 4 — MFMA LDS-direct mode (skip the VGPR staging)

**What it is.** MFMA instructions can read operands directly from LDS rather
than from VGPRs. The compiler emits `v_mfma_*` with `ds_*` operands.

**NV equivalent.** WGMMA reads from shared memory natively.

**How to trigger in TileLang.** `T.gemm(A_shared, B_shared, C_register)` —
when both A and B are LDS allocations rather than register fragments, the
compiler can pick LDS-direct MFMA. The `dpsk_v4_fp8_partial_kernel` already
uses this pattern.

**How to verify.** In amdgcn, MFMA instructions should reference LDS
addresses rather than VGPRs. If the compiler inserts `ds_read_b128 → v_mov →
v_mfma`, the LDS-direct mode was not taken.

### Weapon 5 — `mfma_scale_f32_*_f8f6f4` (multiply FP8 directly)

**What it is.** A gfx950 MFMA variant that accepts FP8 operands plus a
block scale, accumulates in FP32, and skips the dequant-to-BF16 step
entirely.

**NV equivalent.** None directly — NV's `cvt_fp8x8_bf16x8` is faster than
manual dequant but still produces BF16, which then goes into WGMMA. AMD can
skip the BF16 intermediate.

**Why it matters in the DSv4 kernel.** The current kernel spends roughly
10–20% of attention compute on the bit-twiddling FP8 → BF16 dequant
implemented in `dpsk_v4_fp8_partial_kernel`. A direct FP8 GEMM eliminates
those instructions outright.

**How to use.** In principle:

```python
T.gemm(Q_shared,           # bf16
       K_packed_shared,    # FP8 packed
       acc_s,              # fp32 accumulator
       scale_b=K_scale_shared,
       b_dtype="float8_e4m3fnuz")
```

The feasibility depends on whether TileLang's `T.gemm` exposes a `b_dtype`
parameter that lowers to `mfma_scale_*_f8f6f4` on gfx950 and accepts the
particular block-scale layout the kernel uses. This needs to be confirmed
against TileLang source before counting on it.

**Layout constraints.** `mfma_scale_*` is strict about operand tiling and
the block-scale alignment. The packed-uint32 K layout in the current kernel
may need re-organization in LDS to match the instruction's expected layout.

### Weapon 6 — Wide wave (64 threads)

**What it is.** An AMD wavefront is 64 threads (NV warp is 32). A single
MFMA instruction is issued by all 64 lanes together.

**Why it matters.** One MFMA covers double the work per issue compared to
NV's WMMA. Wave-level barriers, shuffles, and reductions amortize over 64
data points. Wave-internal synchronization is implicit (no `__syncwarp`
needed).

**How to use.** Pick `threads` in `T.Kernel(..., threads=N)` as a multiple
of 64. The current kernel uses `threads=512` = 8 waves/CTA, which sits at
the gfx950 wave-per-CU maximum (8). Smaller CTAs (256 = 4 waves) may
increase CTAs/CU but reduce per-MFMA tile.

### Weapon 7 — Hardware 2:4 structured sparsity

**What it is.** gfx950 MFMA can skip zeroed elements in a 2:4 pattern (two
zeros per group of four), theoretical 2x speedup on sparsified inputs.

**Relevance to DSv4 sparse MLA.** Coarse sparsity (token-level top-K) is
already done above the kernel. Fine-grained 2:4 sparsity inside the K tile
is a different question; FP8 KV data is typically dense, so the structured
pattern is unlikely to appear without engineered support.

**Recommendation.** Defer. Low expected ROI without a sparsity-aware
KV layout.

### Weapon 8 — LDS atomics for cross-wave coordination

**What it is.** LDS supports atomic operations (e.g. `ds_atomic_add`). One
wave can write and increment a counter; another wave spin-waits on the
counter to coordinate.

**NV equivalent.** Named barriers (much cheaper).

**When to use.** Only if you need to simulate warp specialization on gfx950
manually. The LDS-atomic-based approach is approximately 5–10x more
expensive than NV's named-barrier path, often enough to eat the
specialization win.

**Recommendation.** Avoid unless other paths have been exhausted. Software
pipelining (Weapon 2) achieves most of the same overlap goal without the
synchronization tax.

---

## Status table for `dpsk_v4_fp8_partial_kernel`

A snapshot of which weapons the kernel currently uses, based on local
profiling of the DSv4-Pro sparse FP8 decode kernel on MI350X. The state
here applies to commit `b76bafc59` (r047 v3 multi-config dispatcher) on the
`amd/deepseek_v4` branch.

| # | Weapon | Currently used | Next action |
|---|---|---|---|
| 1 | `global_load_lds_*` | Not verified | Dump amdgcn, grep, fix `T.copy` calls that lower to 2-step path |
| 2 | `T.Pipelined(num_stages>0)` | No (rejected by PipelineInjector) | Restructure loop body so stages have independent buffer access |
| 3 | Dual-issue | Compiler-automatic | Verify in amdgcn, no explicit action required |
| 4 | MFMA LDS-direct | Yes | Maintain |
| 5 | `mfma_scale_f32_*_f8f6f4` | No | Check TileLang `T.gemm` for FP8 `b_dtype` support |
| 6 | Wide wave (`threads=512`) | Yes | Maintain |
| 7 | 2:4 sparsity | No | Defer |
| 8 | LDS-atomic warp-spec | No | Not recommended |

Roughly, weapons 1 + 2 + 3 are the AMD analogue of NV's TMA + producer/
consumer + ILP. Weapons 1 and 3 are largely automatic if the kernel is
written well; Weapon 2 requires the loop body to be pipelining-friendly,
which the current kernel is not.

Weapon 5 (direct FP8 multiplication) is the largest single optimization not
yet attempted, but requires confirming TileLang support and probably some
operand-layout reorganization.

---

## Where this came from

These notes were assembled while investigating why the TileLang sparse MLA
decode kernel underperforms the Triton path by roughly 40% at user
`batch_size >= 256` on MI350X, even after the bs-adaptive `block_per_cu`
fix in `dpsk_v4_fp8_attention_fwd`. The hypothesis is that several of the
above weapons (especially 1, 2, and 5) are either disengaged or
sub-optimally engaged, and that engaging them is the AMD path to closing
the gap with the closed-source `flash_mla` library on Hopper/Blackwell.

Concrete optimization candidates derived from this analysis are tracked
separately and are not part of this document.
