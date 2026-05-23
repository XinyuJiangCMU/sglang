# r050 AMDGCN audit — weapon 1 (`global_load_lds_*`) NOT in use

**TL;DR**: confirmed via objdump of compiled `dpsk_v4_fp8_partial_kernel`:
**zero `global_load_lds_*` instructions emitted**. Tilelang's T.copy lowering
emits the 2-step `global_load + ds_write` pattern instead, wasting VGPR on
intermediate values. This is one of the candidate reasons tilelang under-
performs triton at bs≥256 on gfx950.

## How verified

1. JIT cache directory: `/root/.tilelang/cache/` (env: `TILELANG_CACHE_DIR =
   ~/.tilelang/cache`)
2. Identified `dpsk_v4_fp8_partial_kernel` cache entry by HIP signature
   (`__launch_bounds__(512)` + `Indices_1, Indices_2, K_combined_1, ...`):
   - `cache/7dd66bb19821cb957a8b9d9d939ca25fa9b5a23361c25fe8a8883d1f7a1a7a6d/`
   - device_kernel.cu: 45752 bytes
   - kernel_lib.so: HIP fatbinary
3. Extracted GPU device code via `dd` (the .so has both host x86 + amdgcn
   bundled; `roc-obj-ls` reports `hipv4-amdgcn-amd-amdhsa--gfx950` at
   offset 8192, size 67336):
   ```bash
   dd if=kernel_lib.so of=gpu_bundle.bin bs=1 skip=8192 count=67336
   file gpu_bundle.bin  # → ELF 64-bit LSB shared object, *unknown arch 0xe0* (amdgcn)
   ```
4. Disassembled with `/opt/rocm-7.2.0/lib/llvm/bin/llvm-objdump -d gpu_bundle.bin`
   (the system `/usr/lib/llvm-18/bin/llvm-objdump` segfaults; ROCm's version
   handles `elf64-amdgpu` cleanly).
5. Output: `kernel.amdgcn`, 9921 lines.

## Instruction-family counts (kernel.amdgcn)

| family | count | what it means |
|---|---:|---|
| **`global_load_lds_*`** | **0** | weapon 1 — direct HBM→LDS — NOT used |
| `global_load_*` | 106 | regular HBM→VGPR loads |
| `buffer_load_*` | 0 | alternative HBM→VGPR via buffer descriptor (not used either) |
| `ds_write_*` | 88 | VGPR→LDS writes (the 2nd step of the 2-step pattern) |
| `ds_read_*` | 674 | LDS→VGPR reads (downstream consumers) |
| `v_mfma_*` | 256 | matrix core instructions (real compute) |
| `s_load_*` | 10 | scalar loads (kernel params etc.) |

## Sample 2-step pattern (real lines)

The pattern we see throughout the K load region (around `// offset 0x2088`):

```
global_load_dwordx2 v[92:93], v[2:3], off                  // 8 bytes loaded HBM→VGPR
global_load_dwordx2 v[94:95], v[2:3], off offset:32        // VGPR pressure +2 per load
global_load_dwordx2 v[96:97], v[2:3], off offset:64
global_load_dwordx2 v[98:99], v[2:3], off offset:96
... (continues with offsets 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448)
```

15 consecutive `global_load_dwordx2` instructions in a single burst, each
loading 8 bytes (2 dwords) from HBM into 2 VGPRs. Then ~15 `ds_write_b64`
instructions follow (in another part of the kernel — they're separated
to allow `s_waitcnt vmcnt(0)` between).

**Total VGPR pressure from this load batch alone**: 15 loads × 2 VGPRs =
30 VGPRs held live across the wait + ds_write region. Plus other live
values. With 8 waves/block at gfx950, VGPR budget per wave = 512 fp32-eq
slots; 30 VGPRs * 8 waves = ~240 VGPR-thread-slots just for K data in
transit.

**Weapon 1 alternative** would be `global_load_lds_b64 v[2:3], off, offset:N`
which writes HBM→LDS without going through VGPR. Same load count, but
zero VGPR pressure. The savings allow more concurrent blocks per CU
(higher occupancy).

## Why tilelang didn't emit it

Looking at the HIP source (device_kernel.cu line 86, 96):
```cpp
condval_1 = K_combined_1[((((((int64_t)page) & 127) * 144) + ...) % 144)];
// ... downstream:
K_packed_shared[bi_i, w_i] = ...condval_1...;
```

This is **scalar element-wise indexing**: load `K_combined_1[some_complex_index]`
into a local scalar (forces VGPR), then assign to `K_packed_shared`. The
TVM/tilelang lowering pipeline doesn't pattern-match this to
`global_load_lds_*` because the latter requires a recognizable
contiguous-block copy with proper alignment.

In tilelang_kernel.py source the load is written as:
```python
for bi_i, w_i in T.Parallel(BI, PACKED_W4):
    page = page_idx_shared[bi_i]
    block_id = page // BS_KV_1
    t_in_block = page % BS_KV_1
    K_packed_shared[bi_i, w_i] = K_combined_1[
        block_id, t_in_block * PACKED_W4 + w_i
    ]
```

`T.Parallel` over `(BI, PACKED_W4)` with element-wise assignment goes
through the per-element load path. To emit `global_load_lds_b128` the
write pattern would need to be a `T.copy(src_slice, dst_slice)` with
contiguous source and dst that the lowering can fuse.

**However**: the source pattern uses INDIRECT addressing
(`page_idx_shared[bi_i]` → `block_id`). Each row of `K_packed_shared`
loads from a DIFFERENT page. This is a **gather**, not a contiguous
block-load. `global_load_lds_*` works on contiguous regions; gathers
inherently need per-element addressing.

**This is the killer constraint**: the kernel's design uses
`page_idx_shared[bi_i]` as an indirection, so the loads are
fundamentally gather operations. Weapon 1 (`global_load_lds_*`) only
helps for contiguous block loads — not gather. To use weapon 1, we'd
need to restructure the kernel so the K_packed loads are contiguous
(maybe load multiple pages' contributions for a contiguous chunk,
then permute in LDS).

## Tentative verdict

The "weapon 1 not used" finding IS REAL but the FIX is non-trivial.
The kernel's gather-style K-page access pattern fundamentally prevents
weapon 1 emission. A trivial T.copy rewrite won't help because the
addressing is indirect (per-row page index).

**Two possible paths**:

**(a) Trivial — replace T.Parallel with T.copy where possible**: only
the `Q[b_i, s_i, H0:H1, :D]` load is contiguous (no per-row gather).
Trying T.copy there might emit weapon 1 for Q, but Q is small (one-time
load per kernel) so impact is minimal.

**(b) Restructure K load to be contiguous (big rewrite)**:
- Phase 1: precompute permuted K_combined contiguous chunks ahead of time
- Phase 2: kernel does block-style `T.copy(K_continuous[start:end], K_packed_shared)`
- This is significant kernel restructure + extra prefill work.

## Next: try path (a) on Q load, measure VGPR pressure delta

The Q load is at the kernel's top (line ~280-300 in HIP source). Replace:
```python
T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
```
…with the explicit form that already exists, but verify the resulting
amdgcn emits `global_load_lds_*`. If yes, document weapon 1 IS
emittable by tilelang for contiguous Q (just not for the gather K),
and write up the limitation in the wiki.

If no — even Q can't trigger weapon 1 — tilelang lowering doesn't emit
it on this gfx950 target. Document as a tilelang limitation;
optimization requires kernel-level workaround.
