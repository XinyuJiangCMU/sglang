# r050 VERDICT — weapon 1 root cause located in tilelang upstream

## Question

Does `dpsk_v4_fp8_partial_kernel` (tilelang) on gfx950 use the
`global_load_lds_*` / `buffer_load_dword ... lds` direct HBM→LDS path
(AMD weapon 1)? If not, why, and is it fixable?

## Answer

**NO** — tilelang's HIP code does NOT emit weapon 1 for the bulk loads
(N=8 / N=16 bytes per thread). Root cause located inside tilelang
upstream template `tl_templates/hip/copy.h`. **Fix path identified**:
extend the template's N=8 / N=16 specializations to use multi-dword LDS-
direct loads. Estimated tilelang patch: ~30 lines. Estimated DSv4-Pro
gain: 5-15% on `dpsk_v4_fp8_partial_kernel` (frees VGPRs → allows higher
CU occupancy at bs≥256).

## Evidence chain

### 1. amdgcn audit

```
global_load_lds_*: 0    ← weapon 1 NOT in compiled kernel
global_load_*:    106
buffer_load_*:      0
ds_write_*:        88
ds_read_*:        674
v_mfma_*:         256
```

### 2. tilelang HIP copy template — root cause

`/opt/tilelang/src/tl_templates/hip/copy.h` `cp_async_gs<N>`:

```cpp
if constexpr (N == 16) {
  *(uint4 *)lds_base_ptr = *(const uint4 *)global_base_ptr;     // BROKEN
} else if constexpr (N == 8) {
  *(uint2 *)lds_base_ptr = *(const uint2 *)global_base_ptr;     // BROKEN
} else if constexpr (N == 4) {
  async_buffer_load_dword_v(...);                               // OK (direct)
}
```

`async_buffer_load_dword_v` correctly emits `buffer_load_dword ... lds`.
But only specializes for 4 bytes. For 8/16 bytes (the common bulk loads),
falls through to regular pointer copy → HIP compiler breaks into
`global_load_dwordx{2,4} + s_waitcnt + ds_write_b{64,128}` (the 2-step).

## Fix (deferred to r051)

Add to `tl_templates/hip/copy.h`:

```cpp
CK_TILE_DEVICE void async_buffer_load_dwordx2_v(void *smem, int32x4_t rsrc, index_t voffset) {
  ...
  asm volatile("s_mov_b32 m0, %0; buffer_load_dwordx2 %1, %2, 0 offen lds;" ...);
}
CK_TILE_DEVICE void async_buffer_load_dwordx4_v(void *smem, int32x4_t rsrc, index_t voffset) {
  ...
  asm volatile("s_mov_b32 m0, %0; buffer_load_dwordx4 %1, %2, 0 offen lds;" ...);
}
```

Then update cp_async_gs<8>/<16> to call them. Plus same for
`cp_async_gs_conditional`.

Requires tilelang rebuild (~10-20 min) + correctness test.

## Why tilelang upstream missed this

- AMD-side focus is recent; gfx950 specifically added Apr 2026 by Thomas Wang
- Tilelang's primary perf path is NV Hopper/Blackwell
- N=4 single-dword path is "good enough" for many use cases

## r050 status: METHODOLOGY PASS

- Confirmed weapon 1 NOT in use (amdgcn-level evidence)
- Located exact root cause in tilelang upstream template
- Identified ~30-line fix
- Did NOT apply fix (out of scope: requires tilelang rebuild + careful
  validation)
- Queued for r051

## r051 recommendation

Locally patch `/opt/tilelang/src/tl_templates/hip/copy.h` with N=8/16
LDS-direct specializations, rebuild, microbench, validate amdgcn shows
`buffer_load_dwordx{2,4} ... lds`. Expected 5-15% speedup on dpsk_v4
partial kernel.
