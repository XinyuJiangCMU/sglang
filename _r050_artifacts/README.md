# r050 — Weapon 1 (`global_load_lds_*`) audit

Investigation: does tilelang emit `global_load_lds_*` direct HBM→LDS for
`dpsk_v4_fp8_partial_kernel` on gfx950?

## Files

- `AMDGCN_AUDIT.md` — full audit writeup
- `kernel.amdgcn` — llvm-objdump disassembly of the JIT'd kernel

## TL;DR

**Weapon 1 NOT emitted**. 0 `global_load_lds_*` vs 106 `global_load_*` +
88 `ds_write_*` (the 2-step pattern). Root cause: kernel uses gather-style
indirect addressing (`K_combined_1[page_idx_shared[bi_i], ...]`) which
TVM/tilelang lowering can't pattern-match to direct HBM→LDS (which
requires contiguous source). See AMDGCN_AUDIT.md for details + next steps.
