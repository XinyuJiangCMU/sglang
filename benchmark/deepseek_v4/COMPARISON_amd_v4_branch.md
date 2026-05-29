# DSv4 MTP-on-ROCm fix: `main` vs `amd/deepseek_v4`

This note explains, with GitHub-API evidence, why the MTP HIP-fallback fix
(PR `XinyuJiangCMU/sglang#14`) targets **`main`** and not the
**`amd/deepseek_v4`** integration branch, and what the AMD team would need to
do to pick it up.

All data below was collected from the GitHub REST API against
`sgl-project/sglang` (and the fork `XinyuJiangCMU/sglang`) on 2026-05-29.
Every claim cites its endpoint so it can be re-verified.

## Branch HEADs (verified)

| ref | SHA | source |
|---|---|---|
| `sgl-project/sglang` `main` | `ec075d8bc5ff847f43637e9f6dd0cd03f962b024` | `GET /repos/sgl-project/sglang/branches/main` |
| `sgl-project/sglang` `amd/deepseek_v4` | `76209c2c3bb7e557ab502e64af789ca6f14bd292` (committed 2026-05-28T07:31:50Z) | `GET /repos/sgl-project/sglang/branches/amd/deepseek_v4` |
| `XinyuJiangCMU/sglang` functional-fix commit | `683778d9f81c7482032aa520cef1a3b2b713b6a7` | `GET /repos/XinyuJiangCMU/sglang/commits/683778d9f` |

> The PR branch tip (`dev/dsv4-pro-mtp-rocm-20260529`) advances past
> `683778d9f` as documentation commits — including this file — are pushed, so
> the live branch HEAD and the live PR-level diff counts are larger than the
> numbers below. All numbers in sections 1–2 are anchored to the immutable
> functional-fix commit `683778d9f` (the commit that carries the source
> change); for the live PR-level totals see PR `XinyuJiangCMU/sglang#14`.

## 1. What this PR actually changes (functional-fix delta vs base = fork `main`)

Measured at the functional-fix commit `683778d9f`
(`GET /repos/sgl-project/sglang/compare/ec075d8bc5ff847f43637e9f6dd0cd03f962b024...XinyuJiangCMU:683778d9f81c7482032aa520cef1a3b2b713b6a7`):
4 files changed, +216 / -21.

| status | +/- | file |
|---|---|---|
| added | +62 / 0 | `benchmark/deepseek_v4/NOTES_mtp_rocm.md` |
| added | +30 / 0 | `benchmark/deepseek_v4/run_dsv4_pro_mtp_rocm.sh` |
| modified | +48 / -21 | `python/sglang/jit_kernel/dsv4/elementwise.py` |
| added | +76 / 0 | `test/srt/test_fused_q_indexer_rope_hadamard_quant_hip.py` |

The only source change is the HIP-gated decomposition inside
`jit_kernel/dsv4/elementwise.py`; the rest is docs/tests/launch script.

This comparison doc (`benchmark/deepseek_v4/COMPARISON_amd_v4_branch.md`) is
committed on the same branch on top of `683778d9f`, so the live PR `#14`
diff reports more files / additions than the table above — that extra delta
is documentation only and adds no source change.

## 2. PR head vs `amd/deepseek_v4` (divergence)

`GET /repos/sgl-project/sglang/compare/76209c2c3bb7e557ab502e64af789ca6f14bd292...XinyuJiangCMU:683778d9f81c7482032aa520cef1a3b2b713b6a7`:

- `status: diverged`
- `ahead_by: 931`, `behind_by: 54`, `total_commits: 931`
- files in the API response: 300 (the compare endpoint caps its `files`
  array at 300; summed `additions >= 13334`, `deletions >= 3766` over that
  capped set — treat as a floor, not an exact diff stat).

This huge divergence is **not** the size of this fix. The PR branch was cut
from an older `main` snapshot, so almost all 931/54 commits are unrelated
`main`/`amd` history drift. The fix itself is the 4-file / +216/-21 delta in
section 1.

## 3. Key-file existence across branches (contents API HTTP status)

`GET /repos/sgl-project/sglang/contents/<path>?ref=<ref>` status codes:

| path | `main` | `amd/deepseek_v4` |
|---|---|---|
| `python/sglang/jit_kernel/dsv4/` (dir) | 200 | 404 |
| `python/sglang/jit_kernel/dsv4/elementwise.py` | 200 | 404 |
| `sgl-kernel/csrc/common_extension_rocm.cc` | 200 | 200 |
| `python/sglang/srt/layers/attention/dsa/deepseek_v4_backend_hip_radix.py` | 404 | 404 |
| `test/srt/test_fused_q_indexer_rope_hadamard_quant_hip.py` | 404 | 404 |

Notes:
- `deepseek_v4_backend_hip_radix.py` does **not** exist on `main`,
  `amd/deepseek_v4`, **or** the PR branch (all 404). It is not part of this
  PR and is not referenced by it.
- The test file is 404 on both upstream branches because it is new on the PR
  branch (it is in the 4-file diff above).

### `common_extension_rocm.cc` op-registration difference

Both branches ship this file, but the DSv4 fused ops are registered
differently:

- **`main`** (249 lines) registers, all with the `torch::kCUDA` dispatch key:
  - line 56-57: `dsv4_fused_q_norm_rope`
  - line 60-62: `dsv4_fused_k_norm_rope_flashmla`
  - line 65-67: `dsv4_fused_q_indexer_rope_hadamard_quant`
- **`amd/deepseek_v4`** (235 lines): grep for
  `dsv4|indexer|hadamard|norm_rope` returns **no matches** — none of these
  DSv4 fused ops are registered in the ROCm extension on the AMD branch.

The CUDA-only (`torch::kCUDA`) registration of
`dsv4_fused_q_indexer_rope_hadamard_quant` on `main` is exactly why ROCm needs
the Python decomposition fallback this PR adds.

## 4. Why this PR targets `main`, not `amd/deepseek_v4`

1. The op being patched (`fused_q_indexer_rope_hadamard_quant` in
   `jit_kernel/dsv4/elementwise.py`) only exists on `main`; the
   `jit_kernel/dsv4/` directory is 404 on `amd/deepseek_v4`.
2. `amd/deepseek_v4` lays out its DSv4 kernels completely differently
   (see section 5), so the patch does not apply there as-is.
3. Putting the fix on `main` means the next routine
   `amd/deepseek_v4 <- main` sync carries it automatically, with no manual
   porting.

## 5. Where the equivalent DSv4 code lives on `amd/deepseek_v4`

From the recursive tree of `amd/deepseek_v4`
(`GET /repos/sgl-project/sglang/git/trees/76209c2c...?recursive=1`,
`truncated: false`):

- **`amd/deepseek_v4` uses a JIT-CUDA-header layout** under
  `python/sglang/jit_kernel/csrc/deepseek_v4/`:
  `c128.cuh`, `c4.cuh`, `common.cuh`, `fused_norm_rope.cuh`, `hash_topk.cuh`,
  `hisparse_transfer.cuh`, `paged_mqa_metadata.cuh`, `rope.cuh`,
  `silu_and_mul_masked_post_quant.cuh`, `store.cuh`, `topk.cuh`, `topk_v2.cuh`
  — plus the Python entry point `python/sglang/jit_kernel/deepseek_v4.py`.
  Hadamard lives separately in
  `python/sglang/jit_kernel/csrc/fast-hadamard-transform/` and
  `python/sglang/jit_kernel/hadamard.py`.
- **`main` uses a Python-package layout** under
  `python/sglang/jit_kernel/dsv4/`:
  `__init__.py`, `attn.py`, `compress.py`, `compress_old.py`,
  `elementwise.py`, `gemm.py`, `hisparse.py`, `moe.py`, `topk.py`, `utils.py`.

There is no single `*indexer*` or `*hadamard*`-named DSv4 file on
`amd/deepseek_v4`; the closest functional counterpart is
`jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh` (Q/K norm+rope) combined
with the separate fast-hadamard-transform kernels. The act-quant + indexer
fusion that `main`'s `elementwise.py` expresses as a single op is structured
as distinct JIT headers on the AMD branch.

## 6. AMD team: how to pick this up

**Preferred (zero work):** let the next `amd/deepseek_v4 <- main` sync pull
the change in automatically. Because the fix is HIP-gated and the CUDA path is
untouched, it is a no-op on NVIDIA and only activates the fallback on ROCm.

**If a manual backport is ever needed** (e.g. before the next sync), it is a
path/structure translation, not a cherry-pick — `git cherry-pick
683778d9f` will not apply because `jit_kernel/dsv4/elementwise.py` does not
exist on `amd/deepseek_v4`. Files that would need to be touched on the AMD
branch:

1. `python/sglang/jit_kernel/deepseek_v4.py` — add a ROCm branch that
   decomposes the indexer-Q path into rope (`csrc/deepseek_v4/rope.cuh` /
   `fused_norm_rope.cuh`) + hadamard (`jit_kernel/hadamard.py`) + per-(token,
   head) fp8 act-quant, mirroring the logic in `main`'s
   `jit_kernel/dsv4/elementwise.py::fused_q_indexer_rope_hadamard_quant`.
2. (Optional) `sgl-kernel/csrc/common_extension_rocm.cc` — only if the AMD
   branch later prebuilds the op instead of JIT-compiling it; currently it
   registers none of the `dsv4_*` ops, so no change is required for the
   fallback path.
3. `test/srt/test_fused_q_indexer_rope_hadamard_quant_hip.py` — adapt the
   import to the AMD entry point (`jit_kernel.deepseek_v4`) instead of
   `jit_kernel.dsv4.elementwise`.

Given the layout divergence, landing on `main` and syncing forward is
strictly less work than backporting.
