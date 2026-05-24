"""FlyDSL-backed sparse FP8 MLA decode attention (AMD gfx950 only).

Status: WIP / experimental. The path exposed here today is the
**kgather-only** variant: it exercises a validated FlyDSL weapon-1
(`buffer_load_dwordx4 ... lds`) K-gather kernel against the live K cache
to ensure the FlyDSL toolchain functions under real server load, then
delegates the actual attention math to the production TileLang backend.

The standalone full-attention sub-kernel exists in
``benchmark/sparse_mla_decode_flydsl/`` but is NOT yet wired into the
request path because dual-cache, ``D_tail``, ``attn_sink``,
``topk_length``-aware online softmax, and Partial_O/Partial_LSE +
combine-kernel emission are not yet implemented. Wiring it in before
those gaps close would silently produce wrong outputs.

Hardware support:
  * AMD MI355X / gfx950 only. Other archs are rejected at capability check.

Required Python packages (lazily imported; missing packages produce a
clear error only when this backend is explicitly selected):
  * ``flydsl`` — kernel DSL (provides ``rocdl.buffer_load_to_lds``)
  * ``aiter.ops.flydsl.kernels.tensor_shim`` — typed GTensor wrapper

Environment flags (all default off; production-safe):
  * ``SGLANG_FLYDSL_EXERCISE=1``  — run the kgather kernel on the real
    K cache + ``extra_k_cache`` before delegating math. Adds latency
    (a kernel launch per layer per decode step), should NOT be enabled
    in production until perf is measured.
  * ``SGLANG_FLYDSL_DEBUG=1``     — log the first kgather failure (one-shot).
  * ``SGLANG_FLYDSL_DEBUG_SYNC=1``— after the kgather kernel, call
    ``torch.cuda.synchronize()`` for deterministic timing. Off by default
    so async overlap is preserved on the request path.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch

__all__ = [
    "is_flydsl_kgather_available",
    "dpsk_v4_fp8_attention_fwd_flydsl_kgather_only",
]


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def is_flydsl_kgather_available() -> Tuple[bool, str]:
    """Cheap, side-effect-free probe.

    Returns ``(ok, reason)``. ``reason`` is an empty string on success and
    a one-line human-readable diagnostic on failure.
    """
    # Quick ROCm + arch check — no flydsl import yet (heavy).
    try:
        from sglang.srt.utils import is_hip
    except ImportError:
        return False, "sglang.srt.utils.is_hip unavailable"
    if not is_hip():
        return False, "not running on AMD/ROCm"
    if not torch.cuda.is_available():
        return False, "CUDA/HIP device not available"
    # Probe arch — flydsl-supported gfx95 only.
    try:
        props = torch.cuda.get_device_properties(0)
        gcn = getattr(props, "gcnArchName", "") or ""
    except Exception as e:  # pragma: no cover - defensive
        return False, f"could not query device properties: {e}"
    if not gcn.startswith("gfx95"):
        return False, f"unsupported arch {gcn!r} (requires gfx95*)"
    # Lazy import flydsl + aiter shim. Only import paths, not heavy modules,
    # to keep the probe cheap.
    try:
        import flydsl  # noqa: F401
    except ImportError as e:
        return False, f"flydsl package not importable: {e}"
    try:
        # tensor_shim isn't a normal top-level import — it lives next to
        # aiter's flydsl kernels. We require the canonical aiter path; we
        # don't vendor a fallback shim inside sglang so users always see a
        # clear error if aiter is missing.
        from aiter.ops.flydsl.kernels.tensor_shim import GTensor  # noqa: F401
    except ImportError as e:
        return False, f"aiter FlyDSL tensor_shim missing: {e}"
    return True, ""


# ---------------------------------------------------------------------------
# Kernel build (lazy, cached)
# ---------------------------------------------------------------------------

# Kernel cache keyed by (ROW_BYTES, BYTES_PER_LOAD) so the same compiled
# kernel is reused across layers/decode steps. Guarded by a lock so two
# decode workers cannot race on first compile (kernel build takes seconds;
# duplicate compiles would block the request path).
_KGATHER_KERNEL_CACHE: dict[Tuple[int, int], Any] = {}
_KGATHER_KERNEL_LOCK = threading.Lock()

# Scratch buffer cache so the kgather exercise doesn't allocate every call.
# Keyed by (device_index, grid_x, packed_w_bytes).
_KGATHER_SCRATCH_CACHE: dict[Tuple[int, int, int], torch.Tensor] = {}

# Single-shot logging — capability/feature failures are logged once per
# unique reason so a misconfigured deploy doesn't spam the request path.
_LOGGED_FEATURE_REJECTIONS: set[str] = set()
_LOGGED_EXERCISE_FAILURE: dict[str, bool] = {"once": False}


def _build_kgather_kernel(row_bytes: int, bytes_per_load: int):
    """Compile a FlyDSL weapon-1 K-gather kernel.

    Emitted ISA per thread (gfx950): one ``buffer_load_dwordx4 ... lds``
    (HBM → LDS direct, weapon-1), one ``ds_read_b128`` (LDS → reg), one
    ``buffer_store_dwordx4`` (reg → HBM scratch). Verified byte-exact vs
    ``torch.gather`` in ``test/srt/test_flydsl_kgather.py``.
    """
    # Lazy imports — only paid when a kernel is actually compiled.
    import flydsl.compiler as flyc  # noqa: F401
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, rocdl
    from flydsl.expr.typing import T
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    # tensor_shim is provided by aiter. The capability check has already
    # confirmed availability — this import should not raise here.
    from aiter.ops.flydsl.kernels.tensor_shim import GTensor

    # Only the widths we'll ever use. buffer_load_to_lds supports
    # 1/2/4/8/12/16 byte widths but the AMDGPU LLVM backend has lowering
    # bugs on 8-byte loads from LDS at small alignments (see PR review
    # discussion); stick to 4 and 16.
    assert bytes_per_load in (4, 16), f"unsupported bytes_per_load={bytes_per_load}"
    assert row_bytes % bytes_per_load == 0, (
        f"row_bytes={row_bytes} not divisible by bytes_per_load={bytes_per_load}"
    )
    num_threads = row_bytes // bytes_per_load
    assert 1 <= num_threads <= 1024

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"sglang_flydsl_kgs_r{row_bytes}_b{bytes_per_load}",
    )
    lds_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_row_offset + max(row_bytes, 16)

    def _lds_ptr_ty():
        return ir.Type.parse("!llvm.ptr<3>")

    @flyc.kernel(name=f"sglang_flydsl_kgather_r{row_bytes}")
    def kgather_kernel(
        src_i8: fx.Tensor,
        row_byte_offsets: fx.Tensor,
        scratch_i8: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        wgid = fx.block_idx.x
        ro = GTensor(row_byte_offsets, dtype=T.i32, shape=(-1,))
        src = GTensor(src_i8, dtype=T.i8, shape=(-1,))
        scr_i32 = GTensor(scratch_i8, dtype=T.i32, shape=(-1,))

        thread_lds_byte = tid * fx.Int32(bytes_per_load) + fx.Int32(lds_row_offset)
        tlb_raw = (
            thread_lds_byte.value
            if hasattr(thread_lds_byte, "value")
            else thread_lds_byte
        )
        lds_byte_i64 = arith.extui(T.i64, tlb_raw)
        lds_ptr = _llvm.IntToPtrOp(_lds_ptr_ty(), lds_byte_i64).result

        # voffset = row_byte_offsets[wgid] + tid * bytes_per_load
        row_base = ro.load(wgid, vec_size=1)
        rb = row_base.value if hasattr(row_base, "value") else row_base
        tir = tid * fx.Int32(bytes_per_load)
        tir_raw = tir.value if hasattr(tir, "value") else tir
        voffset = arith.addi(rb, tir_raw)

        # WEAPON 1: HBM → LDS direct, no VGPR transit.
        rocdl.buffer_load_to_lds(
            rsrc=src.rsrc,
            lds_ptr=lds_ptr,
            voffset=voffset,
            size_bytes=bytes_per_load,
            soffset=0,
            offset=0,
        )
        rocdl.barrier()

        # LDS → register → HBM scratch. Cast to i32 vec because the AMDGPU
        # buffer-store path hits a "split result" LLVM bug at <16xi8>.
        words_per_load = bytes_per_load // 4
        lds_view = SmemPtr(
            allocator.get_base(),
            lds_row_offset,
            T.i32,
            shape=(max(row_bytes // 4, 4),),
        )
        lds_mr = lds_view.get()
        vt = T.vec(words_per_load, T.i32)
        lds_idx = (tid * fx.Int32(words_per_load)).value
        lds_idx_ix = arith.index_cast(T.index, lds_idx)
        v = _vector.load(vt, lds_mr, [lds_idx_ix])

        out_word_off = (
            wgid * fx.Int32(row_bytes // 4) + tid * fx.Int32(words_per_load)
        )
        scr_i32.store(out_word_off, v, vec_size=words_per_load)

    @flyc.jit
    def launch(src_i8, row_byte_offsets, scratch_i8, grid_x: fx.Int32):
        # allocator.finalize() must be emitted inside the gpu module body.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kgather_kernel(src_i8, row_byte_offsets, scratch_i8).launch(
            grid=(grid_x, 1, 1),
            block=(num_threads, 1, 1),
        )

    return launch


def _get_kgather_kernel(row_bytes: int, bytes_per_load: int):
    """Return a compiled kgather kernel, building (and caching) on first call.

    Thread-safe: a coarse lock around the build serializes first-call
    compilation across decode workers within a single process.
    """
    key = (row_bytes, bytes_per_load)
    cached = _KGATHER_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _KGATHER_KERNEL_LOCK:
        # Re-check after lock acquisition.
        cached = _KGATHER_KERNEL_CACHE.get(key)
        if cached is None:
            cached = _build_kgather_kernel(row_bytes, bytes_per_load)
            _KGATHER_KERNEL_CACHE[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Kgather exercise (per-cache, single-shot per decode step)
# ---------------------------------------------------------------------------

# DSv4 FP8 K-cache row layout (per token):
#   bytes [0 : PACKED_W_BYTES)        - packed FP8 + BF16 tail (binary blob)
#   bytes [PACKED_W_BYTES : PACKED_W_FULL_BYTES) - per-NOPE_TILE scale bytes + pad
# See `tilelang_kernel.py:1628-1631` (`PACKED_W = dim + 2 * tail_dim = 576`,
# `SCALE_W = 8` → `PACKED_W_FULL = 584`).
_PACKED_W_BYTES = 576
_PACKED_W_FULL_BYTES = 584


def _check_kgather_supported(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[bool, str]:
    """Soft capability check — no exceptions raised on the request path.

    Returns ``(ok, reason)``. Callers should log_once on the rejection
    reason and skip the exercise (the underlying TileLang math path is
    unaffected).
    """
    if k_cache is None or indices is None:
        return False, "k_cache or indices is None"
    if k_cache.dtype.itemsize != 1:
        return False, f"k_cache dtype {k_cache.dtype} is not 1-byte (FP8)"
    if k_cache.ndim != 4:
        return False, f"k_cache must be 4D, got shape={tuple(k_cache.shape)}"
    nb, bs_kv, h_kv, packed_w_full = k_cache.shape
    if h_kv != 1:
        return False, f"H_KV={h_kv} unsupported (kgather assumes MQA layout)"
    if packed_w_full != _PACKED_W_FULL_BYTES:
        return False, (
            f"K cache packed width {packed_w_full} != expected "
            f"{_PACKED_W_FULL_BYTES} (PACKED_W={_PACKED_W_BYTES} + SCALE_W=8)"
        )
    if not k_cache.is_contiguous():
        return False, "k_cache is non-contiguous (would require 200+ MB copy)"
    if indices.ndim != 3:
        return False, f"indices must be 3D, got shape={tuple(indices.shape)}"
    bs, s_q, topk = indices.shape
    if s_q != 1:
        return False, f"decode expects S_Q=1, got S_Q={s_q}"
    if indices.dtype != torch.int32:
        return False, f"indices dtype must be int32, got {indices.dtype}"
    if bs * topk == 0:
        return False, f"empty workload (BS={bs}, TOPK={topk})"
    return True, ""


def _exercise_kgather_on_real(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    do_sync: bool = False,
) -> None:
    """Run the FlyDSL weapon-1 K-gather kernel against the live K cache.

    Output is discarded — this is purely a smoke run that proves the FlyDSL
    pipeline functions under real server load (CUDA graph capture / replay,
    concurrent decode requests, real HBM pressure).

    Layout is assumed to have been validated via ``_check_kgather_supported``
    before calling — this function will raise if it's wrong, which the
    request-path wrapper catches.
    """
    nb, bs_kv, _, packed_w_full = k_cache.shape

    # Layout note: the K cache is laid out as
    #   (NB, BS_KV, H_KV=1, packed_w_full=584)
    # where each token occupies *packed_w_full* contiguous bytes (576-byte
    # packed FP8 + 8-byte scale region). The in-block stride between
    # adjacent tokens is `packed_w_full`, NOT _PACKED_W_BYTES — using the
    # latter would skip the 8-byte scale region of the previous token and
    # read shifted (incorrect) data for in_block > 0.
    idx_flat = indices.reshape(-1)
    # Clip invalid indices to a safe in-range value — captured pickles
    # have sentinel values (negative or >= NB*BS_KV) for padding slots.
    # The math path masks these via `topk_length`, but our buffer load
    # needs a valid offset to avoid OOB reads.
    max_token = nb * bs_kv - 1
    idx_clipped = torch.clamp(idx_flat, min=0, max=max_token)
    block_id = idx_clipped // bs_kv
    in_block = idx_clipped % bs_kv
    row_byte_offsets = (
        block_id.to(torch.int64) * (bs_kv * packed_w_full)
        + in_block.to(torch.int64) * packed_w_full
    ).to(torch.int32).contiguous()

    grid_x = row_byte_offsets.numel()
    k_cache_i8 = k_cache.view(torch.int8).reshape(-1)

    # Scratch buffer cache by (device, grid_x, width) to avoid per-call
    # allocation. The buffer is read-only after the kernel (we discard
    # the output), so it's safe to reuse across calls.
    scratch_key = (k_cache.device.index or 0, grid_x, _PACKED_W_BYTES)
    scratch = _KGATHER_SCRATCH_CACHE.get(scratch_key)
    if scratch is None:
        scratch = torch.empty(
            grid_x * _PACKED_W_BYTES,
            dtype=torch.int8,
            device=k_cache.device,
        )
        _KGATHER_SCRATCH_CACHE[scratch_key] = scratch

    launch = _get_kgather_kernel(_PACKED_W_BYTES, 16)
    launch(k_cache_i8, row_byte_offsets, scratch, grid_x)

    if do_sync:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Backend entry point
# ---------------------------------------------------------------------------

def _log_once(reason: str, *, debug: bool) -> None:
    """One-shot stderr log per unique reason. No-op outside debug mode."""
    if not debug:
        return
    if reason in _LOGGED_FEATURE_REJECTIONS:
        return
    _LOGGED_FEATURE_REJECTIONS.add(reason)
    import sys

    print(f"[flydsl_kgather_only] {reason}", file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _env_flags() -> Tuple[bool, bool, bool]:
    """Snapshot env flags once per process.

    Returns ``(do_exercise, debug, do_sync)``. Cached so we don't pay
    ``os.environ`` lookup on every layer's attention call.
    """
    return (
        os.environ.get("SGLANG_FLYDSL_EXERCISE", "0") == "1",
        os.environ.get("SGLANG_FLYDSL_DEBUG", "0") == "1",
        os.environ.get("SGLANG_FLYDSL_DEBUG_SYNC", "0") == "1",
    )


def dpsk_v4_fp8_attention_fwd_flydsl_kgather_only(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL kgather-only backend.

    When ``SGLANG_FLYDSL_EXERCISE=1`` and the cache layout / indices pass
    ``_check_kgather_supported``, runs the FlyDSL weapon-1 K-gather kernel
    against the primary K cache (and ``extra_k_cache`` if present) as a
    smoke exercise, then **delegates all attention math to the production
    TileLang backend**.

    The model's numerical output is identical to running with
    ``SGLANG_HACK_FLASHMLA_BACKEND=tilelang`` directly. This entry point
    is intentionally not a full FlyDSL kernel — see the module docstring
    for the missing pieces.
    """
    do_exercise, debug, do_sync = _env_flags()

    if do_exercise:
        ok, reason = _check_kgather_supported(k_cache, indices)
        if ok:
            try:
                _exercise_kgather_on_real(k_cache, indices, do_sync=do_sync)
            except Exception as e:  # pragma: no cover - defensive
                if debug and not _LOGGED_EXERCISE_FAILURE["once"]:
                    import sys
                    import traceback

                    print(
                        f"[flydsl_kgather_only] primary exercise failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc()
                    _LOGGED_EXERCISE_FAILURE["once"] = True
        else:
            _log_once(f"primary kgather skipped: {reason}", debug=debug)

        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            ok2, reason2 = _check_kgather_supported(
                extra_k_cache, extra_indices_in_kvcache
            )
            if ok2:
                try:
                    _exercise_kgather_on_real(
                        extra_k_cache,
                        extra_indices_in_kvcache,
                        do_sync=do_sync,
                    )
                except Exception as e:  # pragma: no cover - defensive
                    if debug and not _LOGGED_EXERCISE_FAILURE["once"]:
                        import sys
                        import traceback

                        print(
                            f"[flydsl_kgather_only] extra exercise failed: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        traceback.print_exc()
                        _LOGGED_EXERCISE_FAILURE["once"] = True
            else:
                _log_once(f"extra kgather skipped: {reason2}", debug=debug)

    # Math delegates to TileLang (production-validated). The kgather
    # exercise above does not change the math; we always return TileLang's
    # output unchanged.
    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        dpsk_v4_fp8_attention_fwd,
    )

    return dpsk_v4_fp8_attention_fwd(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
