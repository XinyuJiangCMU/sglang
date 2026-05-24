"""r051 Round 15: BF16 attention kernel with SPARSE K-gather (indirect indexing).

Adds the sparse K-cache lookup pattern from dpsk_v4_partial_kernel:
  indices[batch, n_tile_pos] → token_id
  k[batch, n] = k_cache[token_id // BS_KV, token_id % BS_KV, :]

This matches the memory-access pattern of tilelang/triton but in BF16
(skipping FP8 dequant for now; that's round 16).

Validation: compare to torch.softmax(Q @ K_gathered.T) @ V_gathered.
"""
import os
import sys
import time
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith, vector as _fvec, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor


def build_sparse_attn():
    arch = get_hip_arch()

    LDS_S_F32_BYTES  = 16 * 64 * 4
    LDS_S_BF16_BYTES = 16 * 64 * 2

    allocator = SmemAllocator(None, arch=arch, global_sym_name="attn_sparse_smem_v1")
    lds_s_f32_off  = allocator._align(allocator.ptr, 16);  allocator.ptr = lds_s_f32_off + LDS_S_F32_BYTES
    lds_s_bf16_off = allocator._align(allocator.ptr, 16);  allocator.ptr = lds_s_bf16_off + LDS_S_BF16_BYTES

    M_TILE, N_TILE, K_MFMA = 16, 16, 32
    BI, D, D_V = 64, 448, 448
    N_SUBTILES = BI // N_TILE      # 4
    K_TILES = D // K_MFMA           # 14
    K_V_TILES_16 = BI // 16         # 4
    OUT_N_TILES = D_V // N_TILE     # 28
    BS_KV = 128                     # K cache block size

    @flyc.kernel(name="attn_sparse_kernel")
    def kernel(
        q_tensor: fx.Tensor,         # bf16 [BS * M_HEADS * D]
        k_cache: fx.Tensor,          # bf16 [NB * BS_KV * D]  (flat row-major)
        indices: fx.Tensor,          # i32  [BS * BI]   (gives token_id per (b, n))
        v_cache: fx.Tensor,          # bf16 [NB * BS_KV * D_V]
        o_tensor: fx.Tensor,         # f32  [BS * M_HEADS * D_V]
    ):
        tid = fx.thread_idx.x
        m_block = fx.block_idx.x
        b_block = fx.block_idx.y

        q_  = GTensor(q_tensor,  dtype=T.bf16, shape=(-1,))
        kc_ = GTensor(k_cache,   dtype=T.bf16, shape=(-1,))
        idx_= GTensor(indices,   dtype=T.i32,  shape=(-1,))
        vc_ = GTensor(v_cache,   dtype=T.bf16, shape=(-1,))
        o_  = GTensor(o_tensor,  dtype=T.f32,  shape=(-1,))

        # Lane mapping
        m_in_tile = tid % fx.Int32(16)
        k_lo_32 = (tid // fx.Int32(16)) * fx.Int32(8)
        global_m = b_block * fx.Int32(128) + m_block * fx.Int32(M_TILE) + m_in_tile

        # ---- Step 1: QK gemm with sparse K-gather ----
        f32x4_t = T.vec(4, T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)
        accs = [_vector.broadcast(f32x4_t, zero_f32) for _ in range_constexpr(N_SUBTILES)]

        for n_tile in range_constexpr(N_SUBTILES):
            n_in_tile = tid % fx.Int32(16)
            # Sparse index lookup: token_id = indices[b_block, n_tile*16 + n_in_tile]
            n_global = fx.Int32(n_tile * N_TILE) + n_in_tile
            idx_off = b_block * fx.Int32(BI) + n_global
            token_id = idx_.load(idx_off, vec_size=1)
            tid_raw = token_id.value if hasattr(token_id, "value") else token_id
            # block_id = token_id / BS_KV, in_block = token_id % BS_KV
            bs_kv_c = arith.constant(BS_KV, type=T.i32)
            block_id = arith.divui(tid_raw, bs_kv_c)
            in_block = arith.remui(tid_raw, bs_kv_c)
            # k_row_byte_base = (block_id * BS_KV + in_block) * D  (element offset)
            k_row_elem_base = arith.addi(
                arith.muli(block_id, bs_kv_c),
                in_block,
            )
            k_row_off_base = arith.muli(k_row_elem_base, arith.constant(D, type=T.i32))

            for k_tile in range_constexpr(K_TILES):
                k_off_within_row = fx.Int32(k_tile * K_MFMA) + k_lo_32
                k_off_within_row_raw = k_off_within_row.value if hasattr(k_off_within_row, "value") else k_off_within_row
                k_off = arith.addi(k_row_off_base, k_off_within_row_raw)

                a_off = global_m * fx.Int32(D) + k_off_within_row + fx.Int32(0)
                # Wait — A (Q) doesn't use sparse, so its offset is just global_m * D + k_off_within_row.
                # That doesn't depend on K cache structure. Use existing logic.
                q_a_off = global_m * fx.Int32(D) + fx.Int32(k_tile * K_MFMA) + k_lo_32
                a_frag = q_.load(q_a_off, vec_size=8)
                # K (sparse): load 8 bf16 from k_cache at byte offset k_off
                b_frag = kc_.load(k_off, vec_size=8)

                accs[n_tile] = rocdl.mfma_f32_16x16x32_bf16(
                    f32x4_t, [a_frag, b_frag, accs[n_tile], 0, 0, 0]
                )

        # ---- Step 2: spill to LDS in (m, n_global) layout ----
        m_lo_c = (tid // fx.Int32(16)) * fx.Int32(4)
        n_c = tid % fx.Int32(16)
        s_lds = SmemPtr(allocator.get_base(), lds_s_f32_off, T.f32, shape=(M_TILE*BI,))
        s_lds_mr = s_lds.get()
        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                scalar = _vector.extract(accs[n_tile], static_position=[el], dynamic_position=[])
                ix = (m_lo_c + fx.Int32(el)) * fx.Int32(BI) + fx.Int32(n_tile*N_TILE) + n_c
                ix_raw = ix.value if hasattr(ix, "value") else ix
                ix_ix = arith.index_cast(T.index, ix_raw)
                _memref.store(scalar, s_lds_mr, [ix_ix])
        rocdl.barrier()

        # ---- Step 3: softmax row-wise over N=BI ----
        is_worker = arith.cmpi(
            _mlir_arith.CmpIPredicate.ult,
            tid.value if hasattr(tid, "value") else tid,
            arith.constant(16, type=T.i32),
        )
        log2e = arith.constant(1.4426950408889634, type=T.f32)
        c_neg_inf = arith.constant(-1e30, type=T.f32)
        c_zero = arith.constant(0.0, type=T.f32)
        if_op = _scf.IfOp(is_worker, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            row_base = tid * fx.Int32(BI)
            rb = row_base.value if hasattr(row_base, "value") else row_base
            row_vals = []
            cur_max = c_neg_inf
            for j in range_constexpr(BI):
                idx = arith.addi(rb, arith.constant(j, type=T.i32))
                v = _memref.load(s_lds_mr, [arith.index_cast(T.index, idx)])
                row_vals.append(v)
                cur_max = arith.maxnumf(cur_max, v)
            exp_vals = []
            cur_sum = c_zero
            for j in range_constexpr(BI):
                d = arith.subf(row_vals[j], cur_max)
                e = rocdl.exp2(T.f32, arith.mulf(d, log2e))
                exp_vals.append(e)
                cur_sum = arith.addf(cur_sum, e)
            for j in range_constexpr(BI):
                s = arith.divf(exp_vals[j], cur_sum)
                idx = arith.addi(rb, arith.constant(j, type=T.i32))
                _memref.store(s, s_lds_mr, [arith.index_cast(T.index, idx)])
            _scf.yield_([])
        rocdl.barrier()

        # ---- Step 4: cast S f32 → bf16 ----
        s_bf16_lds = SmemPtr(allocator.get_base(), lds_s_bf16_off, T.bf16, shape=(M_TILE*BI,))
        s_bf16_mr = s_bf16_lds.get()
        for n_tile in range_constexpr(N_SUBTILES):
            for el in range_constexpr(4):
                ix = (m_lo_c + fx.Int32(el)) * fx.Int32(BI) + fx.Int32(n_tile*N_TILE) + n_c
                ix_raw = ix.value if hasattr(ix, "value") else ix
                ix_ix = arith.index_cast(T.index, ix_raw)
                f32_v = _memref.load(s_lds_mr, [ix_ix])
                _memref.store(arith.truncf(T.bf16, f32_v), s_bf16_mr, [ix_ix])
        rocdl.barrier()

        # ---- Step 5: S @ V with SPARSE V gather ----
        i16x4_t = T.vec(4, T.i16)
        bf16x4_t = T.vec(4, T.bf16)
        m_a = tid % fx.Int32(16)
        n_b = tid % fx.Int32(16)
        k_lo_16 = (tid // fx.Int32(16)) * fx.Int32(4)
        m_lo_o = (tid // fx.Int32(16)) * fx.Int32(4)
        n_o = tid % fx.Int32(16)

        for out_n in range_constexpr(OUT_N_TILES):
            o_acc = _vector.broadcast(f32x4_t, zero_f32)
            for k_tile_v in range_constexpr(K_V_TILES_16):
                # A (S) frag
                a_elems = []
                for el in range_constexpr(4):
                    col = fx.Int32(k_tile_v * 16) + k_lo_16 + fx.Int32(el)
                    ix = m_a * fx.Int32(BI) + col
                    ix_raw = ix.value if hasattr(ix, "value") else ix
                    a_elems.append(_memref.load(s_bf16_mr, [arith.index_cast(T.index, ix_raw)]))
                a_frag = _vector.from_elements(bf16x4_t, a_elems)
                a_i16 = _vector.bitcast(i16x4_t, a_frag)

                # B (V) frag — SPARSE: look up token for each k_lo row
                v_elems = []
                for el in range_constexpr(4):
                    k_v_pos = fx.Int32(k_tile_v * 16) + k_lo_16 + fx.Int32(el)
                    # token = indices[b_block, k_v_pos]
                    idx_off_v = b_block * fx.Int32(BI) + k_v_pos
                    tok = idx_.load(idx_off_v, vec_size=1)
                    tok_raw = tok.value if hasattr(tok, "value") else tok
                    bs_kv_c2 = arith.constant(BS_KV, type=T.i32)
                    blk = arith.divui(tok_raw, bs_kv_c2)
                    inb = arith.remui(tok_raw, bs_kv_c2)
                    row_elem = arith.addi(arith.muli(blk, bs_kv_c2), inb)
                    v_row_base = arith.muli(row_elem, arith.constant(D_V, type=T.i32))
                    n_col = fx.Int32(out_n * 16) + n_b
                    n_col_raw = n_col.value if hasattr(n_col, "value") else n_col
                    v_off = arith.addi(v_row_base, n_col_raw)
                    v_elems.append(vc_.load(v_off, vec_size=1))
                v_frag = _vector.from_elements(bf16x4_t, v_elems)
                v_i16 = _vector.bitcast(i16x4_t, v_frag)

                o_acc = rocdl.mfma_f32_16x16x16bf16_1k(
                    f32x4_t, [a_i16, v_i16, o_acc, 0, 0, 0]
                )

            for el in range_constexpr(4):
                scalar = _vector.extract(o_acc, static_position=[el], dynamic_position=[])
                global_m_el = b_block * fx.Int32(128) + m_block * fx.Int32(M_TILE) + m_lo_o + fx.Int32(el)
                global_n_el = fx.Int32(out_n * 16) + n_o
                o_off = global_m_el * fx.Int32(D_V) + global_n_el
                o_.store(o_off, scalar, vec_size=1)

    @flyc.jit
    def launch(q, kc, idx, vc, o, num_m_wgs: fx.Int32, num_bs: fx.Int32):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(q, kc, idx, vc, o).launch(
            grid=(num_m_wgs, num_bs, 1),
            block=(64, 1, 1),
        )

    return launch


def main():
    torch.manual_seed(1)
    BS = int(os.environ.get("BS", "159"))
    M_HEADS, BI, D, D_V = 128, 64, 448, 448
    BS_KV = 128
    NB = max(2, (BS * BI * 2) // BS_KV + 1)
    M_TILE = 16
    NUM_M_WGS = M_HEADS // M_TILE

    print(f"[sparse] BS={BS}, NB={NB}, shapes Q=({BS},{M_HEADS},{D})  k_cache=({NB},{BS_KV},{D})",
          flush=True)
    launch = build_sparse_attn()

    q       = torch.randn(BS, M_HEADS, D, dtype=torch.bfloat16, device="cuda") * 0.05
    k_cache = torch.randn(NB, BS_KV,    D, dtype=torch.bfloat16, device="cuda") * 0.05
    v_cache = torch.randn(NB, BS_KV,  D_V, dtype=torch.bfloat16, device="cuda") * 0.05
    indices = torch.randint(0, NB * BS_KV, (BS, BI), dtype=torch.int32, device="cuda")
    o       = torch.zeros(BS, M_HEADS, D_V, dtype=torch.float32, device="cuda")

    launch(q.reshape(-1), k_cache.reshape(-1), indices.reshape(-1),
           v_cache.reshape(-1), o.reshape(-1), NUM_M_WGS, BS)
    torch.cuda.synchronize()
    print("[sparse] kernel ran OK", flush=True)

    # Reference: gather K and V using indices, then full attention.
    block_id = indices // BS_KV
    in_block = indices % BS_KV
    k_gathered = k_cache[block_id.long(), in_block.long()]   # (BS, BI, D)
    v_gathered = v_cache[block_id.long(), in_block.long()]   # (BS, BI, D_V)
    s_ref = torch.einsum("bhd,bnd->bhn", q.float(), k_gathered.float())
    s_ref_sm = torch.softmax(s_ref, dim=-1)
    o_ref = torch.einsum("bhn,bnv->bhv", s_ref_sm, v_gathered.float())

    diff = (o.cpu() - o_ref.cpu()).abs()
    print(f"[sparse] max abs diff: {diff.max().item():.6e}", flush=True)
    print(f"[sparse] mean abs diff: {diff.mean().item():.6e}", flush=True)
    tol = 5e-3
    n_within = (diff < tol).sum().item()
    total = BS * M_HEADS * D_V
    pct = 100 * n_within / total
    if pct > 99.9:
        print(f"[sparse] PASS — {n_within}/{total} ({pct:.2f}%) within {tol}", flush=True)
    else:
        print(f"[sparse] FAIL — {pct:.2f}% within tol", flush=True)
        sys.exit(2)

    # Bench
    for _ in range(10):
        launch(q.reshape(-1), k_cache.reshape(-1), indices.reshape(-1),
               v_cache.reshape(-1), o.reshape(-1), NUM_M_WGS, BS)
    torch.cuda.synchronize()
    N_ITERS = 100
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        launch(q.reshape(-1), k_cache.reshape(-1), indices.reshape(-1),
               v_cache.reshape(-1), o.reshape(-1), NUM_M_WGS, BS)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N_ITERS * 1e6
    print(f"\n[sparse] perf: {per_call:.2f} µs/call  ({per_call/BS:.3f} µs/batch)  BS={BS}",
          flush=True)
    print(f"\n[sparse] baselines from round 14b:", flush=True)
    print(f"    triton    (FP8 sparse): 0.868 µs/batch", flush=True)
    print(f"    tilelang  (FP8 sparse): 1.825 µs/batch", flush=True)
    print(f"    flydsl    (BF16 dense): 0.216 µs/batch", flush=True)
    print(f"    flydsl    (BF16 SPARSE — this round): {per_call/BS:.3f} µs/batch", flush=True)


if __name__ == "__main__":
    main()
