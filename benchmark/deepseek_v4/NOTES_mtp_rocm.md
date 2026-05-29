# DeepSeek-V4-Pro single-step MTP (EAGLE) on ROCm / MI355X (gfx950)

Status: **WORKING**. Booted to "fired up and ready to roll", spec_accept_rate 0.70–0.875.

## Measured speedup (same source, same env, only --speculative on/off; random 256in/256out)
| concurrency | no-spec | MTP single-step | gain |
|---|---|---|---|
| 1 (single-stream) | 31.5 tok/s, ITL 31.1ms | **55.6 tok/s, ITL 16.9ms** | **1.76x tput / 1.84x ITL** |
| 8 | 196 tok/s, ITL 35.7ms | 133 tok/s, ITL 20.6ms | per-token 1.73x but tput 0.68x (high-batch crossover) |

=> MTP is a big win in the latency-bound low-concurrency regime; hurts throughput at higher batch (classic spec-decode crossover -> optimization target: amortize/skip draft at high batch).

## How it runs (the AMD branch f96ac98 lacks MTP integration; main has it but is NV-oriented)
Run **main source inside the f96ac98 ROCm image**: `git checkout` main in /sgl-workspace/sglang (image's torch/aiter unchanged). main has the DSv4 EAGLE/MTP draft integration (`deepseek_v4_backend_hip_radix.py`, `DeepseekV4MultiStepBackend`); f96ac98 does not.

### Launch recipe
Base = the 25 `SGLANG_OPT_*` AMD env vars from run_dsv4.sh, PLUS:
```
export SGLANG_OPT_FP8_WO_A_GEMM=0          # main defaults True -> _setup_fp8_wo_a_scales imports CUDA-only deep_gemm; f96ac98 defaulted False
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
export SGLANG_OPT_USE_TOPK_V2=0            # avoid topk_v2.cuh CUDA-only JIT (cuda/ptx), use HIP topk op
export SGLANG_TOPK_TRANSFORM_512_TORCH=1   # dynamic-TOPK torch path handles 512 AND 1024 (MTP draft uses 1024)
```
Flags: `--attention-backend compressed` (normalizes to dsv4) `--tp 8 --page-size 256 --chunked-prefill-size 4096 --mem-fraction-static 0.80 --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --max-running-requests 8 --disable-shared-experts-fusion --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4`

### ROCm code fix (1 file, HIP-gated, CUDA path untouched)
`python/sglang/jit_kernel/dsv4/elementwise.py` `fused_q_indexer_rope_hadamard_quant`: the fused indexer-Q kernel (RoPE + 128pt Hadamard + fp8 quant) `torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant` is NOT in the ROCm sgl_kernel build. Added a HIP fallback that decomposes it: RoPE (apply_rotary_emb_triton, interleaved) -> hadamard_transform(scale=head_dim**-0.5) -> per-(token,head) fp8 act-quant (scale=clamp(abs_max,1e-4)/448). Mirrors csrc/deepseek_v4/main_norm_rope.cuh FusedQIndexerRopeHadamardQuantKernel.

## Blockers resolved (all NV-only assumptions on main's DSv4 that break on gfx950)
1. _setup_fp8_wo_a_scales -> deep_gemm.transform_sf_into_required_layout (CUDA) -> env FP8_WO_A_GEMM=0.
2. topk_v2.cuh -> #include <cuda/ptx> (CUDA) -> env USE_TOPK_V2=0 (HIP op).
3. HIP topk op hardcodes TOPK=512 but MTP draft uses 1024 -> env TOPK_TRANSFORM_512_TORCH=1 (dynamic torch path).
4. dsv4_fused_q_indexer_rope_hadamard_quant HIP op missing -> decomposed HIP fallback (the code edit above).

## Optimization opportunities (next)
- High-batch crossover: MTP draft overhead makes tput worse at conc>=~4-8. Dynamic enable (only spec at low batch), or cheaper draft.
- The HIP indexer-Q fallback is decomposed (3 ops) vs CUDA's 1 fused kernel -> a fused HIP/triton kernel would cut launch overhead.
