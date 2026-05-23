"""r051 Round 10: FP8 e4m3 → bf16 dequant via FlyDSL register bit ops.

Replicates the dequant formula from tilelang dpsk_v4_fp8_partial_kernel
(tilelang_kernel.py lines 1780-1797):

  b_u32 = FP8 byte (0..255)
  sign_bf      = (b_u32 & 0x80) * 0x100        # sign → bf16 bit 15
  exp_e4       = (b_u32 & 0x78) >> 3           # 4-bit exp
  mant_bf      = (b_u32 & 0x7) * 0x10          # 3-bit mantissa → bf16 bits 4-6
  exp_combined = exp_e4 + scale_byte - 7       # add block scale, subtract bias
  bf16_bits    = sign_bf | (exp_combined << 7) | mant_bf

Test: load N FP8 bytes + N scale bytes, dequant in FlyDSL kernel, write
N bf16 results to HBM. Compare to torch FP8 e4m3 → bf16 (with scale).
"""
import os
import sys
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, arith, vector as _fvec, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

sys.path.insert(0, "/sgl-workspace/aiter/aiter/ops/flydsl/kernels")
from tensor_shim import GTensor  # noqa: E402


def build_dequant_kernel(N: int):
    arch = get_hip_arch()
    print(f"[dq] gpu_arch = {arch}", flush=True)

    @flyc.kernel(name="fp8_dequant_kernel")
    def dequant_kernel(
        fp8_tensor: fx.Tensor,      # i8 [N]
        scale_tensor: fx.Tensor,    # i8 [N] (one scale per element for simplicity)
        bf16_out_tensor: fx.Tensor, # bf16 [N]
    ):
        tid = fx.thread_idx.x

        fp8_ = GTensor(fp8_tensor, dtype=T.i8, shape=(-1,))
        scl_ = GTensor(scale_tensor, dtype=T.i8, shape=(-1,))
        out_ = GTensor(bf16_out_tensor, dtype=T.bf16, shape=(-1,))

        # Load one FP8 byte and one scale byte for this thread
        b_i8 = fp8_.load(tid, vec_size=1)
        s_i8 = scl_.load(tid, vec_size=1)

        b_raw = b_i8.value if hasattr(b_i8, "value") else b_i8
        s_raw = s_i8.value if hasattr(s_i8, "value") else s_i8

        # Cast i8 → i32 (zero extend treats as u8)
        b_u32 = arith.extui(T.i32, b_raw)
        s_u32 = arith.extui(T.i32, s_raw)

        # Dequant formula
        c80 = arith.constant(0x80, type=T.i32)
        c78 = arith.constant(0x78, type=T.i32)
        c7  = arith.constant(0x7,  type=T.i32)
        c8  = arith.constant(8,    type=T.i32)
        c3  = arith.constant(3,    type=T.i32)
        c4  = arith.constant(4,    type=T.i32)
        c7s = arith.constant(7,    type=T.i32)
        c100h = arith.constant(0x100, type=T.i32)
        c10 = arith.constant(0x10, type=T.i32)
        c7m = arith.constant(7,    type=T.i32)

        sign_bit = arith.andi(b_u32, c80)
        sign_bf  = arith.muli(sign_bit, c100h)           # << 8

        exp_e4   = arith.shrui(arith.andi(b_u32, c78), c3)

        mant_bf  = arith.muli(arith.andi(b_u32, c7), c10)   # << 4

        # exp_combined = exp_e4 + s_u32 - 7
        exp_sum  = arith.addi(exp_e4, s_u32)
        exp_comb = arith.subi(exp_sum, c7m)

        # bf16_bits = sign_bf | (exp_comb << 7) | mant_bf
        exp_shifted = arith.shli(exp_comb, c7m)
        or1 = arith.ori(sign_bf, exp_shifted)
        bf16_bits_i32 = arith.ori(or1, mant_bf)

        # Truncate to i16, then reinterpret as bf16
        bf16_bits_i16 = arith.trunci(T.i16, bf16_bits_i32)
        bf16_val = arith.bitcast(T.bf16, bf16_bits_i16)

        out_.store(tid, bf16_val, vec_size=1)

    @flyc.jit
    def launch(fp8_t, scale_t, bf16_t):
        dequant_kernel(fp8_t, scale_t, bf16_t).launch(
            grid=(1, 1, 1),
            block=(N, 1, 1),
        )

    return launch


def torch_dequant_ref(fp8_bytes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference matching the bit-level dequant formula."""
    b = fp8_bytes.to(torch.int32).to("cuda") & 0xFF
    s = scale_bytes.to(torch.int32).to("cuda") & 0xFF
    sign_bit = b & 0x80
    sign_bf  = sign_bit << 8
    exp_e4   = (b & 0x78) >> 3
    mant_bf  = (b & 0x7) << 4
    exp_comb = exp_e4 + s - 7
    bits = sign_bf | (exp_comb << 7) | mant_bf
    bits_i16 = bits.to(torch.int32) & 0xFFFF
    bits_u16 = bits_i16.to(torch.int16)
    return bits_u16.view(torch.bfloat16)


def main():
    N = 256  # one workgroup, 256 threads
    print(f"[dq] building kernel for N={N}...", flush=True)
    launch = build_dequant_kernel(N)
    print(f"[dq] launcher built", flush=True)

    # Random FP8 bytes (full byte range) + random scale bytes (e.g., 0..14
    # which gives reasonable bf16 exponents after - 7)
    torch.manual_seed(0)
    fp8_bytes = torch.randint(0, 256, (N,), dtype=torch.uint8, device="cuda").to(torch.int8)
    scale_bytes = torch.randint(0, 14, (N,), dtype=torch.uint8, device="cuda").to(torch.int8)
    out = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

    print(f"[dq] launching...", flush=True)
    launch(fp8_bytes, scale_bytes, out)
    torch.cuda.synchronize()
    print(f"[dq] kernel ran OK", flush=True)

    # Compare to torch reference using EXACT same bit formula
    ref = torch_dequant_ref(fp8_bytes, scale_bytes)
    got_bits = out.view(torch.int16).cpu()
    ref_bits = ref.view(torch.int16).cpu()
    diff = (got_bits != ref_bits).sum().item()
    print(f"[dq] byte-exact: {N - diff}/{N} ({100*(N-diff)/N:.2f}%)", flush=True)

    if diff > 0:
        first = torch.where(got_bits != ref_bits)[0][:3].tolist()
        for i in first:
            print(f"   [{i}] got={out[i].item():.4f} ({got_bits[i].item():#06x}) "
                  f"ref={ref[i].item():.4f} ({ref_bits[i].item():#06x}) "
                  f"fp8=0x{fp8_bytes[i].item()&0xFF:02x} scale=0x{scale_bytes[i].item()&0xFF:02x}",
                  flush=True)
        print(f"\n[dq] VERDICT: FAIL", flush=True)
        sys.exit(2)

    print(f"\n[dq] VERDICT: PASS — FP8 e4m3 dequant byte-exact vs torch ref ({N} samples)",
          flush=True)


if __name__ == "__main__":
    main()
