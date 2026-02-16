#!/usr/bin/env python3
"""
Scan matrix sizes to trigger batch-invariant drift on AMD/ROCm.

NVIDIA cuBLAS: ~0.75 diff at M=32,K=128,N=1024.
AMD rocBLAS: diff=0 at that size; larger sizes (64x512x2048+) may show drift.
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
iters = 3

# 多种矩阵规模（来自 test_batch_invariant_ops）
CONFIGS = [
    ("原始-NV复现", 32, 128, 1024, False),   # 原始参数，linspace
    ("Medium-2", 64, 512, 2048, False),
    ("Large-1", 128, 1024, 4096, False),
    ("Small-1", 8, 64, 128, False),
    ("随机-32x128x1024", 32, 128, 1024, True),
    ("随机-64x512x2048", 64, 512, 2048, True),
]


def compute_diff(a, b):
    out1 = torch.mm(a[:1], b)
    out2 = torch.mm(a, b)[:1]
    return (out1 - out2).abs().max().item()


def make_inputs(M, K, N, use_random=False, seed=42):
    if use_random:
        g = torch.Generator(device=device).manual_seed(seed)
        a = torch.randn(M, K, dtype=dtype, device=device, generator=g) * 10
        b = torch.randn(N, K, dtype=dtype, device=device, generator=g).t()
    else:
        a = torch.linspace(-100, 100, M * K, dtype=dtype, device=device).reshape(M, K)
        b = torch.linspace(-100, 100, K * N, dtype=dtype, device=device).reshape(N, K)
        b = b.transpose(0, 1)
    return a, b


print("=" * 70)
print("Batch Invariant 差异扫描（AMD/ROCm）")
print("=" * 70)
print(f"设备: {device}")
print()

for name, M, K, N, use_random in CONFIGS:
    diffs = []
    for i in range(iters):
        seed = 42 + i if use_random else 42
        a, b = make_inputs(M, K, N, use_random, seed)
        d = compute_diff(a, b)
        diffs.append(d)
    max_d = max(diffs)
    marker = " <-- 有差异!" if max_d > 1e-5 else ""
    print(f"  {name:25} M={M:3} K={K:4} N={N:5}  max_diff={max_d:.6f}{marker}")

print()
print("说明：若所有 max_diff 均为 0，说明 rocBLAS 在当前规模下")
print("      对 batch=1 和 batch=M 使用了相同归约顺序。")
print("      这属于 AMD 与 NVIDIA BLAS 实现的正常差异。")
print("=" * 70)
