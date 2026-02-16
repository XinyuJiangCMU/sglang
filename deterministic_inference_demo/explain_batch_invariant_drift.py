#!/usr/bin/env python3
"""
Batch Invariant 误差来源详解

演示：为什么 mm(a[:1], b) 和 mm(a, b)[:1] 在 GPU 上会产生差异（如 0.75）？

核心原因：浮点加法不满足结合律 (a+b)+c ≠ a+(b+c)，
         GPU 在不同 batch size 下使用不同的归约顺序，导致不同结果。

运行：python explain_batch_invariant_drift.py
需要：GPU (NVIDIA or AMD). CPU 上差异可能为 0。
注意：Experiment 4 (batch-invariant mode) fails on AMD due to Triton compatibility
      (tl.range flatten arg). Experiments 1-3 work on both.
"""
import torch

# =============================================================================
# 1. 复现原始测试的参数（与 test_without_batch_invariant_mode 完全一致）
# =============================================================================
M, K, N = 32, 128, 1024
dtype = torch.float32
iters = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Batch Invariant 误差来源详解")
print("=" * 70)
print(f"设备: {device}")
print(f"矩阵规模: A({M}×{K}) @ B({K}×{N}) → 输出({M}×{N})")
print(f"我们比较：out1 = mm(a[:1], b) vs out2 = mm(a, b)[:1]")
print(f"数学上两者应完全相同（都是第 1 行的 dot product 结果）")
print()


# =============================================================================
# 2. 生成与原始测试相同的输入
# =============================================================================
def make_inputs(device, dtype):
    """与 test_batch_invariant_ops 中完全相同的构造方式"""
    a = torch.linspace(-100, 100, M * K, dtype=dtype, device=device).reshape(M, K)
    b = torch.linspace(-100, 100, K * N, dtype=dtype, device=device).reshape(N, K)
    b = b.transpose(0, 1)  # 得到 (K, N)，非连续
    return a, b


# =============================================================================
# 3. 核心实验：两种计算方式
# =============================================================================
def compute_diff(a, b):
    """返回两种方式的最大元素级差异"""
    # 方式 1：只看第 1 行，batch size = 1
    out1 = torch.mm(a[:1], b)  # shape: (1, N)

    # 方式 2：全矩阵计算，再取第 1 行
    out2_pre = torch.mm(a, b)  # shape: (M, N)
    out2 = out2_pre[:1]      # shape: (1, N)

    diff = (out1 - out2).abs()
    return diff.max().item(), out1, out2, diff


# =============================================================================
# 4. 运行多次，复现 [0.75, 0.75, 0.75, 0.75, 0.75]
# =============================================================================
print("【实验 1】关闭 Batch Invariant 模式，重复 5 次")
print("-" * 70)
diffs = []
for i in range(iters):
    a, b = make_inputs(device, dtype)
    max_diff, out1, out2, diff_mat = compute_diff(a, b)
    diffs.append(max_diff)
    print(f"  第 {i+1} 次: max|out1 - out2| = {max_diff:.6f}")

print()
print(f"  汇总: diffs = {[f'{d:.2f}' for d in diffs]}")
print()
print("  若你在 CUDA 上运行，应看到约 0.75 的差异。")
print("  差异来源于：不同 batch size 下，cuBLAS 对 K 维归约的加法顺序不同。")
print()


# =============================================================================
# 5. 找出误差最大的位置
# =============================================================================
print("【实验 2】定位误差最大的输出位置")
print("-" * 70)
a, b = make_inputs(device, dtype)
max_diff, out1, out2, diff_mat = compute_diff(a, b)

# 找 argmax
flat_idx = diff_mat.argmax().item()
j = flat_idx % N  # 列索引
print(f"  最大差异出现在输出第 0 行、第 {j} 列")
print(f"  out1[0, {j}] = {out1[0, j].item():.10f}")
print(f"  out2[0, {j}] = {out2[0, j].item():.10f}")
print(f"  差异 = {abs(out1[0, j].item() - out2[0, j].item()):.10f}")
print()


# =============================================================================
# 6. 手动用 float64 计算“真值”，看浮点误差
# =============================================================================
print("【实验 3】用 float64 计算参考值（展示理论一致）")
print("-" * 70)
a_f64 = a.double()
b_f64 = b.double()
out1_ref = torch.mm(a_f64[:1], b_f64)
out2_ref = torch.mm(a_f64, b_f64)[:1]
manual_ref = (a_f64[0] @ b_f64).unsqueeze(0)  # 手动 dot product

print(f"  第 {j} 列:")
print(f"    out1 (float32): {out1[0, j].item():.12f}")
print(f"    out2 (float32): {out2[0, j].item():.12f}")
print(f"    out1 (float64): {out1_ref[0, j].item():.12f}")
print(f"    out2 (float64): {out2_ref[0, j].item():.12f}")
print(f"    手动 dot (f64): {manual_ref[0, j].item():.12f}")
print()
print("  在 float64 下，out1 和 out2 通常一致（或差异极小）。")
print("  在 float32 下，不同归约顺序导致 ~0.75 的差异。")
print()


# =============================================================================
# 7. 用 Batch Invariant 模式验证
# NOTE: Fails on AMD (Triton tl.range flatten compat). Works on NVIDIA.
# =============================================================================
print("【实验 4】开启 Batch Invariant 模式")
print("-" * 70)
try:
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode

    with set_batch_invariant_mode(True):
        diffs_bi = []
        for i in range(iters):
            a, b = make_inputs(device, dtype)
            max_diff, _, _, _ = compute_diff(a, b)
            diffs_bi.append(max_diff)
        print(f"  diffs = {diffs_bi}")
        print(f"  所有 diff 均为 0：{all(d == 0.0 for d in diffs_bi)}")
except ImportError as e:
    print(f"  跳过（需在 sglang 环境中运行）: {e}")

print()
print("=" * 70)
print("总结：0.75 来自 cuBLAS 在 batch=1 vs batch=32 时不同的归约顺序，")
print("      导致 float32 累加时的舍入误差不同。")
print("      Batch Invariant 通过固定计算顺序消除该差异。")
print("=" * 70)