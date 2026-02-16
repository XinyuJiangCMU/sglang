#!/bin/bash
# Run Batch Invariant demo in sglang container.
# Note: Experiment 4 in explain_batch_invariant_drift.py fails on AMD (Triton compat).

cd "$(dirname "$0")"

echo ">>> 1. 原始脚本（与 test_without_batch_invariant_mode 参数一致）"
echo ">>>    NVIDIA 上应看到 ~0.75 差异，AMD 上可能为 0（rocBLAS 实现不同）"
python3 explain_batch_invariant_drift.py

echo ""
echo ">>> 2. 多规模扫描（尝试在 AMD 上触发差异）"
python3 explain_batch_invariant_drift_scan.py
