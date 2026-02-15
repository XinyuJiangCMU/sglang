#!/bin/bash
# 运行 Miles true_on_policy 示例（debug_minimal 快速验证）
# 需先安装 Miles: pip install -e /data/xinyu/miles
MILES_REPO="${MILES_REPO:-/data/xinyu/miles}"
cd "$MILES_REPO"
MILES_SCRIPT_MODE=debug_minimal MILES_SCRIPT_MODEL_NAME=Qwen3-0.6B python examples/true_on_policy/run_simple.py
