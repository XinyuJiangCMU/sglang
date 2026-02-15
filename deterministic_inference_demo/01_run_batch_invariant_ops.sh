#!/bin/bash
# 无需模型，直接验证 Batch Invariant Ops
cd "$(dirname "$0")/.."
python -m pytest test/registered/core/test_batch_invariant_ops.py -v -s
