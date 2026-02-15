#!/usr/bin/env python3
"""
确定性推理客户端测试脚本。
需先手动启动带 --enable-deterministic-inference 的 SGLang 服务器。
"""
import sys
import os

# 确保 sglang 包可被导入
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sglang_python = os.path.join(_script_dir, "..", "python")
sys.path.insert(0, _sglang_python)
os.chdir(os.path.dirname(__file__) or ".")

from sglang.test.test_deterministic import BenchArgs, test_deterministic

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.sampling_seed = args.sampling_seed or 42
    test_deterministic(args)
