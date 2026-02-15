#!/bin/bash
# 启动 SGLang 确定性推理服务器 (FA3)
# 单卡: CUDA_VISIBLE_DEVICES=0 ./start_deterministic_server.sh
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --trust-remote-code \
    --port 30000 \
    --enable-deterministic-inference
