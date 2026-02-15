#!/bin/bash
# AMD 确定性推理服务器（Triton attention backend）
# FlashInfer/FA3 在 AMD 上不可用，使用 Triton 实现确定性
# 若使用 TP（多卡）：SGLANG_USE_1STAGE_ALLREDUCE=1 ./start_deterministic_server_amd.sh
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend triton \
    --trust-remote-code \
    --port 30000 \
    --enable-deterministic-inference
