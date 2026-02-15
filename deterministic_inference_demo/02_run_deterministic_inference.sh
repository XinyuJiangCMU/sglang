#!/bin/bash
# 启动服务器 + 运行确定性推理测试
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
MODEL="Qwen/Qwen3-8B"
PORT=30000

echo "=== 启动 SGLang 服务器（FlashInfer + 确定性推理）==="
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --attention-backend flashinfer \
    --trust-remote-code \
    --port $PORT \
    --enable-deterministic-inference &
SERVER_PID=$!

# 等待服务器就绪
echo "等待服务器启动..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/generate" -X POST -H "Content-Type: application/json" \
        -d '{"text":"hi","sampling_params":{"max_new_tokens":1}}' >/dev/null 2>&1; then
        echo "服务器已就绪"
        break
    fi
    sleep 2
done

echo "=== 运行确定性推理测试（single 模式）==="
python -m sglang.test.test_deterministic --test-mode single --n-trials 15 --host localhost --port $PORT

kill $SERVER_PID 2>/dev/null || true
echo "=== 测试完成 ==="
