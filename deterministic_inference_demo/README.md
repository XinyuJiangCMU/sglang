# SGLang 确定性推理 Demo（Miles True On-Policy 前置）

用于在 NVIDIA GPU 上体验 SGLang 的确定性推理能力。

## 快速验证（约 1 分钟，无需模型）

```bash
./01_run_batch_invariant_ops.sh
```

## 完整端到端测试（需下载 Qwen3-8B）

```bash
./02_run_deterministic_inference.sh
```

## 手动分步运行

1. 终端 1 启动服务器：

```bash
cd /data/xinyu/top/sglang
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference \
    --port 30000
```

2. 终端 2 运行测试：

```bash
cd /data/xinyu/top/sglang/deterministic_inference_demo
python run_deterministic_client.py --test-mode single --n-trials 20
```

## 测试模式说明

- **single**：同一 prompt 在不同 batch size (1~n) 下结果应完全一致，`Unique samples: 1` 即通过
- **prefix**：不同长度 prefix 共享时，logprob 应一致
- **radix_cache**：有/无 radix cache 时 prefill 结果一致（FlashInfer 不支持）

## 预期结果

- `Unique samples: 1` → 确定性推理工作正常
- `Unique samples: >1` → 存在非确定性，需排查
