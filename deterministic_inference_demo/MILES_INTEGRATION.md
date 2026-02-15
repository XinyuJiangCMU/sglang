# Miles True On-Policy 集成指南

在完成 SGLang 确定性推理验证后，可按以下步骤运行 Miles true_on_policy 示例。

## 目标

- `train/train_rollout_logprob_abs_diff = 0`：训练与推理 logprobs 逐位一致
- 需 SGLang `--enable-deterministic-inference` + 训练侧相同确定性 kernels

## 前置条件

- Miles 已 clone 至 `/data/xinyu/miles`（或设置 `MILES_REPO` 环境变量）
- 安装 Miles：`pip install -e /data/xinyu/miles`
- SGLang 已安装且支持 FA3、batch invariant ops

## 快速运行

### 方式 1：run_simple.py（推荐先试）

```bash
# 一键脚本（从 deterministic_inference_demo 目录）
./run_miles_true_on_policy.sh

# 或手动运行（MILES_REPO 默认 /data/xinyu/miles）
cd /data/xinyu/miles
MILES_SCRIPT_MODE=debug_minimal MILES_SCRIPT_MODEL_NAME=Qwen3-0.6B python examples/true_on_policy/run_simple.py
```

关键参数（已在 run_simple.py 中配置）：
- `--sglang-enable-deterministic-inference`
- `--sglang-attention-backend fa3`
- `--attn-implementation flash_attention_3`
- `--deterministic-mode`
- `--true-on-policy-mode`

### 方式 2：run_qwen3_4b.py

```bash
cd /data/xinyu/miles
python scripts/run_qwen3_4b.py --train-backend fsdp --true-on-policy
# 快速模式
python scripts/run_qwen3_4b.py --train-backend fsdp --true-on-policy --mode debug_minimal
```

## 验证成功

在 wandb 中检查 `train/train_rollout_logprob_abs_diff`，应为 **0**。

## 路径说明

run_simple.py 默认使用 `/root/models`、`/root/datasets`。若需自定义，可修改 `prepare()` 或通过环境变量（若 Miles 支持）指定。

## 参考

- [Miles true_on_policy README](https://github.com/radixark/miles/blob/main/examples/true_on_policy/README.md)
- [SGLang for RL](../docs/advanced_features/sglang_for_rl.md)
