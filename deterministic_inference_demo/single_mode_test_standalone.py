#!/usr/bin/env python3
"""确定性推理 Single 模式测试 - 支持 temp>0，展示输出差异"""
import argparse
import requests

PROMPT = "给我介绍下sglang"

def send_generate(host, port, prompt, batch_size, temperature=0.0, sampling_seed=42):
    json_data = {
        "text": [prompt] * batch_size,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": 100,
            "sampling_seed": sampling_seed,
        },
    }
    resp = requests.post(f"http://{host}:{port}/generate", json=json_data)
    if resp.status_code != 200:
        print(f"Error: {resp.json()}")
        return None
    ret = resp.json()
    return (ret[0] if isinstance(ret, list) else ret)["text"]

def find_first_diff(a, b):
    """找到两个字符串首个不同的位置"""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i, a[max(0,i-20):i+50], b[max(0,i-20):i+50]
    return len(a), a[-50:], b[-50:] if len(a) == len(b) else (a[-50:], b[-50:])

def run_test(host, port, n_trials, temperature=0.0):
    texts = []
    for bs in range(1, n_trials + 1):
        text = send_generate(host, port, PROMPT, bs, temperature=temperature)
        if text is None:
            return
        t = text.replace("\n", " ")
        texts.append(t)
        print(f"Trial {bs} (batch_size={bs}): {t[:80]}...")

    unique_texts = list(dict.fromkeys(texts))  # 保持顺序的去重
    unique = len(unique_texts)

    print(f"\nTotal: {len(texts)}, Unique: {unique}")
    if unique == 1:
        print("✅ 通过")
    else:
        print(f"❌ 存在 {unique} 种不同输出\n")
        # 展示每种输出对应的 trial
        for i, u in enumerate(unique_texts):
            indices = [j+1 for j, t in enumerate(texts) if t == u]
            print(f"  [输出{i+1}] 出现在 trial (batch_size): {indices}")
        # 展示前两种输出的差异
        if len(unique_texts) >= 2:
            pos, seg_a, seg_b = find_first_diff(unique_texts[0], unique_texts[1])
            print(f"\n  【差异位置】首个不同在字符索引 {pos}")
            print(f"  输出1 片段: ...{seg_a}...")
            print(f"  输出2 片段: ...{seg_b}...")
    return unique

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()
    run_test(args.host, args.port, args.n_trials, args.temperature)