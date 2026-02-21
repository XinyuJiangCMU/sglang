#!/usr/bin/env python3
"""
Single-mode test for AMD: same prompt, different batch sizes (1..n).
Checks if outputs are identical. Unique samples: 1 = pass, >1 = inconsistent.
Saves inconsistent outputs and detailed diff to .log for teaching.
"""
import argparse
from datetime import datetime

import requests

PROMPT = "给我介绍下sglang"

# Context window around first diff (chars before/after)
DIFF_CONTEXT = 80


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


def find_first_diff(a, b, context=DIFF_CONTEXT):
    """Return (first_diff_pos, segment_a, segment_b) with context around diff."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            lo = max(0, i - context)
            hi_a = min(len(a), i + context)
            hi_b = min(len(b), i + context)
            return i, a[lo:hi_a], b[lo:hi_b]
    return len(a), a[-context:], b[-context:] if len(a) == len(b) else (a[-context:], b[-context:])


def run_test(host, port, n_trials, temperature=0.0, log_path=None):
    texts = []
    for bs in range(1, n_trials + 1):
        text = send_generate(host, port, PROMPT, bs, temperature=temperature)
        if text is None:
            return
        t = text.replace("\n", " ")
        texts.append(t)
        print(f"Trial {bs} (batch_size={bs}): {t[:80]}...")

    unique_texts = list(dict.fromkeys(texts))
    unique = len(unique_texts)

    if log_path is None:
        log_path = "single_mode_test_amd.log"

    lines = []
    lines.append("=" * 70)
    lines.append(f"Single-Mode Test (AMD) - {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append(f"Prompt: {PROMPT[:60]}...")
    lines.append(f"Trials: 1..{n_trials}, temperature={temperature}")
    lines.append(f"Total: {len(texts)}, Unique: {unique}")
    lines.append("")

    if unique == 1:
        lines.append("PASS")
    else:
        lines.append(f"FAIL: {unique} different outputs\n")
        for i, u in enumerate(unique_texts):
            indices = [j + 1 for j, t in enumerate(texts) if t == u]
            lines.append(f"  [output {i + 1}] at trial (batch_size): {indices}")

        if len(unique_texts) >= 2:
            out1, out2 = unique_texts[0], unique_texts[1]
            pos, seg_a, seg_b = find_first_diff(out1, out2)

            lines.append("")
            lines.append("-" * 70)
            lines.append("INCONSISTENT OUTPUTS (both full texts saved below)")
            lines.append("-" * 70)
            lines.append("")
            lines.append("=== Output 1 (batch_size=1 typically) ===")
            lines.append(out1)
            lines.append("")
            lines.append("=== Output 2 (batch_size>=2 typically) ===")
            lines.append(out2)
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"First diff at char index {pos} (context +/- {DIFF_CONTEXT})")
            lines.append("-" * 70)
            lines.append("Output1 segment:")
            lines.append(seg_a)
            lines.append("")
            lines.append("Output2 segment:")
            lines.append(seg_b)
            lines.append("")
            lines.append("=" * 40)
            lines.append("Why: batch=1 vs batch>=2 use different BLAS/GEMM code paths.")
            lines.append("      Different reduction order -> different float rounding.")
            lines.append("      Enable --enable-deterministic-inference to fix.")
            lines.append("=" * 40)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            lines.append("")
            lines.append(f">>> Full report saved to: {log_path}")

    for line in lines:
        print(line)

    return unique


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AMD single-mode: same prompt, various batch sizes")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.0, help="0 for greedy")
    p.add_argument("--log", default="single_mode_test_amd.log", help="Save full diff report")
    args = p.parse_args()
    run_test(args.host, args.port, args.n_trials, args.temperature, args.log)
