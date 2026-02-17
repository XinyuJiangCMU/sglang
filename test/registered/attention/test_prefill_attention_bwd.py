"""
Test Triton prefill attention backward (training path).

Run with pytest (from repo root):
  PYTHONPATH=python python -m pytest test/registered/attention/test_prefill_attention_bwd.py -v

Run as script (from repo root):
  PYTHONPATH=python python test/registered/attention/test_prefill_attention_bwd.py
  (Script prints the device used, e.g. cuda:0 / GPU name.)
  Pick GPU: NVIDIA: CUDA_VISIBLE_DEVICES=N  |  AMD/ROCm: HIP_VISIBLE_DEVICES=N

Or from python/:
  python -m pytest ../test/registered/attention/test_prefill_attention_bwd.py -v
  python ../test/registered/attention/test_prefill_attention_bwd.py
"""
import sys
import unittest
from pathlib import Path

import torch

# When run as script, ensure sglang is importable (python/ on path)
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    python_dir = root / "python"
    if python_dir.is_dir() and str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_bwd,
    context_attention_fwd,
    context_attention,
)
import sglang.srt.layers.attention.triton_ops.prefill_attention as _prefill_mod
from sglang.test.test_utils import CustomTestCase


def _has_cuda_or_hip():
    return _prefill_mod._is_cuda or _prefill_mod._is_hip


def _print_device_info():
    """Print which device (GPU/CPU) is used. CUDA/HIP backend uses default GPU (cuda:0)."""
    if not torch.cuda.is_available():
        print("Device: CPU (CUDA not available)")
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    print(f"Device: cuda:{idx} ({name})")


@unittest.skipUnless(_has_cuda_or_hip(), "requires CUDA or HIP")
class TestPrefillAttentionBackward(CustomTestCase):
    """Triton prefill attention forward + backward + autograd."""

    def test_forward_with_lse(self):
        batch, seq_len, head, head_dim = 2, 64, 4, 64
        total = batch * seq_len
        # PyTorch on ROCm also uses "cuda" as device name, not "hip"
        device = "cuda"
        q = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        k = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        v = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        b_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        b_seq_len = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
        max_input_len = seq_len

        o = torch.empty_like(q)
        o, lse = context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, return_lse=True
        )
        self.assertEqual(o.shape, (total, head, head_dim))
        self.assertEqual(lse.shape, (total, head))

    def test_backward(self):
        batch, seq_len, head, head_dim = 2, 64, 4, 64
        total = batch * seq_len
        # PyTorch on ROCm also uses "cuda" as device name, not "hip"
        device = "cuda"
        q = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        k = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        v = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        b_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        b_seq_len = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
        max_input_len = seq_len

        o = torch.empty_like(q)
        o, lse = context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, return_lse=True
        )
        dO = torch.ones_like(o)
        dq, dk, dv = context_attention_bwd(
            dO, o, lse, q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal=True
        )
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_autograd(self):
        batch, seq_len, head, head_dim = 2, 64, 4, 64
        total = batch * seq_len
        # PyTorch on ROCm also uses "cuda" as device name, not "hip"
        device = "cuda"
        q = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        k = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        v = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        b_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        b_seq_len = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
        max_input_len = seq_len

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        out = context_attention(q2, k2, v2, b_start_loc, b_seq_len, max_input_len, is_causal=True)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(q2.grad)
        self.assertIsNotNone(k2.grad)
        self.assertIsNotNone(v2.grad)
        self.assertEqual(q2.grad.shape, q.shape)


if __name__ == "__main__":
    # Run as script: forward + backward + autograd, with print
    if not _has_cuda_or_hip():
        print("No CUDA/HIP, skip")
        sys.exit(0)

    _print_device_info()
    batch, seq_len, head, head_dim = 2, 64, 4, 64
    total = batch * seq_len
    # PyTorch on ROCm also uses "cuda" as device name, not "hip"
    device = "cuda"
    q = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
    k = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
    v = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
    b_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
    max_input_len = seq_len

    print("Forward with LSE...")
    o = torch.empty_like(q)
    o, lse = context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, return_lse=True
    )
    print("  o:", o.shape, "lse:", lse.shape)

    print("Backward (first run may compile 1–3 min)...")
    dq, dk, dv = context_attention_bwd(
        o.clone().fill_(1.0), o, lse, q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal=True
    )
    print("  dq:", dq.shape, "dk:", dk.shape, "dv:", dv.shape)

    print("Autograd...")
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out = context_attention(q2, k2, v2, b_start_loc, b_seq_len, max_input_len, is_causal=True)
    loss = out.sum()
    loss.backward()
    print("  q.grad:", q2.grad.shape if q2.grad is not None else None)
    print("OK")
