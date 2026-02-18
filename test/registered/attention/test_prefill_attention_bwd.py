"""
Test Triton prefill attention backward (training path) and correctness vs PyTorch SDPA reference.

Run with pytest (from repo root):
  PYTHONPATH=python python -m pytest test/registered/attention/test_prefill_attention_bwd.py -v

Run validation suite (forward/backward vs reference):
  PYTHONPATH=python python test/registered/attention/test_prefill_attention_bwd.py --validate

Run as script (basic smoke):
  PYTHONPATH=python python test/registered/attention/test_prefill_attention_bwd.py

Pick GPU: NVIDIA: CUDA_VISIBLE_DEVICES=N  |  AMD/ROCm: HIP_VISIBLE_DEVICES=N

Validation results:
  - PASS: Forward/backward match reference within tolerance (atol/rtol).
  - FAIL: assert_close raises; indicates Triton implementation differs from SDPA math.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

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


# -----------------------------------------------------------------------------
# Reference: PyTorch F.scaled_dot_product_attention (varlen prefill)
# -----------------------------------------------------------------------------
def prefill_attention_ref(
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    is_causal=True,
):
    """
    Reference prefill attention using F.scaled_dot_product_attention.
    q, k, v: [total_tokens, head, head_dim]
    b_start_loc: [batch], b_seq_len: [batch]
    """
    batch = b_seq_len.shape[0]
    out = torch.empty_like(q)

    for b in range(batch):
        start = int(b_start_loc[b].item())
        length = int(b_seq_len[b].item())
        q_b = q[start : start + length]  # [S, H_q, D]
        k_b = k[start : start + length]  # [S, H_kv, D]
        v_b = v[start : start + length]

        # [S, H, D] -> [1, H, S, D]
        q_b = q_b.unsqueeze(0).transpose(1, 2)
        k_b = k_b.unsqueeze(0).transpose(1, 2)
        v_b = v_b.unsqueeze(0).transpose(1, 2)

        # SDPA uses scale=1/sqrt(d_k) by default, same as our prefill
        o_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        o_b = o_b.transpose(1, 2).squeeze(0)
        out[start : start + length] = o_b
    return out


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


# -----------------------------------------------------------------------------
# Validation: Triton prefill vs PyTorch SDPA reference
# -----------------------------------------------------------------------------
@unittest.skipUnless(_has_cuda_or_hip(), "requires CUDA or HIP")
class TestPrefillVsReference(CustomTestCase):
    """
    Verify Triton prefill attention matches PyTorch F.scaled_dot_product_attention.
    - Forward: O_ours vs O_ref (assert_close)
    - Backward: dq/dk/dv vs autograd on reference (assert_close)
    - Numerical: sample gradient check for dq
    """

    # Tolerances for float32. Relaxed rtol due to accumulation order differences.
    ATOL_FWD = 1e-3
    RTOL_FWD = 1e-2
    ATOL_BWD = 1e-2
    RTOL_BWD = 1e-1
    ATOL_NUM = 1e-2
    RTOL_NUM = 1e-1

    def _setup(self, batch=2, seq_len=64, head=4, head_dim=64, seed=42):
        torch.manual_seed(seed)
        total = batch * seq_len
        device = "cuda"
        q = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        k = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        v = torch.randn(total, head, head_dim, dtype=torch.float32, device=device) * 0.1
        b_start_loc = torch.tensor(
            [i * seq_len for i in range(batch)],
            dtype=torch.int32,
            device=device,
        )
        b_seq_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
        max_input_len = seq_len
        return q, k, v, b_start_loc, b_seq_len, max_input_len

    def test_forward_vs_reference(self):
        """Forward output O_ours should match O_ref (SDPA) within atol/rtol."""
        q, k, v, b_start_loc, b_seq_len, max_input_len = self._setup()
        o_ref = prefill_attention_ref(q, k, v, b_start_loc, b_seq_len, is_causal=True)
        o_ours = torch.empty_like(q)
        o_ours, _ = context_attention_fwd(
            q, k, v, o_ours, b_start_loc, b_seq_len, max_input_len,
            is_causal=True, return_lse=True,
        )
        torch.testing.assert_close(
            o_ours, o_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD
        )

    def test_forward_vs_reference_non_causal(self):
        """Forward (non-causal) should match reference."""
        q, k, v, b_start_loc, b_seq_len, max_input_len = self._setup()
        o_ref = prefill_attention_ref(q, k, v, b_start_loc, b_seq_len, is_causal=False)
        o_ours = torch.empty_like(q)
        o_ours, _ = context_attention_fwd(
            q, k, v, o_ours, b_start_loc, b_seq_len, max_input_len,
            is_causal=False, return_lse=True,
        )
        torch.testing.assert_close(
            o_ours, o_ref, atol=self.ATOL_FWD, rtol=self.RTOL_FWD
        )

    def test_backward_vs_reference(self):
        """Backward dq/dk/dv should match autograd gradients from reference."""
        q, k, v, b_start_loc, b_seq_len, max_input_len = self._setup()
        q_r = q.detach().clone().requires_grad_(True)
        k_r = k.detach().clone().requires_grad_(True)
        v_r = v.detach().clone().requires_grad_(True)
        o_ref = prefill_attention_ref(q_r, k_r, v_r, b_start_loc, b_seq_len, is_causal=True)
        loss = o_ref.sum()
        loss.backward()
        dq_ref, dk_ref, dv_ref = q_r.grad, k_r.grad, v_r.grad

        o_ours = torch.empty_like(q)
        o_ours, lse = context_attention_fwd(
            q, k, v, o_ours, b_start_loc, b_seq_len, max_input_len,
            is_causal=True, return_lse=True,
        )
        dO = torch.ones_like(o_ours)
        dq_ours, dk_ours, dv_ours = context_attention_bwd(
            dO, o_ours, lse, q, k, v,
            b_start_loc, b_seq_len, max_input_len, is_causal=True,
        )

        torch.testing.assert_close(dq_ours, dq_ref, atol=self.ATOL_BWD, rtol=self.RTOL_BWD)
        torch.testing.assert_close(dk_ours, dk_ref, atol=self.ATOL_BWD, rtol=self.RTOL_BWD)
        torch.testing.assert_close(dv_ours, dv_ref, atol=self.ATOL_BWD, rtol=self.RTOL_BWD)

    def test_backward_numerical_gradient(self):
        """Numerical gradient check for dq (sample a few elements).
        Uses reference forward so numerical grad matches dq_ref; we compare to dq_ours
        which already matches dq_ref (test_backward_vs_reference). FD has inherent noise.
        """
        q, k, v, b_start_loc, b_seq_len, max_input_len = self._setup()
        # eps=1e-3 is more stable for float32 central diff than 1e-4 (avoids round-off)
        eps = 1e-3

        o_base = torch.empty_like(q)
        o_base, lse = context_attention_fwd(
            q, k, v, o_base, b_start_loc, b_seq_len, max_input_len,
            is_causal=True, return_lse=True,
        )
        dO = torch.ones_like(o_base)
        dq_ours, _, _ = context_attention_bwd(
            dO, o_base, lse, q, k, v,
            b_start_loc, b_seq_len, max_input_len, is_causal=True,
        )

        # Use reference forward for numerical grad to match dq_ref/dq_ours (avoids
        # Triton vs SDPA numerics mismatch; backward_vs_reference already validates ours)
        def _loss(q_in):
            o = prefill_attention_ref(
                q_in, k, v, b_start_loc, b_seq_len, is_causal=True
            )
            return o.sum()

        # Sample a few (i, j, k) indices and check numerical gradient
        total, head, head_dim = q.shape
        rng = torch.Generator(device="cpu").manual_seed(123)
        for _ in range(5):
            i = torch.randint(0, total, (1,), generator=rng).item()
            j = torch.randint(0, head, (1,), generator=rng).item()
            kk = torch.randint(0, head_dim, (1,), generator=rng).item()

            q_plus = q.clone()
            q_plus[i, j, kk] += eps
            loss_plus = _loss(q_plus)
            q_minus = q.clone()
            q_minus[i, j, kk] -= eps
            loss_minus = _loss(q_minus)
            numerical = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            analytical = dq_ours[i, j, kk].item()
            self.assertLess(
                abs(numerical - analytical),
                self.ATOL_NUM + self.RTOL_NUM * (abs(numerical) + 1e-8),
                msg=f"dq[{i},{j},{kk}]: numerical={numerical:.6e} analytical={analytical:.6e}",
            )


if __name__ == "__main__":
    if not _has_cuda_or_hip():
        print("No CUDA/HIP, skip")
        sys.exit(0)

    _print_device_info()

    # --validate: run correctness tests vs PyTorch SDPA reference
    if "--validate" in sys.argv:
        print("\n=== Validation vs PyTorch SDPA reference ===\n")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestPrefillVsReference)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        print("\n" + ("PASS" if result.wasSuccessful() else "FAIL") + " (validation)\n")
        sys.exit(0 if result.wasSuccessful() else 1)

    # Default: smoke test (forward + backward + autograd)
    print("\n=== Smoke test ===\n")
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
