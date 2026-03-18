"""Tests for LLaDA2 DLLM on AMD MI300X.

Verifies that DLLM (discrete diffusion LLM) works correctly on AMD
with the AITER attention backend.
"""
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


@unittest.skipIf(
    not __import__("torch").cuda.is_available()
    or "MI" not in __import__("torch").cuda.get_device_name(0),
    "AMD MI GPU required",
)
class TestLLaDA2AMD(CustomTestCase):
    """Test LLaDA2 DLLM on AMD MI300X with AITER backend."""

    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = "http://127.0.0.1:30099"

        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "1",
            "--dllm-algorithm",
            "LowConfidence",
            "--disable-cuda-graph",  # CUDA graph has KV metadata replay bug for DLLM+AITER
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_correctness(self):
        """Test that LLaDA2 produces meaningful output on AMD."""
        import requests

        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        data = resp.json()
        text = data["choices"][0]["text"].lower()
        # Should mention Paris
        self.assertIn("paris", text, f"Expected 'paris' in output: {text}")

    def test_speed(self):
        """Test that LLaDA2 achieves reasonable speed on AMD MI300X."""
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            max_new_tokens=128,
        )
        acc_length, speed = send_one_prompt(args)

        print(f"LLaDA2 on AMD MI300X: {speed:.1f} tok/s")

        # AMD CI threshold is ≥ 10 tok/s (from test_llada2_mini.py)
        # We typically see 49-59 tok/s on MI300X
        self.assertGreater(speed, 10, f"Speed {speed:.1f} tok/s below 10 tok/s threshold")


if __name__ == "__main__":
    unittest.main()
