"""Correctness tests for the Triton RMSNorm forward kernel."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required.")
class TritonRMSNormTest(unittest.TestCase):
    def test_matches_pytorch_reference(self) -> None:
        torch.manual_seed(2026)

        x = torch.randn(2, 3, 1024, device="cuda", dtype=torch.float16)
        weight = torch.randn(1024, device="cuda", dtype=torch.float16)

        expected = rmsnorm_reference(x, weight)
        actual = rmsnorm_triton(x, weight)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_rejects_non_contiguous_input(self) -> None:
        x = torch.randn(2, 1024, 3, device="cuda", dtype=torch.float16).transpose(
            1, 2
        )
        weight = torch.ones(1024, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "contiguous"):
            rmsnorm_triton(x, weight)


if __name__ == "__main__":
    unittest.main()