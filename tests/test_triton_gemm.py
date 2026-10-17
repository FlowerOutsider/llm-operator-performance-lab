from __future__ import annotations

import unittest

import torch

from kernels.triton.gemm import gemm_naive_triton, gemm_tiled_triton


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonGEMMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.a = torch.randn(129, 257, device="cuda", dtype=torch.float16)
        self.b = torch.randn(257, 113, device="cuda", dtype=torch.float16)
        self.expected = torch.matmul(self.a, self.b)

    def test_naive_matches_torch(self) -> None:
        actual = gemm_naive_triton(self.a, self.b)
        torch.testing.assert_close(actual, self.expected, rtol=2e-2, atol=2e-2)

    def test_tiled_matches_torch(self) -> None:
        actual = gemm_tiled_triton(self.a, self.b)
        torch.testing.assert_close(actual, self.expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()