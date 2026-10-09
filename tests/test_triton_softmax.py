from __future__ import annotations

import unittest

import torch

from kernels.triton.softmax import softmax_triton


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonSoftmaxTest(unittest.TestCase):
    def test_matches_pytorch_reference(self) -> None:
        torch.manual_seed(2026)

        x = torch.randn(2, 3, 1024, device="cuda", dtype=torch.float16)

        expected = torch.softmax(x, dim=-1)
        actual = softmax_triton(x)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_is_numerically_stable_for_large_logits(self) -> None:
        x = torch.tensor(
            [[1000.0, 999.0, -1000.0]],
            device="cuda",
            dtype=torch.float16,
        )

        actual = softmax_triton(x)

        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual.float().sum(dim=-1),
            torch.ones(1, device="cuda"),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_rejects_non_contiguous_input(self) -> None:
        x = torch.randn(2, 4, 1024, device="cuda", dtype=torch.float16)
        non_contiguous_x = x.transpose(1, 2)

        with self.assertRaisesRegex(ValueError, "contiguous"):
            softmax_triton(non_contiguous_x)


if __name__ == "__main__":
    unittest.main()