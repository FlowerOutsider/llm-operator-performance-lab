from __future__ import annotations

import unittest

import torch

from kernels.triton.residual_rmsnorm import residual_rmsnorm_triton


def residual_rmsnorm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    z = x.float() + residual.float()
    inverse_rms = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + epsilon)
    return (z * inverse_rms * weight.float()).to(dtype=x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonResidualRMSNormTest(unittest.TestCase):
    def test_matches_pytorch_reference(self) -> None:
        torch.manual_seed(2026)

        x = torch.randn(2, 3, 1024, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(1024, device="cuda", dtype=torch.float16)

        expected = residual_rmsnorm_reference(x, residual, weight)
        actual = residual_rmsnorm_triton(x, residual, weight)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_rejects_shape_mismatch(self) -> None:
        x = torch.randn(2, 3, 1024, device="cuda", dtype=torch.float16)
        residual = torch.randn(2, 3, 512, device="cuda", dtype=torch.float16)
        weight = torch.ones(1024, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "same shape"):
            residual_rmsnorm_triton(x, residual, weight)


if __name__ == "__main__":
    unittest.main()