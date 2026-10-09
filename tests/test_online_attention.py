from __future__ import annotations

import unittest

import torch

from attention.online_softmax import (
    full_attention_reference,
    online_softmax_attention_reference,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class OnlineSoftmaxAttentionTest(unittest.TestCase):
    def test_matches_materialized_attention(self) -> None:
        torch.manual_seed(2026)

        q = torch.randn(2, 3, 17, 16, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 3, 23, 16, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 3, 23, 12, device="cuda", dtype=torch.float16)

        expected = full_attention_reference(q, k, v)
        actual = online_softmax_attention_reference(
            q,
            k,
            v,
            q_block_size=7,
            kv_block_size=8,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_rejects_invalid_block_size(self) -> None:
        q = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 1, 4, 8, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            online_softmax_attention_reference(
                q,
                k,
                v,
                q_block_size=0,
            )


if __name__ == "__main__":
    unittest.main()