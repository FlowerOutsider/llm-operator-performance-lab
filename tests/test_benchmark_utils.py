"""Unit tests for benchmark helpers that do not require a GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmark_utils import benchmark_cuda_interleaved, summarize_latencies


class BenchmarkUtilsTest(unittest.TestCase):
    def test_summarize_latencies(self) -> None:
        result = summarize_latencies("example", [1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result.name, "example")
        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.min_ms, 1.0)
        self.assertEqual(result.median_ms, 3.0)
        self.assertEqual(result.p95_ms, 4.0)

    def test_rejects_empty_latencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            summarize_latencies("empty", [])

    def test_rejects_empty_interleaved_implementations(self) -> None:
        with self.assertRaisesRegex(ValueError, "functions must not be empty"):
            benchmark_cuda_interleaved({})


if __name__ == "__main__":
    unittest.main()