from __future__ import annotations

import unittest

from autotune.tuner.recommend import RecommendationRequest, recommend_config


class RecommendTest(unittest.TestCase):
    def test_recommend_chooses_highest_throughput_safe_record(self) -> None:
        records = [
            {
                "backend": "pytorch",
                "batch_size": 1,
                "thread_count": 1,
                "latency_p95_ms": 20,
                "throughput": 40,
                "peak_rss_mb": 500,
                "memory_budget_exceeded": False,
                "cpu_quota_exceeded": False,
            },
            {
                "backend": "onnxruntime",
                "batch_size": 4,
                "thread_count": 2,
                "latency_p95_ms": 50,
                "throughput": 120,
                "peak_rss_mb": 900,
                "memory_budget_exceeded": False,
                "cpu_quota_exceeded": False,
            },
            {
                "backend": "onnxruntime",
                "batch_size": 8,
                "thread_count": 4,
                "latency_p95_ms": 45,
                "throughput": 180,
                "peak_rss_mb": 3000,
                "memory_budget_exceeded": True,
                "cpu_quota_exceeded": False,
            },
        ]
        best, reasons = recommend_config(
            records,
            RecommendationRequest(objective="throughput", latency_budget_ms=60, memory_budget_mb=2048),
        )
        self.assertEqual(best["batch_size"], 4)
        self.assertTrue(reasons)

    def test_recommend_falls_back_when_no_safe_record_exists(self) -> None:
        records = [
            {
                "backend": "pytorch",
                "batch_size": 1,
                "thread_count": 1,
                "latency_p95_ms": 200,
                "throughput": 10,
                "peak_rss_mb": 500,
                "memory_budget_exceeded": False,
                "cpu_quota_exceeded": False,
            }
        ]
        best, reasons = recommend_config(records, RecommendationRequest(objective="latency", latency_budget_ms=20))
        self.assertEqual(best["backend"], "pytorch")
        self.assertIn("fallback", reasons[0])


if __name__ == "__main__":
    unittest.main()

