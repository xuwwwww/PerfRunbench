from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples"))

from training_workload import summarize_step_latencies


class WorkloadMetricsTest(unittest.TestCase):
    def test_summarize_step_latencies_reports_percentiles(self) -> None:
        summary = summarize_step_latencies([0.01, 0.02, 0.03, 0.04, 0.05])

        self.assertEqual(summary["step_time_sample_count"], 5)
        self.assertEqual(summary["step_time_sample_mean_seconds"], 0.03)
        self.assertEqual(summary["step_time_p50_seconds"], 0.03)
        self.assertEqual(summary["step_time_p95_seconds"], 0.048)
        self.assertEqual(summary["step_time_p99_seconds"], 0.0496)
        self.assertEqual(summary["step_time_max_seconds"], 0.05)

    def test_summarize_step_latencies_handles_empty_samples(self) -> None:
        summary = summarize_step_latencies([])

        self.assertEqual(summary["step_time_sample_count"], 0)
        self.assertIsNone(summary["step_time_p95_seconds"])


if __name__ == "__main__":
    unittest.main()
