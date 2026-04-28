from __future__ import annotations

import unittest

from autotune.scheduler.deadline_aware import schedule_deadline_aware
from autotune.scheduler.simulator import summarize
from autotune.scheduler.workload_generator import generate_workload


class SchedulerTest(unittest.TestCase):
    def test_deadline_aware_completes_all_requests(self) -> None:
        requests = generate_workload("medium", count=20, seed=1)
        completed = schedule_deadline_aware(requests)
        metrics = summarize(completed)
        self.assertEqual(len(completed), len(requests))
        self.assertIn("p95_latency", metrics)


if __name__ == "__main__":
    unittest.main()

