from __future__ import annotations

import unittest
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.affinity import filter_thread_budget
from autotune.tuner.search_space import InferenceConfig


class ResourceBudgetTest(unittest.TestCase):
    def test_allowed_threads_respects_reserve_and_quota(self) -> None:
        budget = ResourceBudget(memory_budget_gb=22, reserve_cores=1, cpu_quota_percent=90)
        self.assertEqual(budget.allowed_threads(8), 7)
        self.assertEqual(budget.allowed_threads(16), 14)

    def test_effective_memory_budget_respects_system_total(self) -> None:
        budget = ResourceBudget(memory_budget_gb=22, reserve_memory_gb=1.7)
        self.assertAlmostEqual(budget.effective_memory_budget_mb(24268.8), 22528.0, places=1)
        self.assertAlmostEqual(budget.effective_memory_budget_mb(12000.0), 10259.2, places=1)

    def test_negative_memory_budget_means_keep_free_memory(self) -> None:
        budget = ResourceBudget(memory_budget_gb=-2)
        self.assertEqual(budget.memory_budget_mode, "reserve_to_full")
        self.assertAlmostEqual(budget.effective_memory_budget_mb(12000.0), 9952.0, places=1)
        self.assertIsNone(budget.effective_memory_budget_mb(None))

    def test_negative_memory_budget_combines_with_reserve_memory_by_stricter_limit(self) -> None:
        budget = ResourceBudget(memory_budget_gb=-2, reserve_memory_gb=3)
        self.assertAlmostEqual(budget.effective_memory_budget_mb(12000.0), 8928.0, places=1)

    def test_to_record_includes_negative_memory_budget_mode(self) -> None:
        record = ResourceBudget(memory_budget_gb=-1).to_record(total_memory_mb=12000.0)
        self.assertEqual(record["memory_budget_mode"], "reserve_to_full")
        self.assertEqual(record["memory_budget_gb"], -1)
        self.assertAlmostEqual(record["effective_memory_budget_mb"], 10976.0, places=1)

    def test_filter_thread_budget_removes_oversized_configs(self) -> None:
        budget = ResourceBudget(reserve_cores=10, cpu_quota_percent=50)
        configs = [
            InferenceConfig("pytorch", 1, 1, "fp32", "disable"),
            InferenceConfig("pytorch", 1, 128, "fp32", "disable"),
        ]
        with patch("autotune.resource.affinity.get_logical_cpu_count", return_value=8):
            filtered = filter_thread_budget(configs, budget)
        self.assertEqual(filtered, configs[:1])


if __name__ == "__main__":
    unittest.main()
