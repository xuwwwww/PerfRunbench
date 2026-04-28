from __future__ import annotations

import unittest

from autotune.profiler.benchmark_runner import synthetic_profile
from autotune.tuner.exhaustive_search import run_exhaustive_search
from autotune.tuner.objective import Objective
from autotune.tuner.search_space import InferenceConfig


class TunerTest(unittest.TestCase):
    def test_exhaustive_search_returns_feasible_best(self) -> None:
        configs = [
            InferenceConfig("pytorch", 1, 1, "fp32", "disable"),
            InferenceConfig("onnxruntime", 1, 4, "int8", "all"),
        ]
        objective = Objective(name="latency", latency_budget_ms=30, memory_budget_mb=2048)
        best, records = run_exhaustive_search(configs, synthetic_profile, objective)
        self.assertEqual(len(records), 2)
        self.assertEqual(best["backend"], "onnxruntime")


if __name__ == "__main__":
    unittest.main()

