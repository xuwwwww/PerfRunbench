from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.recommendation.optimizer import optimize_recommendation
from autotune.resource.budget import ResourceBudget


class OptimizerTest(unittest.TestCase):
    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    def test_optimize_recommendation_caches_best_candidate(self, run_with_budget, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}

        run_dirs = [Path(f".autotuneai/runs/run{i}") for i in range(4)]
        run_with_budget.side_effect = [(0, path) for path in run_dirs]

        def fake_analyze(run_id: str, _runs_dir):
            index = int(run_id.replace("run", ""))
            throughput = [100.0, 120.0, 110.0, 90.0][index]
            return {
                "status": "completed",
                "return_code": 0,
                "workload": {"samples_per_second": throughput, "duration_seconds": 10.0},
                "memory": {"peak_memory_mb": 1000.0, "memory_budget_exceeded": False},
                "cpu": {"observed_peak_process_cpu_percent": 50.0},
            }

        analyze_run.side_effect = fake_analyze
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output = Path(temp_dir) / "recommendation.json"
            cache = Path(temp_dir) / "latest.json"
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=output,
                cache_path=cache,
                max_candidates=4,
            )
            self.assertTrue(output.exists())
            self.assertTrue(cache.exists())

        self.assertEqual(result["recommendation"]["metrics"]["samples_per_second"], 120.0)

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    def test_optimize_recommendation_discards_warmup_runs(self, run_with_budget, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        run_dirs = [Path(".autotuneai/runs/warmup0"), Path(".autotuneai/runs/run0")]
        run_with_budget.side_effect = [(0, path) for path in run_dirs]
        analyze_run.return_value = {
            "status": "completed",
            "return_code": 0,
            "workload": {"samples_per_second": 100.0, "duration_seconds": 10.0},
            "memory": {"peak_memory_mb": 1000.0, "memory_budget_exceeded": False},
            "cpu": {"observed_peak_process_cpu_percent": 50.0},
        }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                warmup_runs=1,
                max_candidates=1,
            )

        self.assertEqual(run_with_budget.call_count, 2)
        self.assertEqual(result["warmups"][0]["run_id"], "warmup0")
        self.assertEqual(result["candidates"][0]["run_ids"], ["run0"])


if __name__ == "__main__":
    unittest.main()
