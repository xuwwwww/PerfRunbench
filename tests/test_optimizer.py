from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.recommendation.optimizer import _candidate_plan, optimize_recommendation
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
                "workload": {
                    "samples_per_second": throughput,
                    "duration_seconds": 10.0,
                    "step_time_p95_seconds": [0.1, 0.08, 0.09, 0.12][index],
                },
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
        self.assertEqual(result["recommendation"]["metrics"]["step_time_p95_seconds"], 0.08)

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

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    def test_performance_mode_uses_unbounded_non_enforcing_budget(self, run_with_budget, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        run_with_budget.return_value = (0, Path(".autotuneai/runs/run0"))
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
                ResourceBudget(memory_budget_gb=-3, reserve_cores=2, cpu_quota_percent=80),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                max_candidates=1,
                optimization_mode="performance",
            )

        budget = run_with_budget.call_args.args[1]
        self.assertFalse(budget.enforce)
        self.assertFalse(budget.enabled)
        self.assertEqual(result["optimization_mode"], "performance")
        self.assertEqual(result["candidates"][0]["guard_mode"], "performance")

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.launch_performance")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    def test_performance_minimal_mode_uses_unmonitored_launcher(self, run_with_budget, launch_performance, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        launch_performance.return_value = (0, Path(".autotuneai/runs/run0"))
        analyze_run.return_value = {
            "status": "completed",
            "return_code": 0,
            "workload": {"samples_per_second": 100.0, "duration_seconds": 10.0},
            "memory": {"memory_budget_exceeded": False},
            "cpu": {},
        }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                max_candidates=1,
                optimization_mode="performance",
                monitor_mode="minimal",
            )

        launch_performance.assert_called_once()
        run_with_budget.assert_not_called()
        self.assertEqual(result["monitor_mode"], "minimal")

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    @patch("autotune.recommendation.optimizer._deadline_expired")
    def test_time_budget_flushes_partial_summary(self, deadline_expired, run_with_budget, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        deadline_expired.side_effect = [False, True, True]
        run_with_budget.return_value = (0, Path(".autotuneai/runs/run0"))
        analyze_run.return_value = {
            "status": "completed",
            "return_code": 0,
            "workload": {"samples_per_second": 100.0, "duration_seconds": 10.0},
            "memory": {"memory_budget_exceeded": False},
            "cpu": {},
        }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                max_candidates=3,
                time_budget_hours=8,
            )
            self.assertTrue((Path(temp_dir) / "recommendation.json").exists())

        self.assertGreaterEqual(len(result["candidates"]), 1)
        self.assertLess(len(result["candidates"]), 3)

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.run_with_budget")
    def test_repeats_use_rotated_interleaved_order(self, run_with_budget, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        run_dirs = [Path(f".autotuneai/runs/run{i}") for i in range(6)]
        run_with_budget.side_effect = [(0, path) for path in run_dirs]
        analyze_run.return_value = {
            "status": "completed",
            "return_code": 0,
            "workload": {"samples_per_second": 100.0, "duration_seconds": 10.0},
            "memory": {"memory_budget_exceeded": False},
            "cpu": {},
        }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                max_candidates=3,
                repeat=2,
            )

        labels = [item["label"] for item in result["execution_order"]]
        self.assertEqual(
            labels,
            [
                "unbounded:baseline",
                "unbounded:runtime-cpu",
                "unbounded:runtime-gpu",
                "unbounded:runtime-cpu",
                "unbounded:runtime-gpu",
                "unbounded:baseline",
            ],
        )
        self.assertEqual(result["schedule"], "interleaved-rotating")

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    @patch("autotune.recommendation.optimizer.analyze_run")
    @patch("autotune.recommendation.optimizer.launch_performance")
    def test_thermal_control_ranks_by_paired_baseline_ratio(self, launch_performance, analyze_run, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": False}
        launch_performance.side_effect = [
            (0, Path(".autotuneai/runs/baseline0")),
            (0, Path(".autotuneai/runs/candidate0")),
        ]

        def fake_analyze(run_id: str, _runs_dir):
            throughput = 100.0 if run_id == "baseline0" else 120.0
            return {
                "status": "completed",
                "return_code": 0,
                "workload": {"samples_per_second": throughput, "duration_seconds": 10.0},
                "memory": {"memory_budget_exceeded": False},
                "cpu": {},
            }

        analyze_run.side_effect = fake_analyze
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            result = optimize_recommendation(
                ["python", "train.py"],
                ResourceBudget(),
                output=Path(temp_dir) / "recommendation.json",
                cache_path=Path(temp_dir) / "latest.json",
                max_candidates=2,
                optimization_mode="performance",
                monitor_mode="minimal",
                thermal_control=True,
            )

        self.assertEqual(result["schedule"], "thermal-controlled-pairs")
        self.assertEqual(result["best_label"], "performance:runtime-cpu")
        self.assertEqual(result["recommendation"]["metrics"]["normalized_samples_per_second_ratio"], 1.2)

    @patch("autotune.recommendation.optimizer.platform.system", return_value="Linux")
    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    def test_candidate_plan_prioritizes_cpu_target(self, recommend_gpu, _system) -> None:
        recommend_gpu.return_value = {"supported": False}

        candidates = _candidate_plan(
            ResourceBudget(enforce=False),
            include_gpu=True,
            optimization_mode="performance",
            optimization_target="cpu",
        )

        self.assertEqual(
            [candidate.label for candidate in candidates[:4]],
            [
                "performance:baseline",
                "performance:runtime-cpu",
                "performance:linux-performance",
                "performance:linux-throughput",
            ],
        )

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    def test_guarded_candidate_plan_includes_gpu_guard_profiles(self, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}

        candidates = _candidate_plan(
            ResourceBudget(memory_budget_gb=-3),
            include_gpu=True,
            optimization_mode="guarded",
        )

        by_label = {candidate.label: candidate for candidate in candidates}
        self.assertEqual(by_label["unbounded:gpu-guard"].gpu_profile, "nvidia-guard")
        self.assertEqual(by_label["budgeted:gpu-guard"].gpu_profile, "nvidia-guard")
        self.assertEqual(by_label["unbounded:gpu-balanced"].gpu_profile, "nvidia-balanced")

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    def test_performance_candidate_plan_omits_guard_gpu_profiles(self, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}

        candidates = _candidate_plan(
            ResourceBudget(enforce=False),
            include_gpu=True,
            optimization_mode="performance",
        )

        gpu_profiles = {candidate.gpu_profile for candidate in candidates if candidate.gpu_profile}
        self.assertEqual(gpu_profiles, {"nvidia-performance"})

    @patch("autotune.recommendation.optimizer.recommend_nvidia_tuning")
    def test_guarded_gpu_target_prioritizes_gpu_guard_profiles(self, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}

        candidates = _candidate_plan(
            ResourceBudget(memory_budget_gb=-3),
            include_gpu=True,
            optimization_mode="guarded",
            optimization_target="gpu",
        )

        self.assertEqual(
            [candidate.label for candidate in candidates[:4]],
            ["unbounded:baseline", "budgeted:baseline", "unbounded:gpu-balanced", "unbounded:gpu-guard"],
        )


if __name__ == "__main__":
    unittest.main()
