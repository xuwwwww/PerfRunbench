from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.comparison_runner import build_comparison_result, compare_profiles, compare_tuning
from autotune.resource.run_state import write_json


class ComparisonRunnerTest(unittest.TestCase):
    def test_build_comparison_result_reports_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            self._write_run(runs_dir, "baseline", peak=100, started="2026-05-02T10:00:00", finished="2026-05-02T10:00:10")
            self._write_run(runs_dir, "tuned", peak=80, started="2026-05-02T10:01:00", finished="2026-05-02T10:01:08")

            result = build_comparison_result("baseline", "tuned", tuned_profile="linux-throughput", runs_dir=runs_dir)

        self.assertEqual(result["deltas"]["duration_seconds"], -2.0)
        self.assertEqual(result["deltas"]["duration_percent"], -20.0)
        self.assertEqual(result["deltas"]["benchmark_duration_seconds"], -2.0)
        self.assertEqual(result["deltas"]["workload_duration_seconds"], -2.0)
        self.assertEqual(result["baseline"]["system_tuning_overhead_seconds"], 0.0)
        self.assertEqual(result["deltas"]["peak_memory_mb"], -20)
        self.assertNotIn("final_accuracy", result["baseline"]["workload"])
        self.assertNotIn("final_accuracy", result["deltas"]["workload"])
        self.assertEqual(result["tuned_profile"], "linux-throughput")

    def test_build_comparison_result_subtracts_system_tuning_apply_restore_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            self._write_run(
                runs_dir,
                "baseline",
                peak=100,
                started="2026-05-02T10:00:00",
                finished="2026-05-02T10:00:10",
                workload_metrics=False,
            )
            self._write_run(
                runs_dir,
                "tuned",
                peak=80,
                started="2026-05-02T10:01:00",
                finished="2026-05-02T10:01:13",
                notes=[
                    "selected_executor=local",
                    "system_tuning_apply_seconds=2.5",
                    "system_tuning_restore_seconds=1.5",
                ],
                workload_metrics=False,
            )

            result = build_comparison_result("baseline", "tuned", tuned_profile="linux-throughput", runs_dir=runs_dir)

        self.assertEqual(result["tuned"]["lifecycle_duration_seconds"], 13.0)
        self.assertEqual(result["tuned"]["adjusted_lifecycle_duration_seconds"], 9.0)
        self.assertEqual(result["tuned"]["benchmark_duration_seconds"], 9.0)
        self.assertEqual(result["deltas"]["benchmark_duration_seconds"], -1.0)

    def test_compare_tuning_raises_when_workload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            baseline_dir = runs_dir / "baseline"
            tuned_dir = runs_dir / "tuned"
            self._write_run(
                runs_dir,
                "baseline",
                peak=0,
                started="2026-05-02T10:00:00",
                finished="2026-05-02T10:00:01",
                status="failed",
                return_code=2,
            )
            self._write_run(
                runs_dir,
                "tuned",
                peak=0,
                started="2026-05-02T10:01:00",
                finished="2026-05-02T10:01:01",
                status="failed",
                return_code=2,
            )
            output = runs_dir / "comparison.json"

            with (
                patch("autotune.resource.comparison_runner.run_with_budget", side_effect=[(2, baseline_dir), (2, tuned_dir)]),
                patch("autotune.resource.comparison_runner.RUNS_DIR", runs_dir),
            ):
                with self.assertRaisesRegex(RuntimeError, "compare-tuning workload failed"):
                    compare_tuning(
                        ["python", "missing_train.py"],
                        ResourceBudget(),
                        tuned_profile="linux-throughput",
                        output=output,
                    )

            self.assertTrue(output.exists())

    def test_compare_tuning_raises_when_any_repeat_trial_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            b1_dir = runs_dir / "b1"
            t1_dir = runs_dir / "t1"
            b2_dir = runs_dir / "b2"
            t2_dir = runs_dir / "t2"
            self._write_run(
                runs_dir,
                "b1",
                peak=0,
                started="2026-05-02T10:00:00",
                finished="2026-05-02T10:00:01",
                status="failed",
                return_code=2,
            )
            self._write_run(
                runs_dir,
                "t1",
                peak=0,
                started="2026-05-02T10:01:00",
                finished="2026-05-02T10:01:01",
            )
            self._write_run(
                runs_dir,
                "b2",
                peak=100,
                started="2026-05-02T10:02:00",
                finished="2026-05-02T10:02:10",
            )
            self._write_run(
                runs_dir,
                "t2",
                peak=80,
                started="2026-05-02T10:03:00",
                finished="2026-05-02T10:03:08",
            )
            output = runs_dir / "comparison.json"

            with (
                patch(
                    "autotune.resource.comparison_runner.run_with_budget",
                    side_effect=[(2, b1_dir), (0, t1_dir), (0, b2_dir), (0, t2_dir)],
                ),
                patch("autotune.resource.comparison_runner.RUNS_DIR", runs_dir),
            ):
                with self.assertRaisesRegex(RuntimeError, "trial1.baseline"):
                    compare_tuning(
                        ["python", "train.py"],
                        ResourceBudget(),
                        tuned_profile="linux-throughput",
                        output=output,
                        repeat=2,
                    )

            self.assertTrue(output.exists())

    def test_compare_tuning_alternates_execution_order_between_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            for run_id in ["b1", "t1", "b2", "t2"]:
                self._write_run(
                    runs_dir,
                    run_id,
                    peak=100 if run_id.startswith("b") else 80,
                    started="2026-05-02T10:00:00",
                    finished="2026-05-02T10:00:10" if run_id.startswith("b") else "2026-05-02T10:00:08",
                )
            output = runs_dir / "comparison.json"

            with (
                patch(
                    "autotune.resource.comparison_runner.run_with_budget",
                    side_effect=[(0, runs_dir / "b1"), (0, runs_dir / "t1"), (0, runs_dir / "t2"), (0, runs_dir / "b2")],
                ),
                patch("autotune.resource.comparison_runner.RUNS_DIR", runs_dir),
            ):
                result = compare_tuning(
                    ["python", "train.py"],
                    ResourceBudget(),
                    tuned_profile="linux-throughput",
                    output=output,
                    repeat=2,
                )

        self.assertEqual(result["trials"][0]["execution_order"], ["baseline", "tuned"])
        self.assertEqual(result["trials"][1]["execution_order"], ["tuned", "baseline"])

    def test_compare_profiles_summarizes_nested_workload_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.json"
            fake_result = {
                "aggregate": {
                    "deltas": {
                        "benchmark_duration_percent": -1.0,
                        "peak_memory_percent": -2.0,
                        "system_tuning_overhead_seconds": 0.1,
                        "workload": {
                            "samples_per_second": {"percent": 3.5},
                        },
                    }
                }
            }
            with patch("autotune.resource.comparison_runner.compare_tuning", return_value=fake_result):
                result = compare_profiles(
                    ["python", "train.py"],
                    ResourceBudget(),
                    profiles=["linux-throughput"],
                    output=output,
                    repeat=1,
                )

        self.assertEqual(result["best_profile"], "linux-throughput")
        self.assertEqual(result["comparisons"][0]["samples_per_second_percent"], 3.5)

    def _write_run(
        self,
        runs_dir: Path,
        run_id: str,
        *,
        peak: int,
        started: str,
        finished: str,
        status: str = "completed",
        return_code: int = 0,
        notes: list[str] | None = None,
        workload_metrics: bool = True,
    ) -> None:
        run_dir = runs_dir / run_id
        run_dir.mkdir()
        write_json(
            run_dir / "manifest.json",
            {
                "run_id": run_id,
                "status": status,
                "return_code": return_code,
                "started_at": started,
                "finished_at": finished,
                "command": ["python", "train.py"],
                "budget": {},
                "notes": notes or ["selected_executor=local"],
            },
        )
        write_json(run_dir / "resource_summary.json", {"peak_rss_mb": peak, "memory_budget_exceeded": False})
        write_json(run_dir / "resource_timeline.json", [{"available_memory_mb": 1000, "system_cpu_percent": 10}])
        if workload_metrics:
            write_json(
                run_dir / "training_metrics.json",
                {
                    "duration_seconds": 10 if run_id == "baseline" else 8,
                    "epoch_time_mean_seconds": 2.0 if run_id == "baseline" else 1.5,
                    "step_time_mean_seconds": 0.5 if run_id == "baseline" else 0.4,
                    "samples_per_second": 100.0 if run_id == "baseline" else 125.0,
                    "final_accuracy": 0.9,
                    "final_loss": 0.2 if run_id == "baseline" else 0.18,
                    "peak_batch_payload_mb": 3.0 if run_id == "baseline" else 2.5,
                },
            )


if __name__ == "__main__":
    unittest.main()
