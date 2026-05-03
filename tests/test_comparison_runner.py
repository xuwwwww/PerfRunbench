from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.comparison_runner import build_comparison_result, compare_tuning
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
        self.assertEqual(result["deltas"]["workload_duration_seconds"], -2.0)
        self.assertEqual(result["baseline"]["system_tuning_overhead_seconds"], 0.0)
        self.assertEqual(result["deltas"]["peak_memory_mb"], -20)
        self.assertEqual(result["deltas"]["workload"]["final_accuracy"]["absolute"], 0.0)
        self.assertEqual(result["tuned_profile"], "linux-throughput")

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
                "notes": ["selected_executor=local"],
            },
        )
        write_json(run_dir / "resource_summary.json", {"peak_rss_mb": peak, "memory_budget_exceeded": False})
        write_json(run_dir / "resource_timeline.json", [{"available_memory_mb": 1000, "system_cpu_percent": 10}])
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
