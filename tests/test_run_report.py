from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.report.run_report import generate_run_report
from autotune.report.comparison_report import generate_comparison_report
from autotune.resource.run_state import write_json


class RunReportTest(unittest.TestCase):
    def test_generate_run_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            write_json(
                run_dir / "manifest.json",
                {
                    "run_id": "run1",
                    "status": "completed",
                    "return_code": 0,
                    "command": ["python", "train.py"],
                    "budget": {"reserve_cores": 1, "allowed_threads": 3},
                    "notes": ["selected_executor=local"],
                },
            )
            write_json(
                run_dir / "resource_summary.json",
                {
                    "peak_rss_mb": 128,
                    "memory_budget_exceeded": False,
                    "peak_child_cpu_percent": 75,
                },
            )
            write_json(run_dir / "resource_timeline.json", [{"system_cpu_percent": 50, "available_memory_mb": 1024}])

            report_path = generate_run_report("run1", runs_dir=runs_dir)

            self.assertEqual(report_path, run_dir / "report.md")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("AutoTuneAI Run Report: run1", report)
            self.assertIn("## Before / After", report)
            self.assertIn("## Visual Summary", report)
            self.assertIn("<svg", report)
            self.assertIn("Available memory at start", report)
            self.assertIn("## CPU", report)
            self.assertIn("## Memory", report)
            self.assertIn("## Diagnostics", report)

    def test_generate_comparison_report_writes_visual_markdown(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "comparison.json"
            write_json(
                path,
                {
                    "tuned_profile": "linux-throughput",
                    "baseline": {"run_id": "b1", "benchmark_duration_seconds": 10, "lifecycle_duration_seconds": 11, "peak_memory_mb": 100},
                    "tuned": {"run_id": "t1", "benchmark_duration_seconds": 8, "lifecycle_duration_seconds": 9, "peak_memory_mb": 90},
                    "deltas": {"benchmark_duration_percent": -20, "peak_memory_percent": -10},
                },
            )

            report_path = generate_comparison_report(path)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("AutoTuneAI Tuning Comparison", report)
            self.assertIn("Performance Deltas", report)
            self.assertIn("<svg", report)


if __name__ == "__main__":
    unittest.main()
