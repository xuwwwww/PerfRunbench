from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.report.run_report import generate_run_report
from autotune.resource.run_state import write_json


class RunReportTest(unittest.TestCase):
    def test_generate_run_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
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
            self.assertIn("## CPU", report)
            self.assertIn("## Memory", report)
            self.assertIn("## Diagnostics", report)


if __name__ == "__main__":
    unittest.main()
