from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.run_analysis import analyze_run, format_analysis
from autotune.resource.run_state import write_json


class RunAnalysisTest(unittest.TestCase):
    def test_analyze_run_reports_cpu_and_memory_diagnostics(self) -> None:
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
                    "budget": {
                        "reserve_cores": 1,
                        "allowed_threads": 7,
                        "cpu_quota_percent": None,
                        "memory_budget_gb": -5,
                        "memory_budget_mode": "reserve_to_full",
                        "effective_memory_budget_mb": 7000,
                    },
                    "notes": [
                        "requested_executor=local",
                        "selected_executor=local",
                        "executor_platform=linux-wsl",
                        "sudo_used=False",
                        "affinity_context={'cpu_affinity_applied': True, 'logical_cpu_count': 8, 'allowed_threads': 7, 'affinity_cores': '0,1,2,3,4,5,6'}",
                    ],
                },
            )
            write_json(
                run_dir / "resource_summary.json",
                {
                    "peak_rss_mb": 3000,
                    "effective_memory_budget_mb": 7000,
                    "available_memory_after_mb": 5000,
                    "peak_child_cpu_percent": 88,
                    "memory_budget_exceeded": False,
                },
            )
            write_json(
                run_dir / "resource_timeline.json",
                [
                    {"available_memory_mb": 8000, "system_cpu_percent": 1},
                    {"available_memory_mb": 4900, "system_cpu_percent": 85},
                ],
            )

            analysis = analyze_run("run1", runs_dir)

        self.assertEqual(analysis["cpu"]["expected_max_total_cpu_percent"], 87.5)
        self.assertEqual(analysis["cpu"]["affinity_cores"], "0,1,2,3,4,5,6")
        self.assertEqual(analysis["memory"]["requested_reserve_to_full_gb"], 5)
        self.assertFalse(analysis["memory"]["memory_budget_exceeded"])
        self.assertTrue(any("WSL" in item or "Linux" in item for item in analysis["diagnostics"]))

    def test_format_analysis_includes_main_sections(self) -> None:
        analysis = {
            "run_id": "run1",
            "status": "completed",
            "return_code": 0,
            "executor": {"requested": "local", "selected": "local", "sudo_used": False},
            "cpu": {},
            "memory": {},
            "cgroup": {},
            "diagnostics": ["ok"],
        }
        output = format_analysis(analysis)
        self.assertIn("Executor", output)
        self.assertIn("CPU", output)
        self.assertIn("Memory", output)
        self.assertIn("Diagnostics", output)


if __name__ == "__main__":
    unittest.main()
