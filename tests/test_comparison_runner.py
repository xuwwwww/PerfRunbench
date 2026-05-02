from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.comparison_runner import build_comparison_result
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
        self.assertEqual(result["deltas"]["peak_memory_mb"], -20)
        self.assertEqual(result["tuned_profile"], "linux-throughput")

    def _write_run(self, runs_dir: Path, run_id: str, *, peak: int, started: str, finished: str) -> None:
        run_dir = runs_dir / run_id
        run_dir.mkdir()
        write_json(
            run_dir / "manifest.json",
            {
                "run_id": run_id,
                "status": "completed",
                "return_code": 0,
                "started_at": started,
                "finished_at": finished,
                "command": ["python", "train.py"],
                "budget": {},
                "notes": ["selected_executor=local"],
            },
        )
        write_json(run_dir / "resource_summary.json", {"peak_rss_mb": peak, "memory_budget_exceeded": False})
        write_json(run_dir / "resource_timeline.json", [{"available_memory_mb": 1000, "system_cpu_percent": 10}])


if __name__ == "__main__":
    unittest.main()
