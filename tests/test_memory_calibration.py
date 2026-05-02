from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.memory_calibration import calibrate_memory
from autotune.resource.run_state import write_json


class MemoryCalibrationTest(unittest.TestCase):
    def test_calibrate_memory_writes_records_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            output = Path(temp_dir) / "calibration.json"
            counter = {"value": 0}

            def fake_runner(command, budget, **kwargs):
                counter["value"] += 1
                run_dir = runs_dir / f"run{counter['value']}"
                run_dir.mkdir(parents=True)
                write_json(
                    run_dir / "manifest.json",
                    {
                        "run_id": run_dir.name,
                        "status": "completed",
                        "return_code": 0,
                        "command": command,
                        "budget": {
                            "memory_budget_gb": budget.memory_budget_gb,
                            "memory_budget_mode": budget.memory_budget_mode,
                            "effective_memory_budget_mb": 20_000,
                        },
                        "notes": ["selected_executor=local"],
                    },
                )
                write_json(run_dir / "resource_summary.json", {"peak_rss_mb": 1024, "memory_budget_exceeded": False})
                write_json(run_dir / "resource_timeline.json", [{"available_memory_mb": 4096}])
                return 0, run_dir

            result = calibrate_memory(
                [-5.0],
                workload_memory_mb=512,
                duration_seconds=1,
                output=output,
                runs_dir=runs_dir,
                runner=fake_runner,
            )

            self.assertTrue(output.exists())
            self.assertEqual(result["records"][0]["requested_budget_mode"], "reserve_to_full")
            self.assertEqual(result["records"][0]["budget_utilization"], 0.0512)
            self.assertEqual(result["records"][0]["reserve_error_gb"], -1.0)
            self.assertTrue(any("did not approach" in item for item in result["recommendations"]))


if __name__ == "__main__":
    unittest.main()
