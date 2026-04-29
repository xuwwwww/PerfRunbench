from __future__ import annotations

import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest
from autotune.resource.workload_runner import run_with_budget


class WorkloadRunnerTest(unittest.TestCase):
    def test_run_with_budget_records_manifest_and_summary(self) -> None:
        command = ["python", "tests/fixtures/sleep_workload.py"]
        return_code, run_dir = run_with_budget(
            command,
            ResourceBudget(memory_budget_gb=1),
            sample_interval_seconds=0.05,
        )
        self.assertEqual(return_code, 0)
        self.assertTrue((run_dir / "manifest.json").exists())
        self.assertTrue((run_dir / "resource_summary.json").exists())
        self.assertTrue((run_dir / "resource_timeline.json").exists())
        manifest = load_manifest(Path(run_dir))
        self.assertEqual(manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()

