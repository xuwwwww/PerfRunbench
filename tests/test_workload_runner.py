from __future__ import annotations

import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest
from autotune.resource.workload_runner import ChildSample, _summarize_timeline, run_with_budget


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

    def test_summary_includes_cgroup_fields_when_samples_have_them(self) -> None:
        summary = _summarize_timeline(
            [
                ChildSample(
                    timestamp=1.0,
                    rss_mb=100,
                    child_rss_mb=5,
                    available_memory_mb=900,
                    child_cpu_percent=1,
                    system_cpu_percent=10,
                    cgroup_path="/sys/fs/cgroup/demo.scope",
                    cgroup_memory_current_mb=100,
                    cgroup_memory_peak_mb=120,
                    cgroup_cpu_percent=50,
                    cgroup_cpu_usage_usec=1_000_000,
                ),
                ChildSample(
                    timestamp=2.0,
                    rss_mb=150,
                    child_rss_mb=5,
                    available_memory_mb=850,
                    child_cpu_percent=2,
                    system_cpu_percent=12,
                    cgroup_path="/sys/fs/cgroup/demo.scope",
                    cgroup_memory_current_mb=150,
                    cgroup_memory_peak_mb=180,
                    cgroup_cpu_percent=75,
                    cgroup_cpu_usage_usec=1_750_000,
                ),
            ],
            ResourceBudget(memory_budget_gb=1),
        )

        self.assertEqual(summary["peak_cgroup_memory_current_mb"], 150)
        self.assertEqual(summary["peak_cgroup_memory_peak_mb"], 180)
        self.assertEqual(summary["average_cgroup_cpu_percent"], 62.5)
        self.assertEqual(summary["peak_cgroup_cpu_percent"], 75)
        self.assertEqual(summary["cgroup_path"], "/sys/fs/cgroup/demo.scope")


if __name__ == "__main__":
    unittest.main()
