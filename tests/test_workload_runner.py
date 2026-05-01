from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest
from autotune.resource.workload_runner import (
    ChildSample,
    _resolve_command_executable,
    _resolve_executor,
    _summarize_timeline,
    run_with_budget,
)


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

    @patch("autotune.resource.workload_runner.collect_executor_capabilities")
    def test_auto_executor_falls_back_to_local(self, collect_capabilities) -> None:
        collect_capabilities.return_value = {
            "platform": "windows",
            "recommended_executor": "local",
            "executors": {"local": {"available": True}},
        }
        selected, use_sudo, notes = _resolve_executor("auto", use_sudo=False, allow_sudo_auto=False)
        self.assertEqual(selected, "local")
        self.assertFalse(use_sudo)
        self.assertIn("selected_executor=local", notes)

    @patch("autotune.resource.workload_runner.collect_executor_capabilities")
    def test_auto_executor_requires_explicit_sudo_permission(self, collect_capabilities) -> None:
        collect_capabilities.return_value = {
            "platform": "linux-wsl",
            "recommended_executor": "systemd",
            "executors": {
                "systemd": {
                    "available": True,
                    "requires_sudo": True,
                    "sudo_available": True,
                    "sudo_cached": False,
                }
            },
        }
        with self.assertRaisesRegex(RuntimeError, "requires sudo"):
            _resolve_executor("auto", use_sudo=False, allow_sudo_auto=False)

    @patch("autotune.resource.workload_runner.collect_executor_capabilities")
    def test_auto_executor_can_use_sudo_when_allowed(self, collect_capabilities) -> None:
        collect_capabilities.return_value = {
            "platform": "linux-wsl",
            "recommended_executor": "systemd",
            "executors": {
                "systemd": {
                    "available": True,
                    "requires_sudo": True,
                    "sudo_available": True,
                    "sudo_cached": True,
                }
            },
        }
        selected, use_sudo, notes = _resolve_executor("auto", use_sudo=False, allow_sudo_auto=True)
        self.assertEqual(selected, "systemd")
        self.assertTrue(use_sudo)
        self.assertIn("sudo_used=True", notes)

    @patch("autotune.resource.workload_runner.shutil.which")
    def test_resolve_command_executable_for_systemd_environment(self, which) -> None:
        which.return_value = "/env/bin/python"
        self.assertEqual(
            _resolve_command_executable(["python", "train.py"]),
            ["/env/bin/python", "train.py"],
        )

    @patch("autotune.resource.workload_runner.shutil.which")
    def test_resolve_command_executable_leaves_explicit_paths(self, which) -> None:
        self.assertEqual(
            _resolve_command_executable(["/env/bin/python", "train.py"]),
            ["/env/bin/python", "train.py"],
        )
        which.assert_not_called()


if __name__ == "__main__":
    unittest.main()
