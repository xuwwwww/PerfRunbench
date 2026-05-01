from __future__ import annotations

import unittest
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.systemd_executor import build_systemd_run_command, preflight_systemd_executor


class SystemdExecutorTest(unittest.TestCase):
    @patch("autotune.resource.systemd_executor.shutil.which")
    def test_build_systemd_run_command_with_limits(self, which) -> None:
        which.side_effect = lambda name: f"/usr/bin/{name}"
        command = build_systemd_run_command(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=2, cpu_quota_percent=90),
        )
        self.assertEqual(command.command[:3], ["systemd-run", "--scope", "--quiet"])
        self.assertIn("MemoryMax=2048M", command.command)
        self.assertIn("CPUQuota=90%", command.command)
        self.assertEqual(command.command[-2:], ["python", "train.py"])

    @patch("autotune.resource.systemd_executor.shutil.which")
    def test_build_systemd_run_command_with_sudo_keeps_user(self, which) -> None:
        which.side_effect = lambda name: f"/usr/bin/{name}"
        command = build_systemd_run_command(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=1),
            use_sudo=True,
            run_as_user="alice",
        )
        self.assertEqual(command.command[0], "sudo")
        self.assertIn("--uid", command.command)
        self.assertIn("alice", command.command)

    @patch("autotune.resource.systemd_executor.read_systemd_state")
    @patch("autotune.resource.systemd_executor.shutil.which")
    def test_preflight_reports_missing_systemd_run(self, which, read_state) -> None:
        which.return_value = None
        read_state.return_value = None
        result = preflight_systemd_executor(["python", "train.py"], ResourceBudget(memory_budget_gb=1))
        self.assertFalse(result.runnable)
        self.assertTrue(any("systemd-run" in error for error in result.errors))

    @patch("autotune.resource.systemd_executor.sudo_credential_cached")
    @patch("autotune.resource.systemd_executor.probe_systemd_scope")
    @patch("autotune.resource.systemd_executor.read_systemd_state")
    @patch("autotune.resource.systemd_executor.shutil.which")
    def test_preflight_warns_when_sudo_not_cached(self, which, read_state, probe, sudo_cached) -> None:
        which.side_effect = lambda name: f"/usr/bin/{name}"
        read_state.return_value = "running"
        probe.return_value = (True, "")
        sudo_cached.return_value = False
        result = preflight_systemd_executor(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=1, cpu_quota_percent=50),
            use_sudo=True,
            run_as_user="alice",
            check_sudo_cache=True,
            probe=True,
        )
        self.assertTrue(result.runnable)
        self.assertFalse(result.sudo_cached)
        self.assertTrue(result.probe_succeeded)
        self.assertTrue(any("sudo credential" in warning for warning in result.warnings))
        self.assertIn("MemoryMax=1024M", result.command_preview)

    @patch("autotune.resource.systemd_executor.probe_systemd_scope")
    @patch("autotune.resource.systemd_executor.read_systemd_state")
    @patch("autotune.resource.systemd_executor.shutil.which")
    def test_preflight_probe_failure_blocks_runnable(self, which, read_state, probe) -> None:
        which.side_effect = lambda name: f"/usr/bin/{name}"
        read_state.return_value = "running"
        probe.return_value = (False, "Interactive authentication required.")
        result = preflight_systemd_executor(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=1),
            probe=True,
        )
        self.assertFalse(result.runnable)
        self.assertFalse(result.probe_succeeded)
        self.assertTrue(any("probe failed" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
