from __future__ import annotations

import unittest
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.systemd_executor import build_systemd_run_command


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


if __name__ == "__main__":
    unittest.main()
