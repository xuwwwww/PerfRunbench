from __future__ import annotations

import unittest
from unittest.mock import patch

from autotune.resource.budget import ResourceBudget
from autotune.resource.docker_executor import build_docker_run_command


class DockerExecutorTest(unittest.TestCase):
    @patch("autotune.resource.docker_executor.shutil.which", return_value="/usr/bin/docker")
    def test_build_docker_run_command_applies_memory_and_cpu_limits(self, _which) -> None:
        command = build_docker_run_command(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=2, reserve_cores=1, cpu_quota_percent=50),
            image="example/train:latest",
            workdir=".",
            total_cores=8,
            total_memory_mb=24 * 1024,
        )

        self.assertEqual(command.command[0:3], ["docker", "run", "--rm"])
        self.assertIn("example/train:latest", command.command)
        self.assertIn("--memory", command.command)
        self.assertIn("2048m", command.command)
        self.assertIn("--memory-swap", command.command)
        self.assertIn("--cpus", command.command)
        self.assertIn("4", command.command)
        self.assertTrue(any("docker image=example/train:latest" in note for note in command.notes))

    @patch("autotune.resource.docker_executor.shutil.which", return_value="/usr/bin/docker")
    def test_build_docker_run_command_supports_negative_memory_budget(self, _which) -> None:
        command = build_docker_run_command(
            ["python", "train.py"],
            ResourceBudget(memory_budget_gb=-5),
            image="python:3.12-slim",
            workdir=".",
            total_cores=8,
            total_memory_mb=24 * 1024,
        )

        self.assertIn("19456m", command.command)


if __name__ == "__main__":
    unittest.main()
