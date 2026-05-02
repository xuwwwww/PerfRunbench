from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.gpu.nvidia_tuner import (
    apply_nvidia_tuning,
    recommend_nvidia_tuning,
    restore_nvidia_tuning,
    snapshot_nvidia,
)


class NvidiaTunerTest(unittest.TestCase):
    @patch("autotune.gpu.nvidia_tuner.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_snapshot_nvidia_parses_gpu_rows(self, _which) -> None:
        def runner(command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="0, RTX 4090, Disabled, 300, 100, 450, 1200, 10000\n",
                stderr="",
            )

        snapshot = snapshot_nvidia(runner=runner)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["gpus"][0]["name"], "RTX 4090")
        self.assertEqual(snapshot["gpus"][0]["power.max_limit"], "450")

    @patch("autotune.gpu.nvidia_tuner.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_recommend_nvidia_tuning_reports_planned_changes(self, _which) -> None:
        result = recommend_nvidia_tuning(runner=lambda command: subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
        self.assertTrue(result["supported"])
        self.assertTrue(any(item["key"] == "power.limit" for item in result["planned_changes"]))

    @patch("autotune.gpu.nvidia_tuner.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_apply_and_restore_nvidia_tuning_write_snapshots(self, _which) -> None:
        commands = []

        def runner(command):
            commands.append(command)
            if "--query-gpu=index,name,persistence_mode,power.limit,power.min_limit,power.max_limit,clocks.current.graphics,clocks.current.memory" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="0, RTX 4090, Disabled, 300, 100, 450, 1200, 10000\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, result = apply_nvidia_tuning("nvidia-throughput", runner=runner, runs_dir=Path(temp_dir))
            restored = restore_nvidia_tuning(run_dir, runner=runner)
            self.assertTrue((run_dir / "gpu_tuning_before.json").exists())
            self.assertTrue((run_dir / "gpu_tuning_after.json").exists())
            self.assertTrue((run_dir / "gpu_tuning_diff.json").exists())

        self.assertTrue(result["changes"])
        self.assertTrue(restored)
        self.assertTrue(any("-pm" in command for command in commands))
        self.assertTrue(any("-pl" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
