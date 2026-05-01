from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.system_tuner.runtime import (
    SettingSnapshot,
    apply_system_tuning,
    recommend_system_tuning,
    restore_system_tuning,
)


class SystemTunerTest(unittest.TestCase):
    @patch("autotune.system_tuner.runtime.platform.system")
    def test_recommend_system_tuning_reports_non_linux_unsupported(self, system) -> None:
        system.return_value = "Windows"
        result = recommend_system_tuning("linux-training-safe")
        self.assertFalse(result["supported"])
        self.assertTrue(any("Linux-only" in note for note in result["notes"]))

    @patch("autotune.system_tuner.runtime.platform.system")
    @patch("autotune.system_tuner.runtime.read_setting")
    def test_apply_system_tuning_writes_before_after_and_diff(self, read_setting, system) -> None:
        system.return_value = "Linux"
        values = {"vm.swappiness": "60", "kernel.numa_balancing": "1"}

        def fake_read(key: str) -> SettingSnapshot:
            return SettingSnapshot(key=key, value=values.get(key), exists=key in values)

        def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            assignment = command[-1]
            key, value = assignment.split("=", 1)
            values[key] = value
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        read_setting.side_effect = fake_read
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, result = apply_system_tuning(
                "linux-training-safe",
                runner=fake_runner,
                runs_dir=Path(temp_dir),
            )
            diff_path = run_dir / "system_tuning_diff.json"
            before_path = run_dir / "system_tuning_before.json"
            after_path = run_dir / "system_tuning_after.json"
            self.assertTrue(diff_path.exists())
            self.assertTrue(before_path.exists())
            self.assertTrue(after_path.exists())

        self.assertEqual(result["changes"][0]["before"], "60")
        self.assertEqual(result["changes"][0]["after"], "10")

    @patch("autotune.system_tuner.runtime.read_setting")
    def test_restore_system_tuning_reapplies_before_values(self, read_setting) -> None:
        values = {"vm.swappiness": "10"}

        def fake_read(key: str) -> SettingSnapshot:
            return SettingSnapshot(key=key, value=values.get(key), exists=key in values)

        def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            key, value = command[-1].split("=", 1)
            values[key] = value
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        read_setting.side_effect = fake_read
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "system_tuning_before.json").write_text(
                '[{"key": "vm.swappiness", "value": "60", "exists": true, "error": null}]',
                encoding="utf-8",
            )
            restored = restore_system_tuning(run_dir, runner=fake_runner)

        self.assertEqual(restored[0]["key"], "vm.swappiness")
        self.assertEqual(values["vm.swappiness"], "60")


if __name__ == "__main__":
    unittest.main()
