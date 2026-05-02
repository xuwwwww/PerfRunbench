from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.system_tuner.runtime import (
    RuntimeSetting,
    SettingSnapshot,
    apply_system_tuning,
    available_profiles,
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
        values = {
            "vm.swappiness": "60",
            "kernel.numa_balancing": "1",
            "vm.dirty_background_ratio": "10",
            "vm.dirty_ratio": "20",
            "vm.zone_reclaim_mode": "0",
        }

        def fake_read(setting: RuntimeSetting | str) -> SettingSnapshot:
            key = setting.key if isinstance(setting, RuntimeSetting) else setting
            source = setting.source if isinstance(setting, RuntimeSetting) else "sysctl"
            path = setting.path if isinstance(setting, RuntimeSetting) else None
            return SettingSnapshot(key=key, value=values.get(key), exists=key in values, source=source, path=path)

        def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            if "sysctl" in command:
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

        def fake_read(setting: RuntimeSetting | str) -> SettingSnapshot:
            key = setting.key if isinstance(setting, RuntimeSetting) else setting
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

    def test_multiple_system_tuning_profiles_are_available(self) -> None:
        profiles = available_profiles()
        self.assertIn("linux-training-safe", profiles)
        self.assertIn("linux-memory-conservative", profiles)
        self.assertIn("linux-throughput", profiles)
        self.assertIn("linux-low-latency", profiles)

    @patch("autotune.system_tuner.runtime.platform.system")
    @patch("autotune.system_tuner.runtime.read_setting")
    def test_recommend_system_tuning_includes_sources_and_paths(self, read_setting, system) -> None:
        system.return_value = "Linux"

        def fake_read(setting: RuntimeSetting | str) -> SettingSnapshot:
            key = setting.key if isinstance(setting, RuntimeSetting) else setting
            source = setting.source if isinstance(setting, RuntimeSetting) else "sysctl"
            path = setting.path if isinstance(setting, RuntimeSetting) else None
            return SettingSnapshot(key=key, value="0", exists=True, source=source, path=path)

        read_setting.side_effect = fake_read
        result = recommend_system_tuning("linux-low-latency")

        self.assertTrue(all("source" in item for item in result["settings"]))
        self.assertTrue(all("path" in item for item in result["settings"]))
        self.assertTrue(any(item["source"] == "file" for item in result["settings"]))


if __name__ == "__main__":
    unittest.main()
