from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.system_tuner.runtime import (
    RuntimeSetting,
    SettingSnapshot,
    _dynamic_profile_settings,
    apply_system_tuning,
    available_profiles,
    recommend_system_tuning,
    restore_system_tuning,
)


class SystemTunerTest(unittest.TestCase):
    @patch("autotune.system_tuner.runtime.platform.system")
    def test_recommend_system_tuning_reports_unsupported_profile_platform(self, system) -> None:
        system.return_value = "Windows"
        result = recommend_system_tuning("linux-training-safe")
        self.assertFalse(result["supported"])
        self.assertTrue(any("not supported on Windows" in note for note in result["notes"]))

    @patch("autotune.system_tuner.runtime.platform.system")
    @patch("autotune.system_tuner.runtime.read_setting")
    def test_recommend_system_tuning_supports_windows_profiles(self, read_setting, system) -> None:
        system.return_value = "Windows"
        read_setting.return_value = SettingSnapshot(
            key="power.active_scheme",
            value="381b4222-f694-41f0-9685-ff5bb260df2e",
            exists=True,
            source="powercfg",
            path="powercfg://active-scheme",
        )
        result = recommend_system_tuning("windows-throughput")
        self.assertTrue(result["supported"])
        self.assertEqual(result["settings"][0]["source"], "powercfg")
        self.assertEqual(result["settings"][0]["target"], "SCHEME_MIN")

    @patch("autotune.system_tuner.runtime._run_command")
    def test_read_windows_power_scheme_parses_powercfg_output(self, run_command) -> None:
        from autotune.system_tuner.runtime import read_setting

        run_command.return_value = subprocess.CompletedProcess(
            ["powercfg", "/getactivescheme"],
            0,
            stdout="Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n",
            stderr="",
        )
        snapshot = read_setting(
            RuntimeSetting(
                key="power.active_scheme",
                value="SCHEME_MIN",
                reason="test",
                source="powercfg",
                path="powercfg://active-scheme",
            )
        )
        self.assertTrue(snapshot.exists)
        self.assertEqual(snapshot.value, "381b4222-f694-41f0-9685-ff5bb260df2e")

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

    @patch("autotune.system_tuner.runtime.platform.system")
    @patch("autotune.system_tuner.runtime.read_setting")
    def test_apply_windows_system_tuning_uses_powercfg(self, read_setting, system) -> None:
        system.return_value = "Windows"
        values = {"power.active_scheme": "381b4222-f694-41f0-9685-ff5bb260df2e"}

        def fake_read(setting: RuntimeSetting | str) -> SettingSnapshot:
            key = setting.key if isinstance(setting, RuntimeSetting) else setting
            source = setting.source if isinstance(setting, RuntimeSetting) else "powercfg"
            path = setting.path if isinstance(setting, RuntimeSetting) else "powercfg://active-scheme"
            return SettingSnapshot(key=key, value=values.get(key), exists=True, source=source, path=path)

        def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[:2], ["powercfg", "/setactive"])
            values["power.active_scheme"] = command[2]
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        read_setting.side_effect = fake_read
        with tempfile.TemporaryDirectory() as temp_dir:
            _run_dir, result = apply_system_tuning(
                "windows-throughput",
                runner=fake_runner,
                runs_dir=Path(temp_dir),
            )

        self.assertEqual(result["changes"][0]["before"], "381b4222-f694-41f0-9685-ff5bb260df2e")
        self.assertEqual(result["changes"][0]["after"], "SCHEME_MIN")

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
        self.assertIn("linux-performance", profiles)
        self.assertIn("linux-extreme-throughput", profiles)
        self.assertIn("linux-low-latency", profiles)
        self.assertIn("linux-cpu-conservative", profiles)
        self.assertIn("windows-training-safe", profiles)
        self.assertIn("windows-memory-conservative", profiles)
        self.assertIn("windows-throughput", profiles)
        self.assertIn("windows-performance", profiles)
        self.assertIn("windows-low-latency", profiles)
        self.assertIn("windows-cpu-conservative", profiles)

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

    def test_linux_performance_discovers_cpufreq_runtime_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = Path(temp_dir) / "policy0"
            policy.mkdir()
            (policy / "scaling_governor").write_text("powersave\n", encoding="utf-8")
            (policy / "scaling_available_governors").write_text("powersave performance\n", encoding="utf-8")
            (policy / "energy_performance_preference").write_text("balance_power\n", encoding="utf-8")
            (policy / "energy_performance_available_preferences").write_text("balance_power performance\n", encoding="utf-8")
            (policy / "scaling_min_freq").write_text("400000\n", encoding="utf-8")
            (policy / "cpuinfo_max_freq").write_text("3800000\n", encoding="utf-8")

            settings = _dynamic_profile_settings("linux-extreme-throughput", cpufreq_base=Path(temp_dir))

        keys = {setting.key: setting.value for setting in settings}
        self.assertEqual(keys["cpu.cpufreq.policy0.scaling_governor"], "performance")
        self.assertEqual(keys["cpu.cpufreq.policy0.energy_performance_preference"], "performance")
        self.assertEqual(keys["cpu.cpufreq.policy0.scaling_min_freq"], "3800000")


if __name__ == "__main__":
    unittest.main()
