from __future__ import annotations

import unittest
from unittest.mock import patch

from autotune.resource.executor_capabilities import collect_executor_capabilities, recommend_executor


class ExecutorCapabilitiesTest(unittest.TestCase):
    def test_recommend_systemd_when_hard_limits_available(self) -> None:
        result = recommend_executor(
            {
                "local": {"available": True},
                "systemd": {"available": True, "hard_memory_limit": True},
                "docker": {"available": False},
            }
        )
        self.assertEqual(result, "systemd")

    def test_recommend_local_when_only_local_is_available(self) -> None:
        result = recommend_executor(
            {
                "local": {"available": True},
                "systemd": {"available": False},
                "docker": {"available": False},
            }
        )
        self.assertEqual(result, "local")

    @patch("autotune.resource.executor_capabilities.importlib.util.find_spec")
    @patch("autotune.resource.executor_capabilities.probe_systemd_scope")
    @patch("autotune.resource.executor_capabilities._path_exists")
    @patch("autotune.resource.executor_capabilities.is_wsl")
    @patch("autotune.resource.executor_capabilities.read_systemd_state")
    @patch("autotune.resource.executor_capabilities.shutil.which")
    @patch("autotune.resource.executor_capabilities.platform.system")
    def test_collect_linux_systemd_capabilities(
        self,
        system,
        which,
        read_state,
        is_wsl,
        path_exists,
        probe_systemd,
        find_spec,
    ) -> None:
        system.return_value = "Linux"
        which.side_effect = lambda name: f"/usr/bin/{name}" if name in {"systemd-run", "systemctl", "sudo"} else None
        read_state.return_value = "running"
        is_wsl.return_value = True
        path_exists.return_value = True
        probe_systemd.return_value = (False, "Interactive authentication required.")
        find_spec.return_value = object()

        result = collect_executor_capabilities(probe_systemd=True)

        self.assertEqual(result["platform"], "linux-wsl")
        self.assertEqual(result["recommended_executor"], "systemd")
        self.assertTrue(result["executors"]["local"]["available"])
        self.assertTrue(result["executors"]["systemd"]["available"])
        self.assertTrue(result["executors"]["systemd"]["cgroup_monitoring"])
        self.assertTrue(result["executors"]["systemd"]["requires_sudo"])
        self.assertFalse(result["executors"]["systemd"]["transient_scope_without_sudo"])
        self.assertTrue(result["executors"]["docker"]["implemented"])

    @patch("autotune.resource.executor_capabilities.importlib.util.find_spec")
    @patch("autotune.resource.executor_capabilities.is_wsl")
    @patch("autotune.resource.executor_capabilities.shutil.which")
    @patch("autotune.resource.executor_capabilities.platform.system")
    def test_collect_windows_marks_job_object_planned(self, system, which, is_wsl, find_spec) -> None:
        system.return_value = "Windows"
        which.return_value = None
        is_wsl.return_value = False
        find_spec.return_value = object()

        result = collect_executor_capabilities()

        self.assertEqual(result["platform"], "windows")
        self.assertEqual(result["recommended_executor"], "local")
        self.assertTrue(result["executors"]["windows_job"]["platform_supported"])
        self.assertFalse(result["executors"]["windows_job"]["implemented"])


if __name__ == "__main__":
    unittest.main()
