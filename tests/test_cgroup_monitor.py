from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotune.resource.cgroup_monitor import cgroup_path, read_cgroup_stats, read_systemd_control_group


class CgroupMonitorTest(unittest.TestCase):
    def test_cgroup_path_resolves_under_root(self) -> None:
        root = Path("/tmp/cgroup")
        self.assertEqual(cgroup_path("/user.slice/demo.scope", root), root / "user.slice" / "demo.scope")

    def test_read_cgroup_stats_reads_memory_and_cpu_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scope = root / "user.slice" / "autotuneai.scope"
            scope.mkdir(parents=True)
            (scope / "memory.current").write_text(str(128 * 1024 * 1024), encoding="utf-8")
            (scope / "memory.peak").write_text(str(256 * 1024 * 1024), encoding="utf-8")
            (scope / "cpu.stat").write_text(
                "usage_usec 1000000\nuser_usec 700000\nsystem_usec 300000\n",
                encoding="utf-8",
            )

            stats = read_cgroup_stats("/user.slice/autotuneai.scope", root)

        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.memory_current_mb, 128)
        self.assertEqual(stats.memory_peak_mb, 256)
        self.assertEqual(stats.cpu_usage_usec, 1_000_000)
        self.assertEqual(stats.cpu_user_usec, 700_000)
        self.assertEqual(stats.cpu_system_usec, 300_000)

    @patch("autotune.resource.cgroup_monitor.subprocess.run")
    def test_read_systemd_control_group_uses_systemctl_show(self, run) -> None:
        run.return_value.stdout = "/user.slice/autotuneai.scope\n"
        result = read_systemd_control_group("autotuneai.scope")
        self.assertEqual(result, "/user.slice/autotuneai.scope")
        self.assertEqual(run.call_args.args[0][:3], ["systemctl", "show", "autotuneai.scope"])


if __name__ == "__main__":
    unittest.main()
