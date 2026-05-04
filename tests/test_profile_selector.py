from __future__ import annotations

import unittest

from autotune.resource.budget import ResourceBudget
from autotune.system_tuner.profile_selector import select_system_profile


class ProfileSelectorTest(unittest.TestCase):
    def test_auto_selects_memory_profile_for_memory_budget(self) -> None:
        selection = select_system_profile(ResourceBudget(memory_budget_gb=-3), workload_profile="auto", runtime_platform="Linux")
        self.assertEqual(selection.profile, "linux-memory-conservative")

    def test_auto_selects_cpu_conservative_for_cpu_quota_without_memory_budget(self) -> None:
        selection = select_system_profile(ResourceBudget(cpu_quota_percent=50), workload_profile="auto", runtime_platform="Linux")
        self.assertEqual(selection.profile, "linux-cpu-conservative")

    def test_manual_workload_profile_wins(self) -> None:
        selection = select_system_profile(ResourceBudget(memory_budget_gb=-3), workload_profile="throughput", runtime_platform="Linux")
        self.assertEqual(selection.profile, "linux-throughput")

    def test_manual_performance_profile_wins(self) -> None:
        selection = select_system_profile(ResourceBudget(memory_budget_gb=-3), workload_profile="performance", runtime_platform="Linux")
        self.assertEqual(selection.profile, "linux-performance")

    def test_selects_windows_profile_on_windows(self) -> None:
        selection = select_system_profile(ResourceBudget(memory_budget_gb=-3), workload_profile="auto", runtime_platform="Windows")
        self.assertEqual(selection.profile, "windows-memory-conservative")

    def test_manual_cpu_conservative_profile_wins(self) -> None:
        selection = select_system_profile(ResourceBudget(), workload_profile="cpu-conservative", runtime_platform="Linux")
        self.assertEqual(selection.profile, "linux-cpu-conservative")


if __name__ == "__main__":
    unittest.main()
