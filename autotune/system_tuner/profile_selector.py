from __future__ import annotations

import platform
from dataclasses import dataclass

from autotune.resource.budget import ResourceBudget


@dataclass(frozen=True)
class ProfileSelection:
    profile: str
    reason: str


def select_system_profile(
    budget: ResourceBudget,
    *,
    workload_profile: str = "auto",
    runtime_platform: str | None = None,
) -> ProfileSelection:
    if workload_profile not in {"auto", "training", "memory", "throughput", "low-latency", "cpu-conservative"}:
        raise ValueError(f"unknown workload profile: {workload_profile}")
    prefix = "windows" if (runtime_platform or platform.system()) == "Windows" else "linux"
    if workload_profile == "training":
        return ProfileSelection(f"{prefix}-training-safe", "workload_profile=training")
    if workload_profile == "memory":
        return ProfileSelection(f"{prefix}-memory-conservative", "workload_profile=memory")
    if workload_profile == "throughput":
        return ProfileSelection(f"{prefix}-throughput", "workload_profile=throughput")
    if workload_profile == "low-latency":
        return ProfileSelection(f"{prefix}-low-latency", "workload_profile=low-latency")
    if workload_profile == "cpu-conservative":
        return ProfileSelection(f"{prefix}-cpu-conservative", "workload_profile=cpu-conservative")
    if budget.memory_budget_gb is not None or budget.reserve_memory_gb > 0:
        return ProfileSelection(
            f"{prefix}-memory-conservative",
            "auto selected memory-conservative because a memory budget or memory reserve is configured",
        )
    if budget.reserve_cores > 0 or (budget.cpu_quota_percent is not None and budget.cpu_quota_percent < 100):
        return ProfileSelection(
            f"{prefix}-cpu-conservative",
            "auto selected cpu-conservative because a CPU quota or core reserve is configured",
        )
    return ProfileSelection(f"{prefix}-training-safe", "auto selected general training profile")
