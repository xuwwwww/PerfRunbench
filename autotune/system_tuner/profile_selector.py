from __future__ import annotations

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
) -> ProfileSelection:
    if workload_profile not in {"auto", "training", "memory", "throughput", "low-latency"}:
        raise ValueError(f"unknown workload profile: {workload_profile}")
    if workload_profile == "training":
        return ProfileSelection("linux-training-safe", "workload_profile=training")
    if workload_profile == "memory":
        return ProfileSelection("linux-memory-conservative", "workload_profile=memory")
    if workload_profile == "throughput":
        return ProfileSelection("linux-throughput", "workload_profile=throughput")
    if workload_profile == "low-latency":
        return ProfileSelection("linux-low-latency", "workload_profile=low-latency")
    if budget.memory_budget_gb is not None or budget.reserve_memory_gb > 0:
        return ProfileSelection(
            "linux-memory-conservative",
            "auto selected memory-conservative because a memory budget or memory reserve is configured",
        )
    if budget.cpu_quota_percent is not None and budget.cpu_quota_percent < 100:
        return ProfileSelection(
            "linux-low-latency",
            "auto selected low-latency because a CPU quota is configured",
        )
    return ProfileSelection("linux-training-safe", "auto selected general training profile")
