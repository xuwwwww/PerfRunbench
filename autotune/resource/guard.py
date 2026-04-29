from __future__ import annotations

from autotune.resource.budget import ResourceBudget


class ResourceBudgetError(RuntimeError):
    pass


def check_memory_start_guard(budget: ResourceBudget) -> dict:
    snapshot = {
        "total_memory_mb": None,
        "available_memory_before_mb": None,
        "memory_start_guard_passed": True,
    }
    if not budget.enabled:
        return snapshot
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_mb = memory.total / (1024 * 1024)
        available_mb = memory.available / (1024 * 1024)
        snapshot.update(
            {
                "total_memory_mb": round(total_mb, 3),
                "available_memory_before_mb": round(available_mb, 3),
            }
        )
        required_free_mb = budget.reserve_memory_gb * 1024.0
        if budget.enforce and required_free_mb > 0 and available_mb < required_free_mb:
            snapshot["memory_start_guard_passed"] = False
            raise ResourceBudgetError(
                f"available memory {available_mb:.1f} MB is below reserved memory {required_free_mb:.1f} MB"
            )
    except ModuleNotFoundError:
        pass
    return snapshot

