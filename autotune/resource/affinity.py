from __future__ import annotations

from autotune.resource.budget import ResourceBudget


def get_logical_cpu_count() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except ModuleNotFoundError:
        return None


def filter_thread_budget(configs: list, budget: ResourceBudget) -> list:
    allowed = budget.allowed_threads(get_logical_cpu_count())
    if allowed is None:
        return configs
    return [config for config in configs if config.thread_count <= allowed]


def apply_cpu_affinity(budget: ResourceBudget) -> dict:
    total_cores = get_logical_cpu_count()
    allowed = budget.allowed_threads(total_cores)
    result = {
        "cpu_affinity_applied": False,
        "logical_cpu_count": total_cores,
        "allowed_threads": allowed,
        "affinity_cores": None,
    }
    if not budget.enabled or allowed is None:
        return result
    try:
        import psutil

        process = psutil.Process()
        current = process.cpu_affinity()
        selected = current[: min(allowed, len(current))]
        if selected:
            process.cpu_affinity(selected)
            result.update(
                {
                    "cpu_affinity_applied": True,
                    "affinity_cores": ",".join(str(core) for core in selected),
                }
            )
    except (AttributeError, NotImplementedError, OSError, ModuleNotFoundError):
        pass
    return result

