from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR, write_json
from autotune.resource.workload_runner import run_with_budget

RunWithBudget = Callable[..., tuple[int, Path]]


def calibrate_memory(
    budget_gb_values: list[float],
    *,
    workload_memory_mb: int,
    duration_seconds: float = 5.0,
    workers: int = 2,
    output: str | Path = "results/reports/memory_calibration.json",
    executor: str = "local",
    sample_interval_seconds: float = 0.1,
    hard_kill: bool = False,
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    docker_image: str = "python:3.12-slim",
    runs_dir: Path = RUNS_DIR,
    runner: RunWithBudget = run_with_budget,
) -> dict[str, Any]:
    if not budget_gb_values:
        raise ValueError("budget_gb_values cannot be empty")
    if workload_memory_mb <= 0:
        raise ValueError("workload_memory_mb must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    records = []
    for budget_gb in budget_gb_values:
        budget = ResourceBudget(memory_budget_gb=budget_gb)
        command = _stress_command(workers, duration_seconds, workload_memory_mb)
        return_code, run_dir = runner(
            command,
            budget,
            sample_interval_seconds=sample_interval_seconds,
            hard_kill=hard_kill,
            executor=executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
            docker_image=docker_image,
        )
        analysis = analyze_run(run_dir.name, runs_dir)
        records.append(_record_from_analysis(budget_gb, workload_memory_mb, return_code, run_dir, analysis))

    result = {
        "kind": "memory_calibration",
        "workload_memory_mb": workload_memory_mb,
        "duration_seconds": duration_seconds,
        "workers": workers,
        "executor": executor,
        "sample_interval_seconds": sample_interval_seconds,
        "hard_kill": hard_kill,
        "docker_image": docker_image,
        "records": records,
        "recommendations": _recommend(records),
    }
    write_json(Path(output), result)
    return result


def _stress_command(workers: int, duration_seconds: float, memory_mb: int) -> list[str]:
    return [
        sys.executable,
        "scripts/stress_workload.py",
        "--workers",
        str(workers),
        "--duration-seconds",
        str(duration_seconds),
        "--memory-mb",
        str(memory_mb),
    ]


def _record_from_analysis(
    budget_gb: float,
    workload_memory_mb: int,
    return_code: int,
    run_dir: Path,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    memory = analysis.get("memory", {})
    observed_min_available_gb = memory.get("observed_min_available_memory_gb")
    requested_reserve_gb = abs(budget_gb) if budget_gb < 0 else None
    reserve_error_gb = None
    if requested_reserve_gb is not None and observed_min_available_gb is not None:
        reserve_error_gb = round(observed_min_available_gb - requested_reserve_gb, 3)
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "return_code": return_code,
        "requested_budget_gb": budget_gb,
        "requested_budget_mode": "reserve_to_full" if budget_gb < 0 else "absolute",
        "workload_memory_mb": workload_memory_mb,
        "effective_budget_mb": memory.get("effective_budget_mb"),
        "peak_memory_mb": memory.get("peak_memory_mb"),
        "observed_min_available_memory_gb": observed_min_available_gb,
        "memory_budget_exceeded": memory.get("memory_budget_exceeded"),
        "reserve_error_gb": reserve_error_gb,
    }


def _recommend(records: list[dict[str, Any]]) -> list[str]:
    recommendations = []
    reserve_errors = [
        record["reserve_error_gb"]
        for record in records
        if record.get("requested_budget_mode") == "reserve_to_full" and record.get("reserve_error_gb") is not None
    ]
    if reserve_errors:
        average_error = round(sum(reserve_errors) / len(reserve_errors), 3)
        recommendations.append(
            "For negative memory budgets on this machine, observed free-memory headroom differed from the target "
            f"by {average_error:+.3f} GB on average in this calibration workload."
        )
    exceeded = [record for record in records if record.get("memory_budget_exceeded")]
    if exceeded:
        recommendations.append(
            "At least one calibration run exceeded the monitored budget; use hard-kill or a cgroup executor for strict enforcement."
        )
    if not recommendations:
        recommendations.append("No memory calibration drift was detected from the collected records.")
    return recommendations
