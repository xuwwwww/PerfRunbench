from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR, load_manifest, write_json
from autotune.resource.workload_runner import run_with_budget


def compare_tuning(
    command: list[str],
    budget: ResourceBudget,
    *,
    tuned_profile: str,
    output: str | Path = "results/reports/tuning_comparison.json",
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    system_tuning_sudo: bool = False,
    docker_image: str = "python:3.12-slim",
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    baseline_code, baseline_dir = run_with_budget(
        command,
        budget,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        docker_image=docker_image,
    )
    tuned_code, tuned_dir = run_with_budget(
        command,
        budget,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        tune_system_profile=tuned_profile,
        restore_system_after=True,
        system_tuning_sudo=system_tuning_sudo,
        docker_image=docker_image,
    )
    result = build_comparison_result(
        baseline_dir.name,
        tuned_dir.name,
        tuned_profile=tuned_profile,
        baseline_return_code=baseline_code,
        tuned_return_code=tuned_code,
    )
    write_json(Path(output), result)
    return result


def build_comparison_result(
    baseline_run_id: str,
    tuned_run_id: str,
    *,
    tuned_profile: str,
    baseline_return_code: int | None = None,
    tuned_return_code: int | None = None,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    baseline = _run_metrics(baseline_run_id, runs_dir)
    tuned = _run_metrics(tuned_run_id, runs_dir)
    return {
        "kind": "tuning_comparison",
        "tuned_profile": tuned_profile,
        "baseline": baseline,
        "tuned": tuned,
        "return_codes": {
            "baseline": baseline_return_code,
            "tuned": tuned_return_code,
        },
        "deltas": _deltas(baseline, tuned),
    }


def _run_metrics(run_id: str, runs_dir: Path) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    manifest = load_manifest(run_dir)
    analysis = analyze_run(run_id, runs_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": manifest.get("status"),
        "return_code": manifest.get("return_code"),
        "duration_seconds": _duration_seconds(manifest),
        "peak_memory_mb": analysis["memory"].get("peak_memory_mb"),
        "min_available_memory_mb": analysis["memory"].get("observed_min_available_memory_mb"),
        "memory_budget_exceeded": analysis["memory"].get("memory_budget_exceeded"),
        "peak_process_cpu_percent": analysis["cpu"].get("observed_peak_process_cpu_percent"),
        "peak_system_cpu_percent": analysis["cpu"].get("observed_peak_system_cpu_percent"),
        "diagnostics": analysis.get("diagnostics", []),
    }


def _duration_seconds(manifest: dict[str, Any]) -> float | None:
    started = manifest.get("started_at")
    finished = manifest.get("finished_at")
    if not started or not finished:
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        finish_dt = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return round((finish_dt - start_dt).total_seconds(), 3)


def _deltas(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_seconds": _delta(tuned.get("duration_seconds"), baseline.get("duration_seconds")),
        "duration_percent": _percent_delta(tuned.get("duration_seconds"), baseline.get("duration_seconds")),
        "peak_memory_mb": _delta(tuned.get("peak_memory_mb"), baseline.get("peak_memory_mb")),
        "peak_memory_percent": _percent_delta(tuned.get("peak_memory_mb"), baseline.get("peak_memory_mb")),
        "min_available_memory_mb": _delta(
            tuned.get("min_available_memory_mb"),
            baseline.get("min_available_memory_mb"),
        ),
        "peak_process_cpu_percent": _delta(
            tuned.get("peak_process_cpu_percent"),
            baseline.get("peak_process_cpu_percent"),
        ),
    }


def _delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return round(tuned - baseline, 3)


def _percent_delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return round(((tuned - baseline) / baseline) * 100.0, 3)
