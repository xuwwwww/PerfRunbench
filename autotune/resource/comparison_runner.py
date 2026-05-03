from __future__ import annotations

from datetime import datetime
import statistics
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
    repeat: int = 1,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    runs = []
    for _index in range(repeat):
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
        runs.append((baseline_code, baseline_dir, tuned_code, tuned_dir))

    baseline_code, baseline_dir, tuned_code, tuned_dir = runs[-1]
    result = build_comparison_result(
        baseline_dir.name,
        tuned_dir.name,
        tuned_profile=tuned_profile,
        baseline_return_code=baseline_code,
        tuned_return_code=tuned_code,
        runs_dir=RUNS_DIR,
    )
    if repeat > 1:
        result["repeat"] = repeat
        result["trials"] = [
            build_comparison_result(
                baseline_dir.name,
                tuned_dir.name,
                tuned_profile=tuned_profile,
                baseline_return_code=baseline_code,
                tuned_return_code=tuned_code,
                runs_dir=RUNS_DIR,
            )
            for baseline_code, baseline_dir, tuned_code, tuned_dir in runs
        ]
        result["aggregate"] = _aggregate_trials(result["trials"])
    write_json(Path(output), result)
    failures = _failed_runs(result)
    if failures:
        details = ", ".join(
            f"{item['label']}(run_id={item['run_id']}, return_code={item['return_code']}, status={item['status']})"
            for item in failures
        )
        raise RuntimeError(
            "compare-tuning workload failed; tuning comparison is not valid. "
            f"Failed run(s): {details}"
        )
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
    lifecycle_duration = _duration_seconds(manifest)
    workload = _workload_performance_metrics(analysis.get("workload", {}))
    workload_duration = _workload_duration_seconds(workload)
    system_tuning_overhead = _system_tuning_overhead_seconds(analysis.get("system_tuning", {}))
    adjusted_duration = _adjusted_duration_seconds(lifecycle_duration, system_tuning_overhead)
    benchmark_duration = workload_duration if workload_duration is not None else adjusted_duration
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": manifest.get("status"),
        "return_code": manifest.get("return_code"),
        "duration_seconds": benchmark_duration,
        "benchmark_duration_seconds": benchmark_duration,
        "lifecycle_duration_seconds": lifecycle_duration,
        "adjusted_lifecycle_duration_seconds": adjusted_duration,
        "workload_duration_seconds": workload_duration,
        "system_tuning_overhead_seconds": system_tuning_overhead,
        "peak_memory_mb": analysis["memory"].get("peak_memory_mb"),
        "min_available_memory_mb": analysis["memory"].get("observed_min_available_memory_mb"),
        "memory_budget_exceeded": analysis["memory"].get("memory_budget_exceeded"),
        "peak_process_cpu_percent": analysis["cpu"].get("observed_peak_process_cpu_percent"),
        "peak_system_cpu_percent": analysis["cpu"].get("observed_peak_system_cpu_percent"),
        "system_tuning": analysis.get("system_tuning", {}),
        "workload": workload,
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


def _workload_duration_seconds(workload: dict[str, Any]) -> float | None:
    value = workload.get("duration_seconds")
    return round(value, 6) if isinstance(value, (int, float)) else None


def _system_tuning_overhead_seconds(system_tuning: dict[str, Any]) -> float:
    apply_seconds = system_tuning.get("apply_seconds")
    restore_seconds = system_tuning.get("restore_seconds")
    return round(
        sum(value for value in [apply_seconds, restore_seconds] if isinstance(value, (int, float))),
        6,
    )


def _adjusted_duration_seconds(lifecycle_duration: float | None, system_tuning_overhead: float | None) -> float | None:
    if lifecycle_duration is None:
        return None
    overhead = system_tuning_overhead if isinstance(system_tuning_overhead, (int, float)) else 0.0
    return round(max(0.0, lifecycle_duration - overhead), 6)


def _deltas(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_seconds": _delta(tuned.get("duration_seconds"), baseline.get("duration_seconds")),
        "duration_percent": _percent_delta(tuned.get("duration_seconds"), baseline.get("duration_seconds")),
        "benchmark_duration_seconds": _delta(
            tuned.get("benchmark_duration_seconds"),
            baseline.get("benchmark_duration_seconds"),
        ),
        "benchmark_duration_percent": _percent_delta(
            tuned.get("benchmark_duration_seconds"),
            baseline.get("benchmark_duration_seconds"),
        ),
        "lifecycle_duration_seconds": _delta(
            tuned.get("lifecycle_duration_seconds"),
            baseline.get("lifecycle_duration_seconds"),
        ),
        "lifecycle_duration_percent": _percent_delta(
            tuned.get("lifecycle_duration_seconds"),
            baseline.get("lifecycle_duration_seconds"),
        ),
        "workload_duration_seconds": _delta(
            tuned.get("workload_duration_seconds"),
            baseline.get("workload_duration_seconds"),
        ),
        "workload_duration_percent": _percent_delta(
            tuned.get("workload_duration_seconds"),
            baseline.get("workload_duration_seconds"),
        ),
        "system_tuning_overhead_seconds": _delta(
            tuned.get("system_tuning_overhead_seconds"),
            baseline.get("system_tuning_overhead_seconds"),
        ),
        "adjusted_lifecycle_duration_seconds": _delta(
            tuned.get("adjusted_lifecycle_duration_seconds"),
            baseline.get("adjusted_lifecycle_duration_seconds"),
        ),
        "adjusted_lifecycle_duration_percent": _percent_delta(
            tuned.get("adjusted_lifecycle_duration_seconds"),
            baseline.get("adjusted_lifecycle_duration_seconds"),
        ),
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
        "workload": _workload_deltas(
            baseline.get("workload", {}),
            tuned.get("workload", {}),
        ),
    }


def _failed_runs(result: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    comparisons = result.get("trials") if isinstance(result.get("trials"), list) else [result]
    for index, comparison in enumerate(comparisons, start=1):
        for label in ("baseline", "tuned"):
            run = comparison.get(label, {})
            if run.get("return_code") != 0 or run.get("status") != "completed":
                failures.append(
                    {
                        "label": f"trial{index}.{label}" if len(comparisons) > 1 else label,
                        "run_id": run.get("run_id"),
                        "return_code": run.get("return_code"),
                        "status": run.get("status"),
                    }
                )
    return failures


def _workload_deltas(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "duration_seconds",
        "epoch_time_mean_seconds",
        "epoch_time_max_seconds",
        "step_time_mean_seconds",
        "samples_per_second",
        "peak_batch_payload_mb",
        "optimizer_steps",
        "completed_epochs",
        "feature_count",
        "train_samples",
    ]
    deltas: dict[str, Any] = {}
    for key in keys:
        deltas[key] = {
            "absolute": _delta(tuned.get(key), baseline.get(key)),
            "percent": _percent_delta(tuned.get(key), baseline.get(key)),
        }
    return deltas


def _workload_performance_metrics(workload: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "duration_seconds",
        "epoch_time_mean_seconds",
        "epoch_time_max_seconds",
        "step_time_mean_seconds",
        "samples_per_second",
        "peak_batch_payload_mb",
        "optimizer_steps",
        "completed_epochs",
        "feature_count",
        "train_samples",
        "cache_copies",
        "config_path",
        "dataset",
    ]
    return {key: workload[key] for key in allowed if key in workload}


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_runs = [trial["baseline"] for trial in trials]
    tuned_runs = [trial["tuned"] for trial in trials]
    return {
        "baseline": _aggregate_runs(baseline_runs),
        "tuned": _aggregate_runs(tuned_runs),
        "deltas": _deltas(_aggregate_runs(baseline_runs), _aggregate_runs(tuned_runs)),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    workload = [_run.get("workload", {}) for _run in runs]
    return {
        "run_ids": [run.get("run_id") for run in runs],
        "status": "completed" if all(run.get("status") == "completed" for run in runs) else "mixed",
        "return_code": 0 if all(run.get("return_code") == 0 for run in runs) else None,
        "duration_seconds": _median_value(runs, "duration_seconds"),
        "benchmark_duration_seconds": _median_value(runs, "benchmark_duration_seconds"),
        "lifecycle_duration_seconds": _median_value(runs, "lifecycle_duration_seconds"),
        "adjusted_lifecycle_duration_seconds": _median_value(runs, "adjusted_lifecycle_duration_seconds"),
        "workload_duration_seconds": _median_value(runs, "workload_duration_seconds"),
        "system_tuning_overhead_seconds": _median_value(runs, "system_tuning_overhead_seconds"),
        "peak_memory_mb": _median_value(runs, "peak_memory_mb"),
        "min_available_memory_mb": _median_value(runs, "min_available_memory_mb"),
        "memory_budget_exceeded": any(run.get("memory_budget_exceeded") for run in runs),
        "peak_process_cpu_percent": _median_value(runs, "peak_process_cpu_percent"),
        "peak_system_cpu_percent": _median_value(runs, "peak_system_cpu_percent"),
        "workload": {
            "duration_seconds": _median_value(workload, "duration_seconds"),
            "epoch_time_mean_seconds": _median_value(workload, "epoch_time_mean_seconds"),
            "epoch_time_max_seconds": _median_value(workload, "epoch_time_max_seconds"),
            "step_time_mean_seconds": _median_value(workload, "step_time_mean_seconds"),
            "samples_per_second": _median_value(workload, "samples_per_second"),
            "peak_batch_payload_mb": _median_value(workload, "peak_batch_payload_mb"),
            "optimizer_steps": _median_value(workload, "optimizer_steps"),
            "completed_epochs": _median_value(workload, "completed_epochs"),
            "feature_count": _median_value(workload, "feature_count"),
            "train_samples": _median_value(workload, "train_samples"),
        },
    }


def _median_value(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return round(tuned - baseline, 3)


def _percent_delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return round(((tuned - baseline) / baseline) * 100.0, 3)
