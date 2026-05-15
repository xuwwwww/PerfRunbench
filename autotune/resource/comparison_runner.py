from __future__ import annotations

from datetime import datetime
import statistics
import platform
import time
from pathlib import Path
from typing import Any

from autotune.resource.advanced_tuning import AdvancedRunOptions
from autotune.resource.budget import ResourceBudget
from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR, load_manifest, write_json
from autotune.resource.workload_runner import run_with_budget
from autotune.system_tuner.runtime import available_profiles


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
    tuned_gpu_profile: str | None = None,
    gpu_tuning_sudo: bool = False,
    tuned_runtime_env_profile: str | None = None,
    docker_image: str = "python:3.12-slim",
    repeat: int = 1,
    alternate_order: bool = True,
    cooldown_seconds: float = 0.0,
    advanced_options: AdvancedRunOptions | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    runs: list[dict[str, Any]] = []
    for index in range(repeat):
        tuned_first = alternate_order and index % 2 == 1
        run_order = ["tuned", "baseline"] if tuned_first else ["baseline", "tuned"]
        trial: dict[str, Any] = {"execution_order": run_order}
        for position, label in enumerate(run_order):
            if label == "baseline":
                return_code, run_dir = run_with_budget(
                    command,
                    budget,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    docker_image=docker_image,
                    advanced_options=advanced_options,
                )
            else:
                return_code, run_dir = run_with_budget(
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
                    tune_gpu_profile=tuned_gpu_profile,
                    restore_gpu_after=bool(tuned_gpu_profile),
                    gpu_tuning_sudo=gpu_tuning_sudo,
                    runtime_env_profile=tuned_runtime_env_profile,
                    docker_image=docker_image,
                    advanced_options=advanced_options,
                )
            trial[f"{label}_code"] = return_code
            trial[f"{label}_dir"] = run_dir
            if cooldown_seconds > 0 and not (index == repeat - 1 and position == len(run_order) - 1):
                time.sleep(cooldown_seconds)
        runs.append(trial)

    last_trial = runs[-1]
    baseline_code = last_trial["baseline_code"]
    baseline_dir = last_trial["baseline_dir"]
    tuned_code = last_trial["tuned_code"]
    tuned_dir = last_trial["tuned_dir"]
    result = build_comparison_result(
        baseline_dir.name,
        tuned_dir.name,
        tuned_profile=tuned_profile,
        tuned_runtime_env_profile=tuned_runtime_env_profile,
        tuned_gpu_profile=tuned_gpu_profile,
        baseline_return_code=baseline_code,
        tuned_return_code=tuned_code,
        runs_dir=RUNS_DIR,
    )
    result["execution_order"] = list(last_trial["execution_order"])
    if repeat > 1:
        result["repeat"] = repeat
        result["trials"] = [
            _trial_result_from_record(
                trial,
                tuned_profile=tuned_profile,
                tuned_runtime_env_profile=tuned_runtime_env_profile,
                tuned_gpu_profile=tuned_gpu_profile,
            )
            for trial in runs
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


def compare_profiles(
    command: list[str],
    budget: ResourceBudget,
    *,
    profiles: list[str] | None = None,
    output: str | Path = "results/reports/profile_comparison_summary.json",
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    system_tuning_sudo: bool = False,
    tuned_gpu_profile: str | None = None,
    gpu_tuning_sudo: bool = False,
    tuned_runtime_env_profile: str | None = None,
    docker_image: str = "python:3.12-slim",
    repeat: int = 3,
    alternate_order: bool = True,
    cooldown_seconds: float = 0.0,
    advanced_options: AdvancedRunOptions | None = None,
) -> dict[str, Any]:
    selected_profiles = profiles or _default_profile_sweep(executor)
    comparisons = []
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for profile in selected_profiles:
        profile_output = output_path.parent / f"{profile.replace('-', '_')}_comparison.json"
        result = compare_tuning(
            command,
            budget,
            tuned_profile=profile,
            output=profile_output,
            sample_interval_seconds=sample_interval_seconds,
            hard_kill=hard_kill,
            executor=executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
            system_tuning_sudo=system_tuning_sudo,
            tuned_gpu_profile=tuned_gpu_profile,
            gpu_tuning_sudo=gpu_tuning_sudo,
            tuned_runtime_env_profile=tuned_runtime_env_profile,
            docker_image=docker_image,
            repeat=repeat,
            alternate_order=alternate_order,
            cooldown_seconds=cooldown_seconds,
            advanced_options=advanced_options,
        )
        aggregate = result.get("aggregate", {})
        deltas = aggregate.get("deltas", result.get("deltas", {}))
        comparisons.append(
            {
                "profile": profile,
                "output": str(profile_output),
                "samples_per_second_percent": _nested(deltas, "workload", "samples_per_second", "percent"),
                "benchmark_duration_percent": deltas.get("benchmark_duration_percent"),
                "peak_memory_percent": deltas.get("peak_memory_percent"),
                "system_tuning_overhead_seconds": deltas.get("system_tuning_overhead_seconds"),
                "runtime_env_profile": tuned_runtime_env_profile,
                "gpu_profile": tuned_gpu_profile,
            }
        )
    ranked = sorted(comparisons, key=_profile_rank_key, reverse=True)
    summary = {
        "kind": "profile_comparison_summary",
        "repeat": repeat,
        "profiles": selected_profiles,
        "best_profile": ranked[0]["profile"] if ranked else None,
        "best_profile_beats_baseline": _profile_beats_baseline(ranked[0]) if ranked else False,
        "comparisons": ranked,
    }
    write_json(output_path, summary)
    return summary


def compare_budget_modes(
    command: list[str],
    budgeted: ResourceBudget,
    *,
    unbounded: ResourceBudget | None = None,
    tuned_profile: str | None = None,
    output: str | Path = "results/reports/budget_comparison.json",
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    system_tuning_sudo: bool = False,
    tuned_gpu_profile: str | None = None,
    gpu_tuning_sudo: bool = False,
    tuned_runtime_env_profile: str | None = None,
    docker_image: str = "python:3.12-slim",
    repeat: int = 3,
    alternate_order: bool = True,
    cooldown_seconds: float = 0.0,
    advanced_options: AdvancedRunOptions | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    unbounded_budget = unbounded or ResourceBudget()
    runs: list[dict[str, Any]] = []
    for index in range(repeat):
        budgeted_first = alternate_order and index % 2 == 1
        run_order = ["budgeted", "unbounded"] if budgeted_first else ["unbounded", "budgeted"]
        trial: dict[str, Any] = {"execution_order": run_order}
        for position, label in enumerate(run_order):
            current_budget = budgeted if label == "budgeted" else unbounded_budget
            return_code, run_dir = run_with_budget(
                command,
                current_budget,
                sample_interval_seconds=sample_interval_seconds,
                hard_kill=hard_kill,
                executor=executor,
                use_sudo=use_sudo,
                allow_sudo_auto=allow_sudo_auto,
                tune_system_profile=tuned_profile,
                restore_system_after=bool(tuned_profile),
                system_tuning_sudo=system_tuning_sudo,
                tune_gpu_profile=tuned_gpu_profile,
                restore_gpu_after=bool(tuned_gpu_profile),
                gpu_tuning_sudo=gpu_tuning_sudo,
                runtime_env_profile=tuned_runtime_env_profile,
                docker_image=docker_image,
                advanced_options=advanced_options,
            )
            trial[f"{label}_code"] = return_code
            trial[f"{label}_dir"] = run_dir
            if cooldown_seconds > 0 and not (index == repeat - 1 and position == len(run_order) - 1):
                time.sleep(cooldown_seconds)
        runs.append(trial)

    last_trial = runs[-1]
    result = build_comparison_result(
        last_trial["unbounded_dir"].name,
        last_trial["budgeted_dir"].name,
        tuned_profile=tuned_profile or "no-system-tuning",
        tuned_runtime_env_profile=tuned_runtime_env_profile,
        tuned_gpu_profile=tuned_gpu_profile,
        baseline_return_code=last_trial["unbounded_code"],
        tuned_return_code=last_trial["budgeted_code"],
        runs_dir=RUNS_DIR,
    )
    result["kind"] = "budget_mode_comparison"
    result["comparison_target"] = "budget"
    result["baseline_label"] = "unbounded"
    result["tuned_label"] = "budgeted"
    result["execution_order"] = list(last_trial["execution_order"])
    if repeat > 1:
        result["repeat"] = repeat
        result["trials"] = []
        for trial in runs:
            trial_result = build_comparison_result(
                trial["unbounded_dir"].name,
                trial["budgeted_dir"].name,
                tuned_profile=tuned_profile or "no-system-tuning",
                tuned_runtime_env_profile=tuned_runtime_env_profile,
                tuned_gpu_profile=tuned_gpu_profile,
                baseline_return_code=trial["unbounded_code"],
                tuned_return_code=trial["budgeted_code"],
                runs_dir=RUNS_DIR,
            )
            trial_result["kind"] = "budget_mode_comparison"
            trial_result["comparison_target"] = "budget"
            trial_result["baseline_label"] = "unbounded"
            trial_result["tuned_label"] = "budgeted"
            trial_result["execution_order"] = list(trial["execution_order"])
            result["trials"].append(trial_result)
        result["aggregate"] = _aggregate_trials(result["trials"])
    write_json(Path(output), result)
    failures = _failed_runs(result)
    if failures:
        details = ", ".join(
            f"{item['label']}(run_id={item['run_id']}, return_code={item['return_code']}, status={item['status']})"
            for item in failures
        )
        raise RuntimeError(
            "compare-budgets workload failed; budget comparison is not valid. "
            f"Failed run(s): {details}"
        )
    return result


def build_comparison_result(
    baseline_run_id: str,
    tuned_run_id: str,
    *,
    tuned_profile: str,
    tuned_runtime_env_profile: str | None = None,
    tuned_gpu_profile: str | None = None,
    baseline_return_code: int | None = None,
    tuned_return_code: int | None = None,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    baseline = _run_metrics(baseline_run_id, runs_dir)
    tuned = _run_metrics(tuned_run_id, runs_dir)
    return {
        "kind": "tuning_comparison",
        "tuned_profile": tuned_profile,
        "tuned_runtime_env_profile": tuned_runtime_env_profile,
        "tuned_gpu_profile": tuned_gpu_profile,
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
        "gpu_tuning": analysis.get("gpu_tuning", {}),
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


def _trial_result_from_record(
    trial: dict[str, Any],
    *,
    tuned_profile: str,
    tuned_runtime_env_profile: str | None = None,
    tuned_gpu_profile: str | None = None,
) -> dict[str, Any]:
    result = build_comparison_result(
        trial["baseline_dir"].name,
        trial["tuned_dir"].name,
        tuned_profile=tuned_profile,
        tuned_runtime_env_profile=tuned_runtime_env_profile,
        tuned_gpu_profile=tuned_gpu_profile,
        baseline_return_code=trial["baseline_code"],
        tuned_return_code=trial["tuned_code"],
        runs_dir=RUNS_DIR,
    )
    result["execution_order"] = list(trial["execution_order"])
    return result


def _workload_deltas(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "duration_seconds",
        "epoch_time_mean_seconds",
        "epoch_time_max_seconds",
        "step_time_mean_seconds",
        "step_time_sample_mean_seconds",
        "step_time_p50_seconds",
        "step_time_p95_seconds",
        "step_time_p99_seconds",
        "step_time_max_seconds",
        "step_time_sample_count",
        "samples_per_second",
        "peak_batch_payload_mb",
        "optimizer_steps",
        "completed_epochs",
        "feature_count",
        "train_samples",
        "cpu_workers",
        "memory_target_mb",
        "memory_touched_mb",
        "gpu_matmuls_per_second",
        "gpu_tflops_estimate",
        "gpu_peak_memory_allocated_mb",
        "gpu_peak_memory_reserved_mb",
        "gpu_memory_target_mb",
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
        "step_time_sample_mean_seconds",
        "step_time_p50_seconds",
        "step_time_p95_seconds",
        "step_time_p99_seconds",
        "step_time_max_seconds",
        "step_time_sample_count",
        "samples_per_second",
        "peak_batch_payload_mb",
        "optimizer_steps",
        "completed_epochs",
        "feature_count",
        "train_samples",
        "cache_copies",
        "cpu_workers",
        "memory_target_mb",
        "memory_touched_mb",
        "gpu_matmuls_per_second",
        "gpu_tflops_estimate",
        "gpu_peak_memory_allocated_mb",
        "gpu_peak_memory_reserved_mb",
        "gpu_memory_target_mb",
        "config_path",
        "dataset",
        "device",
        "cuda_version",
        "dtype",
        "allow_tf32",
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


def _default_profile_sweep(executor: str) -> list[str]:
    prefix = "windows" if platform.system() == "Windows" else "linux"
    available = available_profiles()
    matched = [item for item in available if item.startswith(prefix)]
    return matched or available


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
            "step_time_sample_mean_seconds": _median_value(workload, "step_time_sample_mean_seconds"),
            "step_time_p50_seconds": _median_value(workload, "step_time_p50_seconds"),
            "step_time_p95_seconds": _median_value(workload, "step_time_p95_seconds"),
            "step_time_p99_seconds": _median_value(workload, "step_time_p99_seconds"),
            "step_time_max_seconds": _median_value(workload, "step_time_max_seconds"),
            "step_time_sample_count": _median_value(workload, "step_time_sample_count"),
            "samples_per_second": _median_value(workload, "samples_per_second"),
            "peak_batch_payload_mb": _median_value(workload, "peak_batch_payload_mb"),
            "optimizer_steps": _median_value(workload, "optimizer_steps"),
            "completed_epochs": _median_value(workload, "completed_epochs"),
            "feature_count": _median_value(workload, "feature_count"),
            "train_samples": _median_value(workload, "train_samples"),
            "cpu_workers": _median_value(workload, "cpu_workers"),
            "memory_target_mb": _median_value(workload, "memory_target_mb"),
            "memory_touched_mb": _median_value(workload, "memory_touched_mb"),
            "gpu_matmuls_per_second": _median_value(workload, "gpu_matmuls_per_second"),
            "gpu_tflops_estimate": _median_value(workload, "gpu_tflops_estimate"),
            "gpu_peak_memory_allocated_mb": _median_value(workload, "gpu_peak_memory_allocated_mb"),
            "gpu_peak_memory_reserved_mb": _median_value(workload, "gpu_peak_memory_reserved_mb"),
            "gpu_memory_target_mb": _median_value(workload, "gpu_memory_target_mb"),
        },
    }


def _median_value(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _profile_rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
    throughput = float(item.get("samples_per_second_percent") or float("-inf"))
    duration = -(float(item.get("benchmark_duration_percent") or float("inf")))
    memory = -(float(item.get("peak_memory_percent") or 0.0))
    return throughput, duration, memory


def _profile_beats_baseline(item: dict[str, Any]) -> bool:
    throughput = item.get("samples_per_second_percent")
    duration = item.get("benchmark_duration_percent")
    return isinstance(throughput, (int, float)) and isinstance(duration, (int, float)) and throughput > 0 and duration <= 0


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return round(tuned - baseline, 3)


def _percent_delta(tuned: Any, baseline: Any) -> float | None:
    if not isinstance(tuned, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return round(((tuned - baseline) / baseline) * 100.0, 3)
