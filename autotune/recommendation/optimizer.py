from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotune.gpu.nvidia_tuner import recommend_nvidia_tuning
from autotune.resource.budget import ResourceBudget
from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR, write_json
from autotune.resource.workload_runner import launch_performance, run_with_budget
from autotune.runtime_tuner.env import available_runtime_profiles
from autotune.system_tuner.runtime import available_profiles


RECOMMENDATIONS_DIR = Path(".autotuneai") / "recommendations"
LATEST_RECOMMENDATION = RECOMMENDATIONS_DIR / "latest.json"
OPTIMIZATION_TARGETS = {"auto", "cpu", "memory", "gpu", "mixed"}
DEFAULT_NOISE_BAND_PERCENT = 2.0


@dataclass(frozen=True)
class RecommendationCandidate:
    label: str
    guard_mode: str
    budget: ResourceBudget
    system_profile: str | None = None
    runtime_profile: str | None = None
    gpu_profile: str | None = None


def optimize_recommendation(
    command: list[str],
    budget: ResourceBudget,
    *,
    output: str | Path = "results/reports/auto_recommendation.json",
    cache_path: str | Path = LATEST_RECOMMENDATION,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    system_tuning_sudo: bool = False,
    gpu_tuning_sudo: bool = False,
    docker_image: str = "python:3.12-slim",
    repeat: int = 1,
    warmup_runs: int = 0,
    cooldown_seconds: float = 0.0,
    include_gpu: bool = True,
    max_candidates: int | None = None,
    optimization_mode: str = "guarded",
    optimization_target: str = "auto",
    monitor_mode: str = "full",
    time_budget_hours: float | None = None,
    thermal_control: bool | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if optimization_mode not in {"guarded", "performance"}:
        raise ValueError("optimization_mode must be 'guarded' or 'performance'")
    if optimization_target not in OPTIMIZATION_TARGETS:
        raise ValueError(f"optimization_target must be one of {sorted(OPTIMIZATION_TARGETS)}")
    if monitor_mode not in {"full", "minimal"}:
        raise ValueError("monitor_mode must be 'full' or 'minimal'")
    if monitor_mode == "minimal" and optimization_mode != "performance":
        raise ValueError("minimal monitor_mode is only supported for performance optimization")
    if time_budget_hours is not None and time_budget_hours <= 0:
        raise ValueError("time_budget_hours must be > 0")
    if thermal_control is None:
        thermal_control = optimization_mode == "performance" and monitor_mode == "minimal"
    effective_budget = ResourceBudget(enforce=False) if optimization_mode == "performance" else budget
    fingerprint = _fingerprint(command, effective_budget, executor, optimization_mode=optimization_mode, optimization_target=optimization_target)
    candidates = _candidate_plan(
        effective_budget,
        include_gpu=include_gpu,
        optimization_mode=optimization_mode,
        optimization_target=optimization_target,
    )
    if max_candidates is not None:
        candidates = candidates[: max(1, max_candidates)]

    deadline = time.monotonic() + time_budget_hours * 3600 if time_budget_hours is not None else None
    warmups = []
    if warmup_runs:
        warmup_candidate = RecommendationCandidate("warmup:baseline", "warmup", ResourceBudget())
        for warmup_index in range(warmup_runs):
            if _deadline_expired(deadline):
                break
            return_code, run_dir = _run_candidate(
                command,
                warmup_candidate,
                sample_interval_seconds=sample_interval_seconds,
                hard_kill=hard_kill,
                executor=executor,
                use_sudo=use_sudo,
                allow_sudo_auto=allow_sudo_auto,
                system_tuning_sudo=system_tuning_sudo,
                gpu_tuning_sudo=gpu_tuning_sudo,
                docker_image=docker_image,
                monitor_mode=monitor_mode,
                optimization_mode=optimization_mode,
            )
            warmups.append({"run_id": run_dir.name, "return_code": return_code, "index": warmup_index})
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)

    trial_map: dict[str, list[dict[str, Any]]] = {candidate.label: [] for candidate in candidates}
    execution_order: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if thermal_control and optimization_mode == "performance" and len(candidates) > 1:
        results, execution_order = _run_thermal_controlled_trials(
            command,
            candidates,
            trial_map,
            deadline=deadline,
            output=output,
            cache_path=cache_path,
            fingerprint=fingerprint,
            executor=executor,
            repeat=repeat,
            warmup_runs=warmup_runs,
            warmups=warmups,
            optimization_mode=optimization_mode,
            optimization_target=optimization_target,
            monitor_mode=monitor_mode,
            time_budget_hours=time_budget_hours,
            thermal_control=thermal_control,
            sample_interval_seconds=sample_interval_seconds,
            hard_kill=hard_kill,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
            system_tuning_sudo=system_tuning_sudo,
            gpu_tuning_sudo=gpu_tuning_sudo,
            docker_image=docker_image,
            cooldown_seconds=cooldown_seconds,
        )
        return _write_summary(
            output=output,
            cache_path=cache_path,
            fingerprint=fingerprint,
            command=command,
            executor=executor,
            repeat=repeat,
            warmup_runs=warmup_runs,
            warmups=warmups,
            optimization_mode=optimization_mode,
            optimization_target=optimization_target,
            monitor_mode=monitor_mode,
            time_budget_hours=time_budget_hours,
            thermal_control=thermal_control,
            results=results,
            execution_order=execution_order,
            complete=all(len(trial_map[candidate.label]) >= repeat for candidate in candidates if candidate.label != candidates[0].label),
        )

    for trial_index in range(repeat):
        for order_index, candidate in enumerate(_rotated_execution_order(candidates, trial_index)):
            if _deadline_expired(deadline):
                break
            return_code, run_dir = _run_candidate(
                command,
                candidate,
                sample_interval_seconds=sample_interval_seconds,
                hard_kill=hard_kill,
                executor=executor,
                use_sudo=use_sudo,
                allow_sudo_auto=allow_sudo_auto,
                system_tuning_sudo=system_tuning_sudo,
                gpu_tuning_sudo=gpu_tuning_sudo,
                docker_image=docker_image,
                monitor_mode=monitor_mode,
                optimization_mode=optimization_mode,
            )
            metrics = _run_metrics(run_dir.name)
            trial = {
                "run_id": run_dir.name,
                "return_code": return_code,
                "trial_index": trial_index,
                "order_index": order_index,
                **metrics,
            }
            trial_map[candidate.label].append(trial)
            execution_order.append(
                {
                    "trial_index": trial_index,
                    "order_index": order_index,
                    "label": candidate.label,
                    "run_id": run_dir.name,
                    "return_code": return_code,
                }
            )
            results = _results_from_trials(candidates, trial_map)
            if cooldown_seconds > 0 and not (trial_index == repeat - 1 and order_index == len(candidates) - 1):
                time.sleep(cooldown_seconds)
            _write_summary(
                output=output,
                cache_path=cache_path,
                fingerprint=fingerprint,
                command=command,
                executor=executor,
                repeat=repeat,
                warmup_runs=warmup_runs,
                warmups=warmups,
                optimization_mode=optimization_mode,
                optimization_target=optimization_target,
                monitor_mode=monitor_mode,
                time_budget_hours=time_budget_hours,
                thermal_control=thermal_control,
                results=results,
                execution_order=execution_order,
                complete=False,
            )
        if _deadline_expired(deadline):
            break

    return _write_summary(
            output=output,
            cache_path=cache_path,
            fingerprint=fingerprint,
            command=command,
            executor=executor,
            repeat=repeat,
            warmup_runs=warmup_runs,
            warmups=warmups,
            optimization_mode=optimization_mode,
            optimization_target=optimization_target,
            monitor_mode=monitor_mode,
            time_budget_hours=time_budget_hours,
            thermal_control=thermal_control,
            results=results,
            execution_order=execution_order,
            complete=all(len(trial_map[candidate.label]) >= repeat for candidate in candidates),
    )


def _run_candidate(
    command: list[str],
    candidate: RecommendationCandidate,
    *,
    sample_interval_seconds: float,
    hard_kill: bool,
    executor: str,
    use_sudo: bool,
    allow_sudo_auto: bool,
    system_tuning_sudo: bool,
    gpu_tuning_sudo: bool,
    docker_image: str,
    monitor_mode: str,
    optimization_mode: str,
) -> tuple[int, Path]:
    if optimization_mode == "performance" and monitor_mode == "minimal":
        return launch_performance(
            command,
            candidate.budget,
            executor=executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
            tune_system_profile=candidate.system_profile,
            restore_system_after=bool(candidate.system_profile),
            system_tuning_sudo=system_tuning_sudo,
            tune_gpu_profile=candidate.gpu_profile,
            restore_gpu_after=bool(candidate.gpu_profile),
            gpu_tuning_sudo=gpu_tuning_sudo,
            runtime_env_profile=candidate.runtime_profile,
            docker_image=docker_image,
        )
    return run_with_budget(
        command,
        candidate.budget,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        tune_system_profile=candidate.system_profile,
        restore_system_after=bool(candidate.system_profile),
        system_tuning_sudo=system_tuning_sudo,
        tune_gpu_profile=candidate.gpu_profile,
        restore_gpu_after=bool(candidate.gpu_profile),
        gpu_tuning_sudo=gpu_tuning_sudo,
        runtime_env_profile=candidate.runtime_profile,
        docker_image=docker_image,
    )


def _run_thermal_controlled_trials(
    command: list[str],
    candidates: list[RecommendationCandidate],
    trial_map: dict[str, list[dict[str, Any]]],
    *,
    deadline: float | None,
    output: str | Path,
    cache_path: str | Path,
    fingerprint: str,
    executor: str,
    repeat: int,
    warmup_runs: int,
    warmups: list[dict[str, Any]],
    optimization_mode: str,
    optimization_target: str,
    monitor_mode: str,
    time_budget_hours: float | None,
    thermal_control: bool,
    sample_interval_seconds: float,
    hard_kill: bool,
    use_sudo: bool,
    allow_sudo_auto: bool,
    system_tuning_sudo: bool,
    gpu_tuning_sudo: bool,
    docker_image: str,
    cooldown_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = candidates[0]
    candidates_to_test = candidates[1:]
    execution_order: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    pair_index = 0
    for trial_index in range(repeat):
        for order_index, candidate in enumerate(_rotated_execution_order(candidates_to_test, trial_index)):
            if _deadline_expired(deadline):
                break
            pair_id = f"trial{trial_index}_pair{pair_index}"
            baseline_first = (trial_index + order_index) % 2 == 0
            if baseline_first:
                baseline_trial = _run_measured_trial(
                    command,
                    baseline,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    system_tuning_sudo=system_tuning_sudo,
                    gpu_tuning_sudo=gpu_tuning_sudo,
                    docker_image=docker_image,
                    monitor_mode=monitor_mode,
                    optimization_mode=optimization_mode,
                    trial_index=trial_index,
                    order_index=order_index * 2,
                    pair_id=pair_id,
                    paired_label=candidate.label,
                    role="baseline_control",
                )
                trial_map[baseline.label].append(baseline_trial)
                execution_order.append(_execution_record(baseline, baseline_trial))
                candidate_trial = _run_measured_trial(
                    command,
                    candidate,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    system_tuning_sudo=system_tuning_sudo,
                    gpu_tuning_sudo=gpu_tuning_sudo,
                    docker_image=docker_image,
                    monitor_mode=monitor_mode,
                    optimization_mode=optimization_mode,
                    trial_index=trial_index,
                    order_index=order_index * 2 + 1,
                    pair_id=pair_id,
                    paired_label=baseline.label,
                    role="candidate",
                    baseline_trial=baseline_trial,
                )
            else:
                candidate_trial = _run_measured_trial(
                    command,
                    candidate,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    system_tuning_sudo=system_tuning_sudo,
                    gpu_tuning_sudo=gpu_tuning_sudo,
                    docker_image=docker_image,
                    monitor_mode=monitor_mode,
                    optimization_mode=optimization_mode,
                    trial_index=trial_index,
                    order_index=order_index * 2,
                    pair_id=pair_id,
                    paired_label=baseline.label,
                    role="candidate",
                )
                baseline_trial = _run_measured_trial(
                    command,
                    baseline,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    system_tuning_sudo=system_tuning_sudo,
                    gpu_tuning_sudo=gpu_tuning_sudo,
                    docker_image=docker_image,
                    monitor_mode=monitor_mode,
                    optimization_mode=optimization_mode,
                    trial_index=trial_index,
                    order_index=order_index * 2 + 1,
                    pair_id=pair_id,
                    paired_label=candidate.label,
                    role="baseline_control",
                )
                candidate_trial = _add_paired_baseline(candidate_trial, baseline_trial)
                trial_map[baseline.label].append(baseline_trial)
                execution_order.append(_execution_record(candidate, candidate_trial))
                execution_order.append(_execution_record(baseline, baseline_trial))
                trial_map[candidate.label].append(candidate_trial)
                results = _results_from_trials(candidates, trial_map)
                _write_summary(
                    output=output,
                    cache_path=cache_path,
                    fingerprint=fingerprint,
                    command=command,
                    executor=executor,
                    repeat=repeat,
                    warmup_runs=warmup_runs,
                    warmups=warmups,
                    optimization_mode=optimization_mode,
                    optimization_target=optimization_target,
                    monitor_mode=monitor_mode,
                    time_budget_hours=time_budget_hours,
                    thermal_control=thermal_control,
                    results=results,
                    execution_order=execution_order,
                    complete=False,
                )
                pair_index += 1
                if cooldown_seconds > 0:
                    time.sleep(cooldown_seconds)
                continue

            candidate_trial = _add_paired_baseline(candidate_trial, baseline_trial)
            trial_map[candidate.label].append(candidate_trial)
            execution_order.append(_execution_record(candidate, candidate_trial))
            results = _results_from_trials(candidates, trial_map)
            _write_summary(
                output=output,
                cache_path=cache_path,
                fingerprint=fingerprint,
                command=command,
                executor=executor,
                repeat=repeat,
                warmup_runs=warmup_runs,
                warmups=warmups,
                optimization_mode=optimization_mode,
                optimization_target=optimization_target,
                monitor_mode=monitor_mode,
                time_budget_hours=time_budget_hours,
                thermal_control=thermal_control,
                results=results,
                execution_order=execution_order,
                complete=False,
            )
            pair_index += 1
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
        if _deadline_expired(deadline):
            break
    return results, execution_order


def _run_measured_trial(
    command: list[str],
    candidate: RecommendationCandidate,
    *,
    sample_interval_seconds: float,
    hard_kill: bool,
    executor: str,
    use_sudo: bool,
    allow_sudo_auto: bool,
    system_tuning_sudo: bool,
    gpu_tuning_sudo: bool,
    docker_image: str,
    monitor_mode: str,
    optimization_mode: str,
    trial_index: int,
    order_index: int,
    pair_id: str,
    paired_label: str,
    role: str,
    baseline_trial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return_code, run_dir = _run_candidate(
        command,
        candidate,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        system_tuning_sudo=system_tuning_sudo,
        gpu_tuning_sudo=gpu_tuning_sudo,
        docker_image=docker_image,
        monitor_mode=monitor_mode,
        optimization_mode=optimization_mode,
    )
    trial = {
        "run_id": run_dir.name,
        "return_code": return_code,
        "trial_index": trial_index,
        "order_index": order_index,
        "pair_id": pair_id,
        "paired_label": paired_label,
        "thermal_role": role,
        **_run_metrics(run_dir.name),
    }
    if baseline_trial is not None:
        trial = _add_paired_baseline(trial, baseline_trial)
    return trial


def _add_paired_baseline(trial: dict[str, Any], baseline_trial: dict[str, Any]) -> dict[str, Any]:
    trial["paired_baseline_run_id"] = baseline_trial.get("run_id")
    trial["paired_baseline_samples_per_second"] = baseline_trial.get("samples_per_second")
    trial["paired_baseline_gpu_tflops_estimate"] = baseline_trial.get("gpu_tflops_estimate")
    trial["normalized_samples_per_second_ratio"] = _ratio(
        trial.get("samples_per_second"),
        baseline_trial.get("samples_per_second"),
    )
    trial["normalized_gpu_tflops_ratio"] = _ratio(
        trial.get("gpu_tflops_estimate"),
        baseline_trial.get("gpu_tflops_estimate"),
    )
    return trial


def _execution_record(candidate: RecommendationCandidate, trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_index": trial.get("trial_index"),
        "order_index": trial.get("order_index"),
        "label": candidate.label,
        "run_id": trial.get("run_id"),
        "return_code": trial.get("return_code"),
        "pair_id": trial.get("pair_id"),
        "thermal_role": trial.get("thermal_role"),
        "paired_label": trial.get("paired_label"),
    }


def _write_summary(
    *,
    output: str | Path,
    cache_path: str | Path,
    fingerprint: str,
    command: list[str],
    executor: str,
    repeat: int,
    warmup_runs: int,
    warmups: list[dict[str, Any]],
    optimization_mode: str,
    optimization_target: str,
    monitor_mode: str,
    time_budget_hours: float | None,
    thermal_control: bool,
    results: list[dict[str, Any]],
    execution_order: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    ranked = sorted(results, key=_rank_key, reverse=True)
    best = ranked[0] if ranked else None
    decision = _decision_summary(
        ranked,
        optimization_mode=optimization_mode,
        noise_band_percent=DEFAULT_NOISE_BAND_PERCENT,
    )
    summary = {
        "kind": "auto_recommendation",
        "fingerprint": fingerprint,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": command,
        "executor": executor,
        "repeat": repeat,
        "warmup_runs": warmup_runs,
        "warmups": warmups,
        "optimization_mode": optimization_mode,
        "optimization_target": optimization_target,
        "monitor_mode": monitor_mode,
        "schedule": "thermal-controlled-pairs" if thermal_control else "interleaved-rotating",
        "thermal_control": thermal_control,
        "time_budget_hours": time_budget_hours,
        "complete": complete,
        "goal": _goal_text(optimization_mode, optimization_target),
        "decision": decision,
        "diagnostics": _summary_diagnostics(
            optimization_mode=optimization_mode,
            optimization_target=optimization_target,
            thermal_control=thermal_control,
            complete=complete,
            results=ranked,
            execution_order=execution_order,
            decision=decision,
        ),
        "best_label": best["label"] if best else None,
        "recommendation": _recommendation_from_result(best) if best else None,
        "candidates": ranked,
        "execution_order": execution_order,
        "cache_path": str(cache_path),
    }
    write_json(Path(output), summary)
    write_json(Path(cache_path), summary)
    fingerprint_path = Path(cache_path).parent / f"{fingerprint}.json"
    write_json(fingerprint_path, summary)
    return summary


def _rotated_execution_order(candidates: list[RecommendationCandidate], trial_index: int) -> list[RecommendationCandidate]:
    if not candidates:
        return []
    offset = trial_index % len(candidates)
    return [*candidates[offset:], *candidates[:offset]]


def _results_from_trials(
    candidates: list[RecommendationCandidate],
    trial_map: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        _candidate_result(candidate, trial_map[candidate.label])
        for candidate in candidates
        if trial_map.get(candidate.label)
    ]


def load_recommendation(path: str | Path = LATEST_RECOMMENDATION) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"recommendation cache not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def budget_from_recommendation(recommendation: dict[str, Any], fallback: ResourceBudget) -> ResourceBudget:
    raw = recommendation.get("budget")
    if not isinstance(raw, dict):
        return fallback
    return ResourceBudget(
        memory_budget_gb=raw.get("memory_budget_gb"),
        reserve_memory_gb=float(raw.get("reserve_memory_gb") or 0.0),
        reserve_cores=int(raw.get("reserve_cores") or 0),
        cpu_quota_percent=raw.get("cpu_quota_percent"),
        enforce=bool(raw.get("resource_budget_enforced", True)),
    )


def _candidate_plan(
    budget: ResourceBudget,
    *,
    include_gpu: bool,
    optimization_mode: str = "guarded",
    optimization_target: str = "auto",
) -> list[RecommendationCandidate]:
    guard_modes = [("performance", ResourceBudget(enforce=False))] if optimization_mode == "performance" else _guard_modes(budget)
    system_profiles = _default_system_profiles()
    runtime_profiles = [None, *_default_runtime_profiles()]
    gpu_profiles = _default_gpu_profiles(include_gpu=include_gpu, optimization_mode=optimization_mode)

    candidates: list[RecommendationCandidate] = []
    for guard_mode, candidate_budget in guard_modes:
        candidates.append(RecommendationCandidate(f"{guard_mode}:baseline", guard_mode, candidate_budget))
        candidates.append(RecommendationCandidate(f"{guard_mode}:runtime-cpu", guard_mode, candidate_budget, runtime_profile="runtime-cpu-performance"))
        for gpu_profile in gpu_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:{_gpu_label(gpu_profile)}",
                    guard_mode,
                    candidate_budget,
                    gpu_profile=gpu_profile,
                )
            )
        if "runtime-pytorch-gpu-performance" in runtime_profiles and "nvidia-performance" in gpu_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:runtime-gpu+gpu",
                    guard_mode,
                    candidate_budget,
                    runtime_profile="runtime-pytorch-gpu-performance",
                    gpu_profile="nvidia-performance",
                )
            )
        if "runtime-pytorch-max-performance" in runtime_profiles and "nvidia-performance" in gpu_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:runtime-max+gpu",
                    guard_mode,
                    candidate_budget,
                    runtime_profile="runtime-pytorch-max-performance",
                    gpu_profile="nvidia-performance",
                )
            )
        if "runtime-pytorch-gpu-performance" in runtime_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:runtime-gpu",
                    guard_mode,
                    candidate_budget,
                    runtime_profile="runtime-pytorch-gpu-performance",
                )
            )
        if "runtime-pytorch-max-performance" in runtime_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:runtime-max",
                    guard_mode,
                    candidate_budget,
                    runtime_profile="runtime-pytorch-max-performance",
                )
            )
        for system_profile in system_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:{system_profile}",
                    guard_mode,
                    candidate_budget,
                    system_profile=system_profile,
                )
            )
            if "runtime-cpu-performance" in runtime_profiles:
                candidates.append(
                    RecommendationCandidate(
                        f"{guard_mode}:{system_profile}+runtime-cpu",
                        guard_mode,
                        candidate_budget,
                        system_profile=system_profile,
                        runtime_profile="runtime-cpu-performance",
                    )
                )
            for gpu_profile in gpu_profiles:
                candidates.append(
                    RecommendationCandidate(
                        f"{guard_mode}:{system_profile}+{_gpu_label(gpu_profile)}",
                        guard_mode,
                        candidate_budget,
                        system_profile=system_profile,
                        gpu_profile=gpu_profile,
                    )
                )
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:{system_profile}+runtime-max",
                    guard_mode,
                    candidate_budget,
                    system_profile=system_profile,
                    runtime_profile="runtime-pytorch-max-performance"
                    if "runtime-pytorch-max-performance" in runtime_profiles
                    else None,
                )
            )
            if "nvidia-performance" in gpu_profiles:
                candidates.append(
                    RecommendationCandidate(
                        f"{guard_mode}:{system_profile}+runtime-max+gpu",
                        guard_mode,
                        candidate_budget,
                        system_profile=system_profile,
                        runtime_profile="runtime-pytorch-max-performance"
                        if "runtime-pytorch-max-performance" in runtime_profiles
                        else None,
                        gpu_profile="nvidia-performance",
                    )
                )
    return _order_candidates(_dedupe_candidates(candidates), optimization_target)


def _guard_modes(budget: ResourceBudget) -> list[tuple[str, ResourceBudget]]:
    if not budget.enabled:
        return [("unbounded", ResourceBudget())]
    return [("unbounded", ResourceBudget()), ("budgeted", budget)]


def _default_system_profiles() -> list[str]:
    prefix = "windows" if platform.system() == "Windows" else "linux"
    preferred = [
        f"{prefix}-performance",
        f"{prefix}-throughput",
        f"{prefix}-low-latency",
        f"{prefix}-memory-conservative",
        f"{prefix}-cpu-conservative",
    ]
    available = set(available_profiles())
    return [profile for profile in preferred if profile in available]


def _default_runtime_profiles() -> list[str]:
    available = set(available_runtime_profiles())
    preferred = [
        "runtime-cpu-performance",
        "runtime-pytorch-gpu-performance",
        "runtime-pytorch-max-performance",
    ]
    return [profile for profile in preferred if profile in available]


def _default_gpu_profiles(*, include_gpu: bool, optimization_mode: str) -> list[str]:
    if not include_gpu:
        return []
    preferred = ["nvidia-performance"] if optimization_mode == "performance" else ["nvidia-balanced", "nvidia-guard", "nvidia-performance"]
    return [profile for profile in preferred if _nvidia_supported_profile(profile)]


def _gpu_label(profile: str) -> str:
    if profile == "nvidia-performance":
        return "gpu"
    if profile == "nvidia-balanced":
        return "gpu-balanced"
    if profile == "nvidia-guard":
        return "gpu-guard"
    return profile.replace("nvidia-", "gpu-")


def _nvidia_supported() -> bool:
    return _nvidia_supported_profile("nvidia-performance")


def _nvidia_supported_profile(profile: str) -> bool:
    try:
        return bool(recommend_nvidia_tuning(profile).get("supported"))
    except Exception:
        return False


def _order_candidates(candidates: list[RecommendationCandidate], optimization_target: str) -> list[RecommendationCandidate]:
    if optimization_target in {"auto", "mixed"}:
        return candidates
    baselines = [candidate for candidate in candidates if candidate.system_profile is None and candidate.runtime_profile is None and candidate.gpu_profile is None]
    rest = [candidate for candidate in candidates if candidate not in baselines]
    ordered_rest = [
        candidate
        for _, candidate in sorted(
            enumerate(rest),
            key=lambda item: (_target_priority(item[1], optimization_target), item[0]),
        )
    ]
    return [*baselines, *ordered_rest]


def _target_priority(candidate: RecommendationCandidate, optimization_target: str) -> int:
    if optimization_target == "gpu":
        if candidate.guard_mode != "performance" and candidate.gpu_profile in {"nvidia-balanced", "nvidia-guard"} and candidate.system_profile is None and candidate.runtime_profile is None:
            return 0
        if candidate.gpu_profile == "nvidia-performance" and candidate.system_profile is None and candidate.runtime_profile is None:
            return 1
        if candidate.gpu_profile == "nvidia-performance" and candidate.runtime_profile is not None:
            return 2
        if candidate.gpu_profile == "nvidia-performance":
            return 3
        if candidate.runtime_profile == "runtime-pytorch-gpu-performance":
            return 4
        if candidate.gpu_profile is not None:
            return 5
        return 8
    if optimization_target == "cpu":
        if candidate.gpu_profile is not None:
            return 9
        if candidate.runtime_profile == "runtime-cpu-performance" and candidate.system_profile is None:
            return 0
        if candidate.system_profile and "performance" in candidate.system_profile and candidate.runtime_profile is None:
            return 1
        if candidate.system_profile and "throughput" in candidate.system_profile and candidate.runtime_profile is None:
            return 2
        if candidate.system_profile and candidate.runtime_profile == "runtime-cpu-performance":
            return 3
        if candidate.runtime_profile == "runtime-pytorch-max-performance":
            return 4
        return 8
    if optimization_target == "memory":
        if candidate.gpu_profile is not None:
            return 9
        if candidate.system_profile and "memory-conservative" in candidate.system_profile:
            return 0
        if candidate.system_profile and "performance" in candidate.system_profile:
            return 1
        if candidate.system_profile and "throughput" in candidate.system_profile:
            return 2
        if candidate.runtime_profile == "runtime-cpu-performance":
            return 3
        return 8
    return 8


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _dedupe_candidates(candidates: list[RecommendationCandidate]) -> list[RecommendationCandidate]:
    seen = set()
    result = []
    for candidate in candidates:
        key = (
            candidate.guard_mode,
            candidate.budget,
            candidate.system_profile,
            candidate.runtime_profile,
            candidate.gpu_profile,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _run_metrics(run_id: str) -> dict[str, Any]:
    analysis = analyze_run(run_id, RUNS_DIR)
    workload = analysis.get("workload", {})
    memory = analysis.get("memory", {})
    cpu = analysis.get("cpu", {})
    return {
        "status": analysis.get("status"),
        "duration_seconds": _number(workload.get("duration_seconds")),
        "samples_per_second": _number(workload.get("samples_per_second")),
        "step_time_p50_seconds": _number(workload.get("step_time_p50_seconds")),
        "step_time_p95_seconds": _number(workload.get("step_time_p95_seconds")),
        "step_time_p99_seconds": _number(workload.get("step_time_p99_seconds")),
        "step_time_max_seconds": _number(workload.get("step_time_max_seconds")),
        "step_time_sample_count": _number(workload.get("step_time_sample_count")),
        "gpu_tflops_estimate": _number(workload.get("gpu_tflops_estimate")),
        "gpu_matmuls_per_second": _number(workload.get("gpu_matmuls_per_second")),
        "gpu_peak_memory_allocated_mb": _number(workload.get("gpu_peak_memory_allocated_mb")),
        "peak_memory_mb": _number(memory.get("peak_memory_mb")),
        "memory_budget_exceeded": bool(memory.get("memory_budget_exceeded")),
        "peak_process_cpu_percent": _number(cpu.get("observed_peak_process_cpu_percent")),
        "average_process_cpu_percent": _number(cpu.get("observed_average_process_cpu_percent")),
        "peak_system_cpu_percent": _number(cpu.get("observed_peak_system_cpu_percent")),
        "average_system_cpu_percent": _number(cpu.get("observed_average_system_cpu_percent")),
        "system_cpu_percent_p50": _number(cpu.get("observed_system_cpu_percent_p50")),
        "system_cpu_percent_p95": _number(cpu.get("observed_system_cpu_percent_p95")),
        "per_cpu_average_max_percent": _number(cpu.get("per_cpu_average_max_percent")),
        "per_cpu_peak_max_percent": _number(cpu.get("per_cpu_peak_max_percent")),
    }


def _candidate_result(candidate: RecommendationCandidate, trials: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_samples = _median(trials, "normalized_samples_per_second_ratio")
    normalized_tflops = _median(trials, "normalized_gpu_tflops_ratio")
    if normalized_samples is None and any(trial.get("thermal_role") == "baseline_control" for trial in trials):
        normalized_samples = 1.0
    if normalized_tflops is None and any(trial.get("thermal_role") == "baseline_control" for trial in trials):
        normalized_tflops = 1.0
    metrics = {
        "samples_per_second": _median(trials, "samples_per_second"),
        "duration_seconds": _median(trials, "duration_seconds"),
        "step_time_p50_seconds": _median(trials, "step_time_p50_seconds"),
        "step_time_p95_seconds": _median(trials, "step_time_p95_seconds"),
        "step_time_p99_seconds": _median(trials, "step_time_p99_seconds"),
        "step_time_max_seconds": _median(trials, "step_time_max_seconds"),
        "step_time_sample_count": _median(trials, "step_time_sample_count"),
        "gpu_tflops_estimate": _median(trials, "gpu_tflops_estimate"),
        "gpu_matmuls_per_second": _median(trials, "gpu_matmuls_per_second"),
        "gpu_peak_memory_allocated_mb": _median(trials, "gpu_peak_memory_allocated_mb"),
        "normalized_samples_per_second_ratio": normalized_samples,
        "normalized_samples_per_second_percent": _ratio_to_percent(normalized_samples),
        "normalized_gpu_tflops_ratio": normalized_tflops,
        "normalized_gpu_tflops_percent": _ratio_to_percent(normalized_tflops),
        "peak_memory_mb": _median(trials, "peak_memory_mb"),
        "peak_process_cpu_percent": _median(trials, "peak_process_cpu_percent"),
        "average_process_cpu_percent": _median(trials, "average_process_cpu_percent"),
        "peak_system_cpu_percent": _median(trials, "peak_system_cpu_percent"),
        "average_system_cpu_percent": _median(trials, "average_system_cpu_percent"),
        "system_cpu_percent_p50": _median(trials, "system_cpu_percent_p50"),
        "system_cpu_percent_p95": _median(trials, "system_cpu_percent_p95"),
        "per_cpu_average_max_percent": _median(trials, "per_cpu_average_max_percent"),
        "per_cpu_peak_max_percent": _median(trials, "per_cpu_peak_max_percent"),
        "memory_budget_exceeded": any(trial.get("memory_budget_exceeded") for trial in trials),
    }
    return {
        "label": candidate.label,
        "guard_mode": candidate.guard_mode,
        "system_profile": candidate.system_profile,
        "runtime_profile": candidate.runtime_profile,
        "gpu_profile": candidate.gpu_profile,
        "budget": candidate.budget.to_record(),
        "run_ids": [trial.get("run_id") for trial in trials],
        "return_codes": [trial.get("return_code") for trial in trials],
        "trials": [
            {
                "run_id": trial.get("run_id"),
                "return_code": trial.get("return_code"),
                "trial_index": trial.get("trial_index"),
                "order_index": trial.get("order_index"),
                "pair_id": trial.get("pair_id"),
                "thermal_role": trial.get("thermal_role"),
                "paired_baseline_run_id": trial.get("paired_baseline_run_id"),
                "samples_per_second": trial.get("samples_per_second"),
                "duration_seconds": trial.get("duration_seconds"),
                "step_time_p95_seconds": trial.get("step_time_p95_seconds"),
                "step_time_p99_seconds": trial.get("step_time_p99_seconds"),
                "gpu_tflops_estimate": trial.get("gpu_tflops_estimate"),
                "paired_baseline_samples_per_second": trial.get("paired_baseline_samples_per_second"),
                "normalized_samples_per_second_ratio": trial.get("normalized_samples_per_second_ratio"),
            }
            for trial in trials
        ],
        "status": "completed" if all(trial.get("status") == "completed" and trial.get("return_code") == 0 for trial in trials) else "failed",
        "metrics": metrics,
        "score": _score(metrics),
        "reason": _reason(candidate, trials),
    }


def _recommendation_from_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "label": result["label"],
        "guard_mode": result["guard_mode"],
        "system_profile": result.get("system_profile"),
        "runtime_profile": result.get("runtime_profile"),
        "gpu_profile": result.get("gpu_profile"),
        "budget": result.get("budget"),
        "metrics": result.get("metrics"),
        "reason": result.get("reason"),
    }


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
    if item.get("status") != "completed" or item.get("metrics", {}).get("memory_budget_exceeded"):
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    metrics = item.get("metrics", {})
    normalized = metrics.get("normalized_samples_per_second_ratio")
    throughput = metrics.get("samples_per_second")
    gpu_tflops = metrics.get("gpu_tflops_estimate")
    duration = metrics.get("duration_seconds")
    step_p95 = metrics.get("step_time_p95_seconds")
    if normalized is not None:
        return (
            2.0,
            float(normalized),
            float(throughput if throughput is not None else 0.0),
            float(gpu_tflops if gpu_tflops is not None else 0.0),
            -float(step_p95 if step_p95 is not None else float("inf")),
        )
    memory = metrics.get("peak_memory_mb")
    has_throughput = 1.0 if throughput is not None else 0.0
    return (
        has_throughput,
        float(throughput if throughput is not None else 0.0),
        float(gpu_tflops if gpu_tflops is not None else 0.0),
        -float(duration if duration is not None else float("inf")),
        -float(step_p95 if step_p95 is not None else float("inf")),
    )


def _summary_diagnostics(
    *,
    optimization_mode: str,
    optimization_target: str,
    thermal_control: bool,
    complete: bool,
    results: list[dict[str, Any]],
    execution_order: list[dict[str, Any]],
    decision: dict[str, Any] | None = None,
) -> list[str]:
    diagnostics = []
    if optimization_mode == "performance":
        if thermal_control:
            diagnostics.append("Performance sweeps use paired baseline controls and rank by candidate speed divided by nearby baseline speed.")
        else:
            diagnostics.append("Performance sweeps use rotated interleaving so baseline does not always get cold-machine priority.")
        diagnostics.append("Formal launch should use launch-performance to avoid resource-monitoring overhead after recommendation selection.")
    if optimization_target != "auto":
        diagnostics.append(f"Candidate order was prioritized for target={optimization_target}.")
    if not complete:
        diagnostics.append("Time budget or interrupt stopped the sweep before all requested trials completed; recommendation is based on partial results.")
    if results and str(results[0].get("label", "")).endswith(":baseline") and len(results) > 1:
        diagnostics.append("Baseline is currently ranked best; this can be valid, but inspect per-trial execution order before treating it as a tuning failure.")
    if decision:
        interpretation = decision.get("interpretation")
        if interpretation:
            diagnostics.append(str(interpretation))
    if execution_order:
        diagnostics.append(f"Recorded {len(execution_order)} measured trial(s) in execution_order for auditability.")
    return diagnostics


def _score(metrics: dict[str, Any]) -> float | None:
    normalized = metrics.get("normalized_samples_per_second_ratio")
    if normalized is not None:
        return round(float(normalized), 6)
    throughput = metrics.get("samples_per_second")
    if throughput is None:
        return None
    return round(float(throughput), 6)


def _reason(candidate: RecommendationCandidate, trials: list[dict[str, Any]]) -> list[str]:
    normalized = _median(trials, "normalized_samples_per_second_ratio")
    if normalized is None and any(trial.get("thermal_role") == "baseline_control" for trial in trials):
        normalized = 1.0
    metrics = {
        "samples_per_second": _median(trials, "samples_per_second"),
        "duration_seconds": _median(trials, "duration_seconds"),
        "step_time_p95_seconds": _median(trials, "step_time_p95_seconds"),
        "gpu_tflops_estimate": _median(trials, "gpu_tflops_estimate"),
        "normalized_samples_per_second_ratio": normalized,
        "peak_memory_mb": _median(trials, "peak_memory_mb"),
    }
    reasons = [
        f"guard_mode={candidate.guard_mode}",
        f"samples_per_second={metrics['samples_per_second']}",
        f"duration_seconds={metrics['duration_seconds']}",
    ]
    if metrics["gpu_tflops_estimate"] is not None:
        reasons.append(f"gpu_tflops_estimate={metrics['gpu_tflops_estimate']}")
    if metrics["step_time_p95_seconds"] is not None:
        reasons.append(f"step_time_p95_seconds={metrics['step_time_p95_seconds']}")
    if metrics["normalized_samples_per_second_ratio"] is not None:
        reasons.append(f"thermal_normalized_speed={metrics['normalized_samples_per_second_ratio']}")
        reasons.append(f"thermal_normalized_delta_percent={_ratio_to_percent(metrics['normalized_samples_per_second_ratio'])}")
    if metrics["peak_memory_mb"] is not None:
        reasons.append(f"peak_memory_mb={metrics['peak_memory_mb']}")
    if candidate.system_profile:
        reasons.append(f"system_profile={candidate.system_profile}")
    if candidate.runtime_profile:
        reasons.append(f"runtime_profile={candidate.runtime_profile}")
    if candidate.gpu_profile:
        reasons.append(f"gpu_profile={candidate.gpu_profile}")
    return reasons


def _decision_summary(
    ranked_results: list[dict[str, Any]],
    *,
    optimization_mode: str,
    noise_band_percent: float,
) -> dict[str, Any]:
    best = ranked_results[0] if ranked_results else None
    baseline = _baseline_result(ranked_results, optimization_mode=optimization_mode)
    if best is None or baseline is None:
        return {
            "status": "insufficient-data",
            "noise_band_percent": noise_band_percent,
            "interpretation": "No baseline/recommendation pair was available for decision analysis.",
        }
    best_metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
    baseline_metrics = baseline.get("metrics", {}) if isinstance(baseline.get("metrics"), dict) else {}
    samples_delta = _delta_percent(
        best_metrics.get("samples_per_second"),
        baseline_metrics.get("samples_per_second"),
    )
    normalized_delta = best_metrics.get("normalized_samples_per_second_percent")
    primary_delta = normalized_delta if isinstance(normalized_delta, (int, float)) else samples_delta
    duration_delta = _delta_percent(best_metrics.get("duration_seconds"), baseline_metrics.get("duration_seconds"))
    step_p95_delta = _delta_percent(
        best_metrics.get("step_time_p95_seconds"),
        baseline_metrics.get("step_time_p95_seconds"),
    )
    gpu_tflops_delta = _delta_percent(
        best_metrics.get("gpu_tflops_estimate"),
        baseline_metrics.get("gpu_tflops_estimate"),
    )
    best_is_baseline = best.get("label") == baseline.get("label")
    within_noise = primary_delta is None or abs(float(primary_delta)) <= noise_band_percent
    if best_is_baseline:
        status = "baseline-best"
        interpretation = "Baseline ranked best; no tested profile beat baseline for this workload."
    elif within_noise:
        status = "within-noise"
        interpretation = (
            f"Recommended profile is within the +/-{noise_band_percent:.1f}% noise band; "
            "confirm with higher repeat or a longer workload before treating it as a real speedup."
        )
    else:
        status = "meaningful-speedup" if float(primary_delta) > 0 else "regression"
        interpretation = (
            f"Recommended profile beat baseline by {primary_delta:+.3f}% on the primary speed metric, "
            f"outside the +/-{noise_band_percent:.1f}% noise band."
        )
    return {
        "status": status,
        "baseline_label": baseline.get("label"),
        "recommended_label": best.get("label"),
        "noise_band_percent": noise_band_percent,
        "primary_speed_delta_percent": round(float(primary_delta), 6) if isinstance(primary_delta, (int, float)) else None,
        "samples_per_second_delta_percent": samples_delta,
        "normalized_samples_per_second_delta_percent": normalized_delta,
        "duration_delta_percent": duration_delta,
        "step_time_p95_delta_percent": step_p95_delta,
        "gpu_tflops_delta_percent": gpu_tflops_delta,
        "within_noise_band": within_noise,
        "baseline_metrics": {
            "samples_per_second": baseline_metrics.get("samples_per_second"),
            "duration_seconds": baseline_metrics.get("duration_seconds"),
            "step_time_p95_seconds": baseline_metrics.get("step_time_p95_seconds"),
            "gpu_tflops_estimate": baseline_metrics.get("gpu_tflops_estimate"),
        },
        "recommended_metrics": {
            "samples_per_second": best_metrics.get("samples_per_second"),
            "duration_seconds": best_metrics.get("duration_seconds"),
            "step_time_p95_seconds": best_metrics.get("step_time_p95_seconds"),
            "gpu_tflops_estimate": best_metrics.get("gpu_tflops_estimate"),
        },
        "recommendation_reason": best.get("reason", []),
        "interpretation": interpretation,
    }


def _baseline_result(results: list[dict[str, Any]], *, optimization_mode: str) -> dict[str, Any] | None:
    preferred = "performance:baseline" if optimization_mode == "performance" else "unbounded:baseline"
    for item in results:
        if item.get("label") == preferred:
            return item
    for item in results:
        if str(item.get("label", "")).endswith(":baseline"):
            return item
    return None


def _fingerprint(
    command: list[str],
    budget: ResourceBudget,
    executor: str,
    *,
    optimization_mode: str = "guarded",
    optimization_target: str = "auto",
) -> str:
    payload = {
        "command": command,
        "budget": budget.to_record(),
        "executor": executor,
        "optimization_mode": optimization_mode,
        "optimization_target": optimization_target,
        "platform": platform.platform(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _median(items: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(item.get(key)) for item in items]
    values = [value for value in values if value is not None]
    if not values:
        return None
    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        return round(values[midpoint], 6)
    return round((values[midpoint - 1] + values[midpoint]) / 2, 6)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ratio(value: Any, baseline: Any) -> float | None:
    value_number = _number(value)
    baseline_number = _number(baseline)
    if value_number is None or baseline_number is None or baseline_number == 0:
        return None
    return round(value_number / baseline_number, 6)


def _ratio_to_percent(ratio: Any) -> float | None:
    ratio_number = _number(ratio)
    if ratio_number is None:
        return None
    return round((ratio_number - 1.0) * 100.0, 3)


def _delta_percent(value: Any, baseline: Any) -> float | None:
    value_number = _number(value)
    baseline_number = _number(baseline)
    if value_number is None or baseline_number is None or baseline_number == 0:
        return None
    return round(((value_number - baseline_number) / abs(baseline_number)) * 100.0, 3)


def _goal_text(optimization_mode: str, optimization_target: str) -> str:
    if optimization_mode == "performance":
        return (
            "maximize raw throughput without resource guard limits; use low-frequency monitoring and workload metrics "
            f"for ranking; candidate priority target={optimization_target}"
        )
    return f"maximize throughput with duration and memory tie-breakers; candidate priority target={optimization_target}"
