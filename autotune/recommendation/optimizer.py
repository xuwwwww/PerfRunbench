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
from autotune.resource.workload_runner import run_with_budget
from autotune.runtime_tuner.env import available_runtime_profiles
from autotune.system_tuner.runtime import available_profiles


RECOMMENDATIONS_DIR = Path(".autotuneai") / "recommendations"
LATEST_RECOMMENDATION = RECOMMENDATIONS_DIR / "latest.json"


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
) -> dict[str, Any]:
    if not command:
        raise ValueError("command cannot be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if optimization_mode not in {"guarded", "performance"}:
        raise ValueError("optimization_mode must be 'guarded' or 'performance'")
    effective_budget = ResourceBudget(enforce=False) if optimization_mode == "performance" else budget
    fingerprint = _fingerprint(command, effective_budget, executor, optimization_mode=optimization_mode)
    candidates = _candidate_plan(effective_budget, include_gpu=include_gpu, optimization_mode=optimization_mode)
    if max_candidates is not None:
        candidates = candidates[: max(1, max_candidates)]

    warmups = []
    if warmup_runs:
        warmup_candidate = RecommendationCandidate("warmup:baseline", "warmup", ResourceBudget())
        for warmup_index in range(warmup_runs):
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
            )
            warmups.append({"run_id": run_dir.name, "return_code": return_code, "index": warmup_index})
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)

    results = []
    for candidate in candidates:
        trials = []
        for trial_index in range(repeat):
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
            )
            metrics = _run_metrics(run_dir.name)
            trials.append({"run_id": run_dir.name, "return_code": return_code, **metrics})
            if cooldown_seconds > 0 and not (candidate == candidates[-1] and trial_index == repeat - 1):
                time.sleep(cooldown_seconds)
        results.append(_candidate_result(candidate, trials))

    ranked = sorted(results, key=_rank_key, reverse=True)
    best = ranked[0] if ranked else None
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
        "goal": _goal_text(optimization_mode),
        "best_label": best["label"] if best else None,
        "recommendation": _recommendation_from_result(best) if best else None,
        "candidates": ranked,
        "cache_path": str(cache_path),
    }
    write_json(Path(output), summary)
    write_json(Path(cache_path), summary)
    fingerprint_path = Path(cache_path).parent / f"{fingerprint}.json"
    write_json(fingerprint_path, summary)
    return summary


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
) -> tuple[int, Path]:
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
) -> list[RecommendationCandidate]:
    guard_modes = [("performance", ResourceBudget(enforce=False))] if optimization_mode == "performance" else _guard_modes(budget)
    system_profiles = _default_system_profiles()
    runtime_profiles = [None, *_default_runtime_profiles()]
    gpu_profiles = [None]
    if include_gpu and _nvidia_supported():
        gpu_profiles.append("nvidia-performance")

    candidates: list[RecommendationCandidate] = []
    for guard_mode, candidate_budget in guard_modes:
        candidates.append(RecommendationCandidate(f"{guard_mode}:baseline", guard_mode, candidate_budget))
        candidates.append(RecommendationCandidate(f"{guard_mode}:runtime-cpu", guard_mode, candidate_budget, runtime_profile="runtime-cpu-performance"))
        if "nvidia-performance" in gpu_profiles:
            candidates.append(
                RecommendationCandidate(
                    f"{guard_mode}:gpu",
                    guard_mode,
                    candidate_budget,
                    gpu_profile="nvidia-performance",
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
            if "nvidia-performance" in gpu_profiles:
                candidates.append(
                    RecommendationCandidate(
                        f"{guard_mode}:{system_profile}+gpu",
                        guard_mode,
                        candidate_budget,
                        system_profile=system_profile,
                        gpu_profile="nvidia-performance",
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
    return _dedupe_candidates(candidates)


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


def _nvidia_supported() -> bool:
    try:
        return bool(recommend_nvidia_tuning("nvidia-performance").get("supported"))
    except Exception:
        return False


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
    return {
        "label": candidate.label,
        "guard_mode": candidate.guard_mode,
        "system_profile": candidate.system_profile,
        "runtime_profile": candidate.runtime_profile,
        "gpu_profile": candidate.gpu_profile,
        "budget": candidate.budget.to_record(),
        "run_ids": [trial.get("run_id") for trial in trials],
        "return_codes": [trial.get("return_code") for trial in trials],
        "status": "completed" if all(trial.get("status") == "completed" and trial.get("return_code") == 0 for trial in trials) else "failed",
        "metrics": {
            "samples_per_second": _median(trials, "samples_per_second"),
            "duration_seconds": _median(trials, "duration_seconds"),
            "gpu_tflops_estimate": _median(trials, "gpu_tflops_estimate"),
            "gpu_matmuls_per_second": _median(trials, "gpu_matmuls_per_second"),
            "gpu_peak_memory_allocated_mb": _median(trials, "gpu_peak_memory_allocated_mb"),
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
        },
        "score": _score(trials),
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


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    if item.get("status") != "completed" or item.get("metrics", {}).get("memory_budget_exceeded"):
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    metrics = item.get("metrics", {})
    throughput = metrics.get("samples_per_second")
    gpu_tflops = metrics.get("gpu_tflops_estimate")
    duration = metrics.get("duration_seconds")
    memory = metrics.get("peak_memory_mb")
    has_throughput = 1.0 if throughput is not None else 0.0
    return (
        has_throughput,
        float(throughput if throughput is not None else 0.0),
        float(gpu_tflops if gpu_tflops is not None else 0.0),
        -float(duration if duration is not None else float("inf")),
    )


def _score(trials: list[dict[str, Any]]) -> float | None:
    throughput = _median(trials, "samples_per_second")
    if throughput is None:
        return None
    return round(float(throughput), 6)


def _reason(candidate: RecommendationCandidate, trials: list[dict[str, Any]]) -> list[str]:
    metrics = {
        "samples_per_second": _median(trials, "samples_per_second"),
        "duration_seconds": _median(trials, "duration_seconds"),
        "gpu_tflops_estimate": _median(trials, "gpu_tflops_estimate"),
        "peak_memory_mb": _median(trials, "peak_memory_mb"),
    }
    reasons = [
        f"guard_mode={candidate.guard_mode}",
        f"samples_per_second={metrics['samples_per_second']}",
        f"duration_seconds={metrics['duration_seconds']}",
    ]
    if metrics["gpu_tflops_estimate"] is not None:
        reasons.append(f"gpu_tflops_estimate={metrics['gpu_tflops_estimate']}")
    if metrics["peak_memory_mb"] is not None:
        reasons.append(f"peak_memory_mb={metrics['peak_memory_mb']}")
    if candidate.system_profile:
        reasons.append(f"system_profile={candidate.system_profile}")
    if candidate.runtime_profile:
        reasons.append(f"runtime_profile={candidate.runtime_profile}")
    if candidate.gpu_profile:
        reasons.append(f"gpu_profile={candidate.gpu_profile}")
    return reasons


def _fingerprint(
    command: list[str],
    budget: ResourceBudget,
    executor: str,
    *,
    optimization_mode: str = "guarded",
) -> str:
    payload = {
        "command": command,
        "budget": budget.to_record(),
        "executor": executor,
        "optimization_mode": optimization_mode,
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


def _goal_text(optimization_mode: str) -> str:
    if optimization_mode == "performance":
        return "maximize raw throughput without resource guard limits; use low-frequency monitoring and workload metrics for ranking"
    return "maximize throughput with duration and memory tie-breakers"
