from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest, write_json
from autotune.training_tuner.batch_size import (
    BatchSizeTuningError,
    find_numeric_assignment,
    replace_assignment_value,
)


@dataclass(frozen=True)
class KnobTrial:
    key: str
    value: int
    run_id: str
    run_dir: str
    return_code: int
    safe: bool
    score: float | None
    reason: str
    training_metrics: dict[str, Any]
    resource_summary: dict[str, Any]


def tune_training_knobs(
    config_file: str | Path,
    knob_values: dict[str, list[int]],
    command: list[str],
    budget: ResourceBudget,
    output: str | Path,
    *,
    objective: str = "throughput",
    min_final_accuracy: float | None = None,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    docker_image: str = "python:3.12-slim",
) -> dict[str, Any]:
    if not knob_values:
        raise BatchSizeTuningError("at least one knob must be provided")
    if not command:
        raise BatchSizeTuningError("command cannot be empty")
    config_path = Path(config_file)
    original_text = config_path.read_text(encoding="utf-8")
    current_text = original_text
    summary: dict[str, Any] = {
        "config_file": str(config_path),
        "objective": objective,
        "knob_order": list(knob_values),
        "stages": [],
        "final_recommendation": {},
    }

    try:
        for key, values in knob_values.items():
            if not values:
                raise BatchSizeTuningError(f"knob {key!r} has no candidate values")
            config_path.write_text(current_text, encoding="utf-8")
            original_value, current_assignment = find_numeric_assignment(config_path, key)
            stage_trials: list[KnobTrial] = []
            best_trial: KnobTrial | None = None
            best_text = current_text
            for value in values:
                candidate_assignment = replace_assignment_value(current_assignment, value)
                candidate_text = current_text.replace(current_assignment, candidate_assignment, 1)
                config_path.write_text(candidate_text, encoding="utf-8")
                return_code, run_dir = _run_candidate(
                    command,
                    budget,
                    sample_interval_seconds=sample_interval_seconds,
                    hard_kill=hard_kill,
                    executor=executor,
                    use_sudo=use_sudo,
                    allow_sudo_auto=allow_sudo_auto,
                    docker_image=docker_image,
                )
                resource_summary = _load_json(Path(run_dir) / "resource_summary.json", default={})
                training_metrics = _load_json(Path(run_dir) / "training_metrics.json", default={})
                safe, reason = _trial_safety(
                    return_code,
                    resource_summary,
                    training_metrics,
                    min_final_accuracy=min_final_accuracy,
                )
                score = _score_trial(objective, safe, training_metrics, resource_summary)
                manifest = load_manifest(Path(run_dir))
                trial = KnobTrial(
                    key=key,
                    value=value,
                    run_id=manifest["run_id"],
                    run_dir=str(run_dir),
                    return_code=return_code,
                    safe=safe,
                    score=score,
                    reason=reason,
                    training_metrics=training_metrics,
                    resource_summary=resource_summary,
                )
                stage_trials.append(trial)
                if safe and (best_trial is None or _better_score(score, best_trial.score)):
                    best_trial = trial
                    best_text = candidate_text
            if best_trial is None:
                raise BatchSizeTuningError(f"no safe candidate found for knob {key!r}")
            current_text = best_text
            summary["stages"].append(
                {
                    "key": key,
                    "original_value": original_value,
                    "recommended_value": best_trial.value,
                    "recommended_run_id": best_trial.run_id,
                    "trials": [_trial_to_dict(trial) for trial in stage_trials],
                }
            )
            summary["final_recommendation"][key] = best_trial.value
    finally:
        config_path.write_text(original_text, encoding="utf-8")

    write_json(Path(output), summary)
    return summary


def _run_candidate(
    command: list[str],
    budget: ResourceBudget,
    *,
    sample_interval_seconds: float,
    hard_kill: bool,
    executor: str,
    use_sudo: bool,
    allow_sudo_auto: bool,
    docker_image: str,
) -> tuple[int, Path]:
    from autotune.resource.workload_runner import run_with_budget

    return run_with_budget(
        command,
        budget,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        docker_image=docker_image,
    )


def parse_knob_specs(specs: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for spec in specs:
        if "=" not in spec:
            raise BatchSizeTuningError(f"invalid knob spec {spec!r}; expected key=v1,v2")
        key, values_text = spec.split("=", 1)
        values = [int(item.strip()) for item in values_text.split(",") if item.strip()]
        if not values:
            raise BatchSizeTuningError(f"invalid knob spec {spec!r}; no values found")
        result[key.strip()] = values
    return result


def _trial_safety(
    return_code: int,
    resource_summary: dict[str, Any],
    training_metrics: dict[str, Any],
    *,
    min_final_accuracy: float | None,
) -> tuple[bool, str]:
    if return_code != 0:
        return False, f"command returned {return_code}"
    if resource_summary.get("memory_budget_exceeded") is True:
        return False, "memory budget exceeded"
    if not training_metrics:
        return False, "training metrics missing"
    if min_final_accuracy is not None and float(training_metrics.get("final_accuracy", 0.0)) < min_final_accuracy:
        return False, f"final_accuracy below threshold {min_final_accuracy}"
    return True, "completed within budget"


def _score_trial(
    objective: str,
    safe: bool,
    training_metrics: dict[str, Any],
    resource_summary: dict[str, Any],
) -> float | None:
    if not safe:
        return None
    if objective == "throughput":
        return float(training_metrics.get("samples_per_second", 0.0))
    if objective == "duration":
        return -float(training_metrics.get("duration_seconds", 0.0))
    if objective == "memory":
        peak = resource_summary.get("peak_rss_mb") or resource_summary.get("peak_cgroup_memory_current_mb") or 0.0
        return -float(peak)
    raise BatchSizeTuningError(f"unsupported objective: {objective}")


def _better_score(candidate: float | None, current: float | None) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate > current


def _trial_to_dict(trial: KnobTrial) -> dict[str, Any]:
    return {
        "key": trial.key,
        "value": trial.value,
        "run_id": trial.run_id,
        "run_dir": trial.run_dir,
        "return_code": trial.return_code,
        "safe": trial.safe,
        "score": trial.score,
        "reason": trial.reason,
        "training_metrics": trial.training_metrics,
        "resource_summary": trial.resource_summary,
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
