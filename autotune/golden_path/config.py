from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.resource.budget import ResourceBudget
from autotune.utils.config import load_config


DEFAULT_CACHE_ROOT = Path(".autotuneai") / "cache" / "recommendations"
DEFAULT_RETUNE_POLICY = "on_hard_drift_or_missing"
DEFAULT_CANARY_POLICY = "on_new_or_drift"
PUBLIC_PRESETS = ("quick", "standard", "deep")
PRESET_DEFAULTS: dict[str, dict[str, int | float]] = {
    "quick": {
        "warmup_runs": 0,
        "search_runs": 1,
        "confirmation_runs": 1,
        "max_candidates": 3,
        "budget_minutes": 5.0,
    },
    "standard": {
        "warmup_runs": 1,
        "search_runs": 1,
        "confirmation_runs": 3,
        "max_candidates": 8,
        "budget_minutes": 20.0,
    },
    "deep": {
        "warmup_runs": 1,
        "search_runs": 2,
        "confirmation_runs": 5,
        "max_candidates": 16,
        "budget_minutes": 60.0,
    },
    # Backward-compatible aliases retained for existing checked-in configs.
    "thorough": {
        "warmup_runs": 1,
        "search_runs": 2,
        "confirmation_runs": 5,
        "max_candidates": 16,
        "budget_minutes": 60.0,
    },
    "reuse-only": {
        "warmup_runs": 0,
        "search_runs": 0,
        "confirmation_runs": 0,
        "max_candidates": 0,
        "budget_minutes": 5.0,
    },
}


class GoldenPathConfigError(ValueError):
    pass


def load_golden_path_config(path: str | Path) -> dict[str, Any]:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise GoldenPathConfigError("golden-path config must be a JSON/YAML object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise GoldenPathConfigError(f"golden-path config schema_version must be {SCHEMA_VERSION!r}")
    return payload


def default_command_config(
    *,
    preset: str = "quick",
    budget_minutes: float | None = None,
) -> dict[str, Any]:
    if preset not in PUBLIC_PRESETS:
        raise GoldenPathConfigError(f"unsupported public tuning preset: {preset}")
    defaults = PRESET_DEFAULTS[preset]
    visible_cpus = _visible_cpu_count()
    thread_counts = _unique_positive([1, max(1, visible_cpus // 2), visible_cpus])
    candidates = [
        {
            "id": f"threads_{threads}",
            "env": {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "NUMEXPR_NUM_THREADS": str(threads),
            },
        }
        for threads in thread_counts
    ]
    selected_budget = float(defaults["budget_minutes"]) if budget_minutes is None else _positive_budget(budget_minutes)
    return {
        "schema_version": SCHEMA_VERSION,
        "workload": {
            "adapter": "generic-command",
            "cwd": ".",
            "timeout_seconds": 60,
            "identity_files": [],
        },
        "objective": {"metric": "wall_time_seconds", "direction": "minimize"},
        "metrics": {"source": "process_wall_time"},
        "tuning": {
            "preset": preset,
            "budget_minutes": selected_budget,
            "min_valid_trials": 1 if preset == "quick" else 2,
            "noise_band_percent": 2.0,
            "shuffle_seed": 42,
            "candidates": candidates,
        },
        "cache": {
            "root": str(DEFAULT_CACHE_ROOT),
            "retune_policy": DEFAULT_RETUNE_POLICY,
            "canary_policy": DEFAULT_CANARY_POLICY,
            "ttl_days": 14,
        },
        "apply": {
            "canary_runs": 1,
            "lease_minutes": 30,
            "correctness_validators": [
                {"type": "exit_code", "expected": 0},
                {"type": "cleanup_passed"},
            ],
        },
    }


def with_tuning_overrides(
    config: dict[str, Any],
    *,
    preset: str | None = None,
    budget_minutes: float | None = None,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    tuning = payload.setdefault("tuning", {})
    if not isinstance(tuning, dict):
        raise GoldenPathConfigError("tuning must be an object")
    if preset is not None:
        if preset not in PUBLIC_PRESETS:
            raise GoldenPathConfigError(f"unsupported public tuning preset: {preset}")
        tuning["preset"] = preset
        for key, value in PRESET_DEFAULTS[preset].items():
            tuning[key] = value
    if budget_minutes is not None:
        tuning["budget_minutes"] = _positive_budget(budget_minutes)
    return payload


def save_golden_path_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(config, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def cache_root(config: dict[str, Any]) -> Path:
    cache = _section(config, "cache")
    return Path(str(cache.get("root") or DEFAULT_CACHE_ROOT))


def budget_from_config(config: dict[str, Any]) -> ResourceBudget:
    return ResourceBudget.from_config(config)


def objective_from_config(config: dict[str, Any]) -> dict[str, str]:
    objective = _section(config, "objective")
    metrics = _section(config, "metrics")
    metric = objective.get("metric") or metrics.get("objective_metric") or "throughput"
    direction = objective.get("direction") or metrics.get("objective_direction") or "maximize"
    if direction not in {"maximize", "minimize"}:
        raise GoldenPathConfigError("objective.direction must be 'maximize' or 'minimize'")
    return {"metric": str(metric), "direction": str(direction)}


def workload_section(config: dict[str, Any]) -> dict[str, Any]:
    return _section(config, "workload")


def metrics_section(config: dict[str, Any]) -> dict[str, Any]:
    return _section(config, "metrics")


def apply_section(config: dict[str, Any]) -> dict[str, Any]:
    return _section(config, "apply")


def tuning_section(config: dict[str, Any]) -> dict[str, Any]:
    tuning = dict(_section(config, "tuning"))
    preset = str(tuning.get("preset") or "quick")
    if preset not in PRESET_DEFAULTS:
        raise GoldenPathConfigError(f"unsupported tuning preset: {preset}")
    merged = dict(PRESET_DEFAULTS[preset])
    merged.update(tuning)
    merged["preset"] = preset
    for key in ("warmup_runs", "search_runs", "confirmation_runs", "max_candidates"):
        merged[key] = max(0, int(merged.get(key, 0)))
    merged["cooldown_seconds"] = max(0.0, float(merged.get("cooldown_seconds", 0.0)))
    merged["noise_band_percent"] = max(0.0, float(merged.get("noise_band_percent", 2.0)))
    merged["min_valid_trials"] = max(1, int(merged.get("min_valid_trials", 2)))
    merged["shuffle_seed"] = int(merged.get("shuffle_seed", 0))
    merged["budget_minutes"] = _positive_budget(merged.get("budget_minutes", 5.0))
    return merged


def cache_policy(config: dict[str, Any]) -> dict[str, str]:
    cache = _section(config, "cache")
    retune_policy = str(cache.get("retune_policy") or DEFAULT_RETUNE_POLICY)
    canary_policy = str(cache.get("canary_policy") or DEFAULT_CANARY_POLICY)
    allowed_retune = {"never", "on_hard_drift_or_missing", "always"}
    allowed_canary = {"never", "on_new_or_drift", "always"}
    if retune_policy not in allowed_retune:
        raise GoldenPathConfigError(f"unsupported retune_policy: {retune_policy}")
    if canary_policy not in allowed_canary:
        raise GoldenPathConfigError(f"unsupported canary_policy: {canary_policy}")
    return {"retune_policy": retune_policy, "canary_policy": canary_policy}


def ttl_days(config: dict[str, Any]) -> int:
    cache = _section(config, "cache")
    return max(1, int(cache.get("ttl_days", 14)))


def candidate_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    tuning = tuning_section(config)
    raw_candidates = _section(config, "tuning").get("candidates", [])
    candidates = [_candidate("baseline", {})]
    if not isinstance(raw_candidates, list):
        raise GoldenPathConfigError("tuning.candidates must be a list")
    seen = {"baseline"}
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise GoldenPathConfigError(f"tuning.candidates[{index}] must be an object")
        candidate_id = str(item.get("id") or f"candidate_{index + 1}")
        if candidate_id == "baseline":
            continue
        if candidate_id in seen:
            continue
        candidates.append(_candidate(candidate_id, item.get("env", {})))
        seen.add(candidate_id)
    max_candidates = int(tuning["max_candidates"])
    return [candidates[0], *candidates[1 : max_candidates + 1]]


def _candidate(candidate_id: str, raw_env: Any) -> dict[str, Any]:
    if not isinstance(raw_env, dict):
        raise GoldenPathConfigError(f"candidate {candidate_id} env must be an object")
    env = {str(key): str(value) for key, value in raw_env.items()}
    return {"id": candidate_id, "env": env}


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    raw = config.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise GoldenPathConfigError(f"{name} must be an object")
    return raw


def _positive_budget(value: Any) -> float:
    if isinstance(value, bool):
        raise GoldenPathConfigError("tuning budget_minutes must be > 0")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GoldenPathConfigError("tuning budget_minutes must be a number") from exc
    if parsed <= 0:
        raise GoldenPathConfigError("tuning budget_minutes must be > 0")
    return parsed


def _visible_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _unique_positive(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        normalized = max(1, int(value))
        if normalized not in result:
            result.append(normalized)
    return result

