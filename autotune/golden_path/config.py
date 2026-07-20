from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.resource.budget import ResourceBudget
from autotune.utils.config import load_config


DEFAULT_CACHE_ROOT = Path(".autotuneai") / "cache" / "recommendations"
DEFAULT_RETUNE_POLICY = "on_hard_drift_or_missing"
DEFAULT_CANARY_POLICY = "on_new_or_drift"
PRESET_DEFAULTS: dict[str, dict[str, int]] = {
    "quick": {"warmup_runs": 0, "search_runs": 1, "confirmation_runs": 2, "max_candidates": 4},
    "standard": {"warmup_runs": 1, "search_runs": 1, "confirmation_runs": 3, "max_candidates": 8},
    "thorough": {"warmup_runs": 1, "search_runs": 2, "confirmation_runs": 5, "max_candidates": 16},
    "reuse-only": {"warmup_runs": 0, "search_runs": 0, "confirmation_runs": 0, "max_candidates": 0},
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

