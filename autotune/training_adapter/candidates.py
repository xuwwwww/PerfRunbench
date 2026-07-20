from __future__ import annotations

import os
from typing import Any

from autotune.recommendation.guards import guard_fail, guard_not_available, guard_pass, guard_warn
from autotune.training_adapter.config import (
    capabilities_config,
    knob_configs,
    search_config,
    semantics_config,
    visible_cpu_count,
    world_size,
)
from autotune.training_adapter.errors import TrainingConfigError


STAGE_ORDER = ("input_pipeline", "batch_config", "cpu_runtime", "experimental")
SEMANTIC_STAGE = {
    "dataloader_workers": "input_pipeline",
    "dataloader_prefetch": "input_pipeline",
    "dataloader_pin_memory": "input_pipeline",
    "dataloader_persistent_workers": "input_pipeline",
    "micro_batch_size": "batch_config",
    "gradient_accumulation": "batch_config",
    "cpu_threads": "cpu_runtime",
    "interop_threads": "cpu_runtime",
    "intraop_threads": "cpu_runtime",
    "amp_mode": "experimental",
    "compile_mode": "experimental",
}


def generate_training_candidates(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    knobs = knob_configs(config)
    baseline_values = {str(knob["name"]): knob["baseline"] for knob in knobs}
    baseline_effective = _effective_batch(baseline_values, world_size(config))
    baseline = _candidate(
        "baseline",
        "baseline",
        knobs,
        baseline_values,
        baseline_effective=baseline_effective,
        effective=baseline_effective,
        semantic_change_state="preserved",
        recommendation_eligible=True,
    )
    candidates = [baseline]
    stage_statuses: list[dict[str, Any]] = []
    seen = {_dedupe_key(baseline_values)}
    max_candidates = int(search_config(config).get("max_candidates", 8))
    for stage in _configured_stages(config):
        if len(candidates) - 1 >= max_candidates:
            stage_statuses.append(_stage_status(stage, "skipped_due_to_candidate_limit", 0))
            continue
        generated = _stage_candidates(stage, knobs, baseline_values, config, baseline_effective)
        added = 0
        for item in generated:
            if len(candidates) - 1 >= max_candidates:
                break
            key = _dedupe_key(item["knob_values"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
            added += 1
        status = "completed" if added or generated else "not_applicable"
        if generated and added < len(generated):
            status = "truncated_due_to_candidate_limit"
        stage_statuses.append(_stage_status(stage, status, added))
    return candidates, stage_statuses


def _stage_candidates(
    stage: str,
    knobs: list[dict[str, Any]],
    baseline_values: dict[str, Any],
    config: dict[str, Any],
    baseline_effective: int | None,
) -> list[dict[str, Any]]:
    if stage == "batch_config":
        return _batch_candidates(knobs, baseline_values, config, baseline_effective)
    generated = []
    for knob in knobs:
        name = str(knob["name"])
        if SEMANTIC_STAGE[name] != stage:
            continue
        for value in knob.get("candidates", []):
            if value == baseline_values[name]:
                continue
            values = dict(baseline_values)
            values[name] = value
            guard_results, eligible = _dependency_guards(values, config, stage=stage)
            if stage == "experimental" and not _experimental_supported(name, value, config):
                guard_results.append(
                    guard_fail("candidate", "unsupported_experimental_knob", f"{name}={value!r} was not declared supported.", recoverable=True).to_dict()
                )
                eligible = False
            generated.append(
                _candidate(
                    _candidate_id(name, value),
                    stage,
                    knobs,
                    values,
                    baseline_effective=baseline_effective,
                    effective=_effective_batch(values, world_size(config)),
                    semantic_change_state="preserved",
                    recommendation_eligible=eligible,
                    guard_results=guard_results,
                )
            )
    return generated


def _batch_candidates(
    knobs: list[dict[str, Any]],
    baseline_values: dict[str, Any],
    config: dict[str, Any],
    baseline_effective: int | None,
) -> list[dict[str, Any]]:
    policy = semantics_config(config)
    micro = next((knob for knob in knobs if knob["name"] == "micro_batch_size"), None)
    accum = next((knob for knob in knobs if knob["name"] == "gradient_accumulation"), None)
    generated: list[dict[str, Any]] = []
    if not micro and not accum:
        return generated
    if policy["effective_batch_policy"] == "preserve" and micro and accum and baseline_effective:
        micro_values = _candidate_values(micro)
        accum_values = _candidate_values(accum)
        for micro_value in micro_values:
            for accum_value in accum_values:
                values = dict(baseline_values)
                values["micro_batch_size"] = micro_value
                values["gradient_accumulation"] = accum_value
                effective = _effective_batch(values, world_size(config))
                if effective != baseline_effective or values == baseline_values:
                    continue
                guard_results, eligible = _dependency_guards(values, config, stage="batch_config")
                generated.append(
                    _candidate(
                        f"micro_batch_size_{micro_value}_gradient_accumulation_{accum_value}",
                        "batch_config",
                        knobs,
                        values,
                        baseline_effective=baseline_effective,
                        effective=effective,
                        semantic_change_state="preserved",
                        recommendation_eligible=eligible,
                        guard_results=guard_results,
                    )
                )
        return generated
    for knob in [item for item in (micro, accum) if item is not None]:
        for value in knob.get("candidates", []):
            if value == baseline_values[str(knob["name"])]:
                continue
            values = dict(baseline_values)
            values[str(knob["name"])] = value
            effective = _effective_batch(values, world_size(config))
            changed = effective != baseline_effective
            guard_results, eligible = _dependency_guards(values, config, stage="batch_config")
            if changed:
                guard_results.append(
                    guard_warn(
                        "candidate",
                        "effective_batch_changed",
                        "Effective batch changed with explicit semantic_change_allowed=true; candidate is not eligible for default recommendation.",
                        recoverable=True,
                    ).to_dict()
                )
                eligible = False
            generated.append(
                _candidate(
                    _candidate_id(str(knob["name"]), value),
                    "batch_config",
                    knobs,
                    values,
                    baseline_effective=baseline_effective,
                    effective=effective,
                    semantic_change_state="changed_explicitly_allowed" if changed else "preserved",
                    recommendation_eligible=eligible,
                    guard_results=guard_results,
                )
            )
    return generated


def _dependency_guards(values: dict[str, Any], config: dict[str, Any], *, stage: str) -> tuple[list[dict[str, Any]], bool]:
    guards = []
    eligible = True
    workers = _optional_int(values.get("dataloader_workers"))
    prefetch = values.get("dataloader_prefetch")
    persistent = values.get("dataloader_persistent_workers")
    pin_memory = values.get("dataloader_pin_memory")
    for key in ("micro_batch_size", "gradient_accumulation", "cpu_threads", "interop_threads", "intraop_threads"):
        if key not in values:
            continue
        value = _optional_int(values.get(key))
        if value is None or value <= 0:
            guards.append(guard_fail("candidate", "invalid_positive_knob", f"{key} must be a positive integer.", recoverable=True).to_dict())
            eligible = False
    if workers is not None and workers < 0:
        guards.append(guard_fail("candidate", "invalid_dataloader_workers", "dataloader_workers must be non-negative.", recoverable=True).to_dict())
        eligible = False
    if workers == 0 and prefetch not in {None, 0, "0"}:
        guards.append(guard_fail("candidate", "prefetch_requires_workers", "dataloader_prefetch requires dataloader_workers > 0.", recoverable=True).to_dict())
        eligible = False
    if workers == 0 and persistent is True:
        guards.append(
            guard_fail("candidate", "persistent_workers_requires_workers", "dataloader_persistent_workers requires dataloader_workers > 0.", recoverable=True).to_dict()
        )
        eligible = False
    if pin_memory is True:
        guards.append(
            guard_not_available(
                "candidate",
                "pin_memory_cpu_benefit_unverified",
                "dataloader_pin_memory is not assumed beneficial on CPU without measured evidence.",
            ).to_dict()
        )
    max_cpus = visible_cpu_count()
    for key in ("cpu_threads", "interop_threads", "intraop_threads"):
        value = _optional_int(values.get(key))
        if value is not None and value > max_cpus:
            guards.append(guard_fail("candidate", "thread_count_exceeds_visible_cpu", f"{key}={value} exceeds visible CPU count {max_cpus}.", recoverable=True).to_dict())
            eligible = False
    if eligible:
        guards.append(guard_pass("candidate", "dependency_guards_passed", f"{stage} dependency guards passed.").to_dict())
    return guards, eligible


def _experimental_supported(name: str, value: Any, config: dict[str, Any]) -> bool:
    if value in {None, "disabled", "none", False}:
        return True
    capabilities = capabilities_config(config)
    if name == "amp_mode":
        return str(value) in set(str(item) for item in capabilities.get("amp_modes", ["disabled"]))
    if name == "compile_mode":
        return str(value) in set(str(item) for item in capabilities.get("compile_modes", ["disabled"]))
    return True


def _candidate(
    candidate_id: str,
    stage: str,
    knobs: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    baseline_effective: int | None,
    effective: int | None,
    semantic_change_state: str,
    recommendation_eligible: bool,
    guard_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": candidate_id,
        "stage": stage,
        "knobs": knobs,
        "knob_values": values,
        "baseline_effective_batch_size": baseline_effective,
        "effective_batch_size": effective,
        "semantic_change_state": semantic_change_state,
        "recommendation_eligible": recommendation_eligible,
        "guard_results": guard_results or [],
    }


def _effective_batch(values: dict[str, Any], world: int) -> int | None:
    micro = _optional_int(values.get("micro_batch_size"))
    accum = _optional_int(values.get("gradient_accumulation"))
    if micro is None or accum is None:
        return None
    return micro * accum * world


def _candidate_values(knob: dict[str, Any]) -> list[Any]:
    values = [knob["baseline"], *list(knob.get("candidates", []))]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _configured_stages(config: dict[str, Any]) -> list[str]:
    raw = search_config(config).get("stages", list(STAGE_ORDER))
    if not isinstance(raw, list):
        raise TrainingConfigError("training.search.stages must be a list")
    stages = [str(item) for item in raw]
    for stage in stages:
        if stage not in STAGE_ORDER:
            raise TrainingConfigError(f"unsupported training search stage: {stage}")
    return stages


def _candidate_id(name: str, value: Any) -> str:
    return f"{name}_{str(value).replace(' ', '_').replace('/', '_')}"


def _dedupe_key(values: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in values.items()))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _stage_status(stage: str, status: str, count: int) -> dict[str, Any]:
    return {"schema_version": "1.0", "stage": stage, "stage_status": status, "candidate_count": count}
