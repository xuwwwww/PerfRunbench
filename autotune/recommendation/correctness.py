from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autotune.recommendation.guards import GuardResult, guard_fail, guard_pass
from autotune.recommendation.schemas import file_sha256


ALLOWED_VALIDATORS = {
    "exit_code",
    "required_metric",
    "metric_finite",
    "metric_min",
    "metric_max",
    "output_file_exists",
    "output_nonempty",
    "artifact_digest_match",
    "request_completed",
    "stream_parsed",
    "cleanup_passed",
}


def evaluate_correctness_contract(
    contract: dict[str, Any] | None,
    observation: dict[str, Any],
    *,
    stage: str = "candidate",
) -> list[GuardResult]:
    raw_contract = contract or {}
    validators = raw_contract.get("validators", [])
    if not isinstance(validators, list):
        return [guard_fail(stage, "invalid_correctness_contract", "Correctness contract validators must be a list.")]
    results: list[GuardResult] = []
    for index, validator in enumerate(validators):
        if not isinstance(validator, dict):
            results.append(guard_fail(stage, "invalid_correctness_validator", f"Validator {index} must be an object."))
            continue
        validator_type = validator.get("type")
        if validator_type not in ALLOWED_VALIDATORS:
            results.append(
                guard_fail(
                    stage,
                    "validator_not_allowed",
                    f"Correctness validator {validator_type!r} is not allowlisted.",
                    evidence_refs=[f"correctness_contract.validators[{index}]"],
                )
            )
            continue
        results.append(_evaluate_validator(str(validator_type), validator, observation, stage=stage, index=index))
    if not validators:
        results.append(guard_pass(stage, "correctness_contract_empty", "No correctness validators were configured."))
    return results


def correctness_passed(results: list[GuardResult]) -> bool:
    return not any(result.status == "fail" for result in results)


def load_metrics_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    metrics = observation.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    metrics_path = observation.get("metrics_path")
    if isinstance(metrics_path, str) and metrics_path:
        path = Path(metrics_path)
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
            if isinstance(loaded, dict):
                if isinstance(loaded.get("metrics"), dict):
                    return loaded["metrics"]
                return loaded
    return {}


def _evaluate_validator(
    validator_type: str,
    validator: dict[str, Any],
    observation: dict[str, Any],
    *,
    stage: str,
    index: int,
) -> GuardResult:
    metrics = load_metrics_from_observation(observation)
    ref = f"correctness_contract.validators[{index}]"
    if validator_type == "exit_code":
        expected = int(validator.get("expected", 0))
        actual = observation.get("exit_code")
        if actual == expected:
            return guard_pass(stage, "exit_code", f"exit_code matched {expected}.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", f"exit_code {actual} did not match {expected}.", evidence_refs=[ref], recoverable=True)
    if validator_type == "required_metric":
        metric = _metric_name(validator)
        if metric in metrics and metrics.get(metric) is not None:
            return guard_pass(stage, "required_metric", f"Required metric {metric} is present.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", f"Required metric {metric} is missing.", evidence_refs=[ref], recoverable=True)
    if validator_type == "metric_finite":
        metric = _metric_name(validator)
        value = _number(metrics.get(metric))
        if value is not None and math.isfinite(value):
            return guard_pass(stage, "metric_finite", f"Metric {metric} is finite.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", f"Metric {metric} is not finite.", evidence_refs=[ref], recoverable=True)
    if validator_type == "metric_min":
        metric = _metric_name(validator)
        value = _number(metrics.get(metric))
        minimum = _number(validator.get("min"))
        if value is not None and minimum is not None and value >= minimum:
            return guard_pass(stage, "metric_min", f"Metric {metric} is >= {minimum}.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", f"Metric {metric} is below minimum {minimum}.", evidence_refs=[ref], recoverable=True)
    if validator_type == "metric_max":
        metric = _metric_name(validator)
        value = _number(metrics.get(metric))
        maximum = _number(validator.get("max"))
        if value is not None and maximum is not None and value <= maximum:
            return guard_pass(stage, "metric_max", f"Metric {metric} is <= {maximum}.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", f"Metric {metric} is above maximum {maximum}.", evidence_refs=[ref], recoverable=True)
    if validator_type == "output_file_exists":
        path = _path_from_validator(validator, observation)
        if path and path.exists() and path.is_file():
            return guard_pass(stage, "output_file_exists", f"Output file exists: {path}.", evidence_refs=[ref, str(path)])
        return guard_fail(stage, "rejected_correctness", f"Output file is missing: {path}.", evidence_refs=[ref], recoverable=True)
    if validator_type == "output_nonempty":
        path = _path_from_validator(validator, observation)
        output_field = validator.get("field")
        if path:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return guard_pass(stage, "output_nonempty", f"Output file is nonempty: {path}.", evidence_refs=[ref, str(path)])
            return guard_fail(stage, "rejected_correctness", f"Output file is empty or missing: {path}.", evidence_refs=[ref], recoverable=True)
        if isinstance(output_field, str) and observation.get(output_field):
            return guard_pass(stage, "output_nonempty", f"Output field {output_field} is nonempty.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", "Required output is empty.", evidence_refs=[ref], recoverable=True)
    if validator_type == "artifact_digest_match":
        path = _path_from_validator(validator, observation)
        expected = validator.get("sha256") or validator.get("expected")
        if path and path.exists() and expected and file_sha256(path) == expected:
            return guard_pass(stage, "artifact_digest_match", f"Artifact digest matched: {path}.", evidence_refs=[ref, str(path)])
        return guard_fail(stage, "rejected_correctness", f"Artifact digest mismatch: {path}.", evidence_refs=[ref], recoverable=True)
    if validator_type == "request_completed":
        if observation.get("request_completed") is True:
            return guard_pass(stage, "request_completed", "Request completed.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", "Request did not complete.", evidence_refs=[ref], recoverable=True)
    if validator_type == "stream_parsed":
        if observation.get("stream_parsed") is True:
            return guard_pass(stage, "stream_parsed", "Stream parsed.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", "Stream was not parsed.", evidence_refs=[ref], recoverable=True)
    if validator_type == "cleanup_passed":
        if observation.get("cleanup_passed") is True:
            return guard_pass(stage, "cleanup_passed", "Cleanup passed.", evidence_refs=[ref])
        return guard_fail(stage, "rejected_correctness", "Cleanup did not pass.", evidence_refs=[ref], recoverable=True)
    return guard_fail(stage, "validator_not_allowed", f"Correctness validator {validator_type!r} is not implemented.", evidence_refs=[ref])


def _metric_name(validator: dict[str, Any]) -> str:
    metric = validator.get("metric") or validator.get("field")
    if not isinstance(metric, str) or not metric:
        return ""
    return metric


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _path_from_validator(validator: dict[str, Any], observation: dict[str, Any]) -> Path | None:
    path_value = validator.get("path")
    if isinstance(path_value, str) and path_value:
        return Path(path_value)
    name = validator.get("name")
    outputs = observation.get("outputs", {})
    if isinstance(name, str) and isinstance(outputs, dict) and isinstance(outputs.get(name), str):
        return Path(outputs[name])
    return None
