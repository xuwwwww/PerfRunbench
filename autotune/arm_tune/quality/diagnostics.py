from __future__ import annotations

import json
from collections import Counter
from typing import Any

from autotune.arm_tune.quality.normalization import NormalizationResult
from autotune.arm_tune.quality.schemas import QualityCase, QualityCaseResult, ValidatorResult


QUALITY_FAILURE_CATEGORIES = {
    "response_format_violation",
    "markdown_fence_present",
    "invalid_json",
    "missing_required_field",
    "wrong_status",
    "wrong_failure_stage",
    "wrong_root_cause",
    "missing_required_evidence",
    "invented_evidence",
    "truncated_output",
    "request_failure",
    "stream_parse_failure",
    "unsupported_structured_output",
    "model_capability_failure",
}


def categorize_quality_failure(
    *,
    case: QualityCase,
    generated_output: str,
    request_status: str,
    stream_parse_status: str,
    validator_results: list[ValidatorResult],
    normalization: NormalizationResult,
    parse_error: str,
) -> list[str]:
    categories: set[str] = set()
    if request_status != "completed":
        categories.add("request_failure")
    if stream_parse_status != "parsed":
        categories.add("stream_parse_failure")
    if "```" in generated_output:
        categories.add("markdown_fence_present")
        categories.add("response_format_violation")
    if normalization.parser_status == "normalization_rejected":
        categories.add("response_format_violation")
    if parse_error:
        categories.add("invalid_json")
        categories.add("response_format_violation")
        if _looks_truncated(normalization.normalized_output):
            categories.add("truncated_output")
    parsed = _parse_if_possible(normalization.normalized_output if normalization.validation_input == "normalized" else generated_output)
    expected_status = case.expected.get("status")
    observed_status = parsed.get("status") if isinstance(parsed, dict) else None
    if isinstance(expected_status, str) and observed_status is not None and observed_status != expected_status:
        categories.add("wrong_status")
        if expected_status == "failure" and observed_status in {"success", "warning"}:
            categories.add("wrong_failure_stage")
    for index, result in enumerate(validator_results):
        if result.status == "passed":
            continue
        spec = case.validators[index] if index < len(case.validators) else None
        field = spec.field if spec is not None else None
        message = result.message.lower()
        if result.validator_type == "valid_json":
            categories.add("invalid_json")
            categories.add("response_format_violation")
        elif result.validator_type == "json_schema":
            if "missing required field" in message:
                categories.add("missing_required_field")
            else:
                categories.add("response_format_violation")
        elif result.validator_type == "exact_field" and field == "status":
            categories.add("wrong_status")
        elif result.validator_type == "exact_field" and field == "failure_stage":
            categories.add("wrong_failure_stage")
        elif result.validator_type == "exact_field" and field == "root_cause_category":
            categories.add("wrong_root_cause")
        elif result.validator_type == "allowed_values" and result.observed != result.expected:
            if field == "failure_stage":
                categories.add("wrong_failure_stage")
            elif field == "root_cause_category":
                categories.add("wrong_root_cause")
            else:
                categories.add("wrong_status")
        elif result.validator_type in {"required_substrings", "regex"}:
            categories.add("wrong_root_cause")
        elif result.validator_type == "evidence_line_ids":
            if "invented evidence" in message:
                categories.add("invented_evidence")
            if "missing required evidence" in message:
                categories.add("missing_required_evidence")
        elif result.validator_type == "nonempty_output":
            categories.add("truncated_output")
    if any(result.critical and result.status != "passed" for result in validator_results):
        categories.add("model_capability_failure")
    return sorted(category for category in categories if category in QUALITY_FAILURE_CATEGORIES)


def failure_category_histogram(results: list[QualityCaseResult]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result.failure_categories)
    return dict(sorted(counter.items()))


def _parse_if_possible(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _looks_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True
    if stripped.startswith("[") and not stripped.endswith("]"):
        return True
    return False
