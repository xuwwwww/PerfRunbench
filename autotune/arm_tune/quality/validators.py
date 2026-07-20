from __future__ import annotations

import json
import re
from typing import Any

from autotune.arm_tune.quality.diagnostics import categorize_quality_failure
from autotune.arm_tune.quality.normalization import NormalizationPolicy, normalize_output
from autotune.arm_tune.quality.schemas import QualityCase, QualityCaseResult, QualityValidatorSpec, ValidatorResult


def validate_quality_case(
    case: QualityCase,
    generated_output: str,
    *,
    request_status: str = "completed",
    stream_parse_status: str = "parsed",
    output_artifact: str = "",
    repetition_index: int | None = None,
    normalization_policy: NormalizationPolicy | None = None,
) -> QualityCaseResult:
    normalization = normalize_output(generated_output, normalization_policy)
    if request_status != "completed" or stream_parse_status != "parsed":
        reason = f"request_status={request_status}, stream_parse_status={stream_parse_status}"
        categories = categorize_quality_failure(
            case=case,
            generated_output=generated_output,
            request_status=request_status,
            stream_parse_status=stream_parse_status,
            validator_results=[],
            normalization=normalization,
            parse_error="",
        )
        return QualityCaseResult(
            case_id=case.case_id,
            category=case.category,
            required=case.required,
            status="execution_failed",
            request_status=request_status,
            stream_parse_status=stream_parse_status,
            validator_results=[],
            output_artifact=output_artifact,
            generated_output=generated_output,
            raw_output_sha256=normalization.raw_output_sha256,
            normalized_output=normalization.normalized_output,
            normalized_output_sha256=normalization.normalized_output_sha256,
            normalization_policy_version=normalization.normalization_policy_version,
            normalization_applied=normalization.normalization_applied,
            normalization_steps=normalization.normalization_steps,
            parser_status="execution_not_validated",
            validation_input=normalization.validation_input,
            failure_categories=categories,
            repetition_index=repetition_index,
            failure_reasons=[reason],
        )
    validation_text = normalization.normalized_output if normalization.validation_input == "normalized" else generated_output
    parsed, parse_error = _parse_json(validation_text)
    parser_status = _parser_status(parse_error, normalization)
    results = [run_validator(validator, validation_text, parsed, parse_error) for validator in case.validators]
    critical_failures = [result for result in results if result.critical and result.status != "passed"]
    categories = categorize_quality_failure(
        case=case,
        generated_output=generated_output,
        request_status=request_status,
        stream_parse_status=stream_parse_status,
        validator_results=results,
        normalization=normalization,
        parse_error=parse_error,
    )
    status = "failed" if critical_failures else "passed"
    return QualityCaseResult(
        case_id=case.case_id,
        category=case.category,
        required=case.required,
        status=status,
        request_status=request_status,
        stream_parse_status=stream_parse_status,
        validator_results=results,
        output_artifact=output_artifact,
        generated_output=generated_output,
        raw_output_sha256=normalization.raw_output_sha256,
        normalized_output=normalization.normalized_output,
        normalized_output_sha256=normalization.normalized_output_sha256,
        normalization_policy_version=normalization.normalization_policy_version,
        normalization_applied=normalization.normalization_applied,
        normalization_steps=normalization.normalization_steps,
        parser_status=parser_status,
        validation_input=normalization.validation_input,
        failure_categories=categories,
        repetition_index=repetition_index,
        failure_reasons=[result.message for result in critical_failures],
    )


def run_validator(
    validator: QualityValidatorSpec,
    output: str,
    parsed_json: Any,
    parse_error: str,
) -> ValidatorResult:
    kind = validator.validator_type
    try:
        if kind == "valid_json":
            return _result(validator, "passed" if parse_error == "" else "failed", validator.expected, parse_error or "valid JSON", "valid JSON" if not parse_error else parse_error)
        if kind == "json_schema":
            return _json_schema(validator, parsed_json, parse_error)
        if kind == "exact_field":
            return _exact_field(validator, parsed_json, parse_error)
        if kind == "allowed_values":
            return _allowed_values(validator, parsed_json, parse_error)
        if kind == "required_substrings":
            return _substrings(validator, output, parsed_json, parse_error, forbidden=False)
        if kind == "forbidden_substrings":
            return _substrings(validator, output, parsed_json, parse_error, forbidden=True)
        if kind == "regex":
            return _regex(validator, output, parsed_json, parse_error)
        if kind == "numeric_range":
            return _numeric_range(validator, parsed_json, parse_error)
        if kind == "evidence_line_ids":
            return _evidence_line_ids(validator, parsed_json, parse_error)
        if kind == "nonempty_output":
            observed = output.strip()
            return _result(validator, "passed" if observed else "failed", "nonempty output", observed, "output is nonempty" if observed else "output is empty")
    except Exception as exc:
        return _result(validator, "failed", validator.expected, None, f"validator error: {exc}")
    return _result(validator, "failed", validator.expected, None, f"unsupported validator type: {kind}")


def _parse_json(output: str) -> tuple[Any, str]:
    try:
        return json.loads(output), ""
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _parser_status(parse_error: str, normalization) -> str:
    if normalization.parser_status == "normalization_rejected":
        return "normalization_rejected"
    if parse_error:
        return "invalid_json"
    return "normalized_json_valid" if normalization.validation_input == "normalized" else "raw_json_valid"


def _json_schema(validator: QualityValidatorSpec, parsed_json: Any, parse_error: str) -> ValidatorResult:
    if parse_error:
        return _result(validator, "failed", validator.expected, parse_error, parse_error)
    expected = validator.expected if isinstance(validator.expected, dict) else {}
    required = expected.get("required", [])
    properties = expected.get("properties", {})
    if not isinstance(parsed_json, dict):
        return _result(validator, "failed", expected, type(parsed_json).__name__, "JSON root is not an object")
    missing = [key for key in required if key not in parsed_json]
    if missing:
        return _result(validator, "failed", expected, parsed_json, f"missing required field(s): {missing}")
    type_failures: list[str] = []
    for field, spec in properties.items():
        if field not in parsed_json or not isinstance(spec, dict) or "type" not in spec:
            continue
        if not _matches_json_type(parsed_json[field], str(spec["type"])):
            type_failures.append(field)
    if type_failures:
        return _result(validator, "failed", expected, parsed_json, f"field type mismatch: {type_failures}")
    return _result(validator, "passed", expected, parsed_json, "JSON schema validation passed")


def _exact_field(validator: QualityValidatorSpec, parsed_json: Any, parse_error: str) -> ValidatorResult:
    if parse_error:
        return _result(validator, "failed", validator.expected, parse_error, parse_error)
    observed = _field_value(parsed_json, validator.field)
    status = "passed" if observed == validator.expected else "failed"
    return _result(validator, status, validator.expected, observed, "exact field matched" if status == "passed" else "exact field mismatch")


def _allowed_values(validator: QualityValidatorSpec, parsed_json: Any, parse_error: str) -> ValidatorResult:
    if parse_error:
        return _result(validator, "failed", validator.expected, parse_error, parse_error)
    expected = validator.expected if isinstance(validator.expected, list) else []
    observed = _field_value(parsed_json, validator.field)
    status = "passed" if observed in expected else "failed"
    return _result(validator, status, expected, observed, "allowed value matched" if status == "passed" else "value is not allowed")


def _substrings(validator: QualityValidatorSpec, output: str, parsed_json: Any, parse_error: str, *, forbidden: bool) -> ValidatorResult:
    haystack = output if validator.field is None else str(_field_value(parsed_json, validator.field) if not parse_error else "")
    expected = validator.expected if isinstance(validator.expected, list) else []
    matches = [item for item in expected if str(item) in haystack]
    if forbidden:
        status = "failed" if matches else "passed"
        message = f"forbidden substring(s) present: {matches}" if matches else "no forbidden substrings present"
        observed = matches
    else:
        missing = [item for item in expected if str(item) not in haystack]
        status = "failed" if missing else "passed"
        message = f"required substring(s) missing: {missing}" if missing else "required substrings present"
        observed = matches
    return _result(validator, status, expected, observed, message)


def _regex(validator: QualityValidatorSpec, output: str, parsed_json: Any, parse_error: str) -> ValidatorResult:
    target = output if validator.field is None else str(_field_value(parsed_json, validator.field) if not parse_error else "")
    pattern = str(validator.expected or "")
    matched = bool(re.search(pattern, target))
    return _result(validator, "passed" if matched else "failed", pattern, target, "regex matched" if matched else "regex did not match")


def _numeric_range(validator: QualityValidatorSpec, parsed_json: Any, parse_error: str) -> ValidatorResult:
    if parse_error:
        return _result(validator, "failed", validator.expected, parse_error, parse_error)
    expected = validator.expected if isinstance(validator.expected, dict) else {}
    observed = _field_value(parsed_json, validator.field)
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return _result(validator, "failed", expected, observed, "observed value is not numeric")
    lower = expected.get("min")
    upper = expected.get("max")
    passed = (lower is None or observed >= lower) and (upper is None or observed <= upper)
    return _result(validator, "passed" if passed else "failed", expected, observed, "numeric range passed" if passed else "numeric range failed")


def _evidence_line_ids(validator: QualityValidatorSpec, parsed_json: Any, parse_error: str) -> ValidatorResult:
    if parse_error:
        return _result(validator, "failed", validator.expected, parse_error, parse_error)
    expected = validator.expected if isinstance(validator.expected, dict) else {}
    field = validator.field or "evidence_line_ids"
    observed = _field_value(parsed_json, field)
    if not isinstance(observed, list) or any(not isinstance(item, str) for item in observed):
        return _result(validator, "failed", expected, observed, "evidence line IDs must be a string list")
    allowed = set(str(item) for item in expected.get("allowed", []))
    required = set(str(item) for item in expected.get("required", []))
    observed_set = set(observed)
    invented = sorted(observed_set - allowed) if allowed else []
    missing = sorted(required - observed_set)
    if invented:
        return _result(validator, "failed", expected, observed, f"invented evidence line ID(s): {invented}")
    if missing:
        return _result(validator, "failed", expected, observed, f"missing required evidence line ID(s): {missing}")
    return _result(validator, "passed", expected, observed, "evidence line IDs are valid")


def _field_value(parsed_json: Any, field: str | None) -> Any:
    if field is None:
        return parsed_json
    current = parsed_json
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _result(validator: QualityValidatorSpec, status: str, expected: Any, observed: Any, message: str) -> ValidatorResult:
    return ValidatorResult(
        validator_type=validator.validator_type,
        status=status,
        expected=expected,
        observed=observed,
        message=message,
        critical=validator.critical,
        required=validator.required,
    )
