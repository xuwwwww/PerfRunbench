from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.schemas import QualityCase, QualitySuiteManifest, QualityValidatorSpec
from autotune.arm_tune.schemas import to_plain


CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_quality_suite(path: str | Path) -> tuple[str, list[QualityCase], QualitySuiteManifest, dict[str, Any]]:
    suite_path = Path(path)
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("quality suite must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArmTuneConfigError(f"unsupported quality suite schema_version: {payload.get('schema_version')!r}")
    suite_id = _required_text(payload, "suite_id")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ArmTuneConfigError("quality suite must contain cases")
    cases: list[QualityCase] = []
    seen: set[str] = set()
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ArmTuneConfigError("quality suite cases must be objects")
        case = _case_from_payload(item)
        if case.case_id in seen:
            raise ArmTuneConfigError(f"duplicate quality case_id: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    digest = quality_suite_digest(payload)
    manifest = QualitySuiteManifest(
        suite_id=suite_id,
        suite_digest=digest,
        case_count=len(cases),
        required_case_count=sum(1 for case in cases if case.required),
        source_path=str(suite_path),
    )
    return suite_id, cases, manifest, payload


def quality_suite_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prompt_suite_from_quality_cases(cases: list[QualityCase]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Derived ArmTune quality prompt suite snapshot.",
        "prompts": [
            {
                "prompt_id": case.case_id,
                "prompt": case.prompt,
                "expected_response_mode": case.response_format,
                "max_predict_tokens": case.maximum_predicted_tokens,
                "deterministic_settings": dict(case.generation_settings),
            }
            for case in cases
        ],
    }


def suite_snapshot_payload(suite_id: str, cases: list[QualityCase]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "cases": [case.to_dict() for case in cases],
    }


def _case_from_payload(payload: dict[str, Any]) -> QualityCase:
    case_id = _required_text(payload, "case_id")
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ArmTuneConfigError(f"quality case_id is not filesystem-safe: {case_id!r}")
    validators_payload = payload.get("validators")
    if not isinstance(validators_payload, list) or not validators_payload:
        raise ArmTuneConfigError(f"quality case {case_id} must contain validators")
    validators = [_validator_from_payload(item, case_id) for item in validators_payload]
    expected = payload.get("expected", {})
    if not isinstance(expected, dict):
        raise ArmTuneConfigError(f"quality case {case_id} expected must be an object")
    generation = payload.get("generation_settings", {})
    if generation is None:
        generation = {}
    if not isinstance(generation, dict):
        raise ArmTuneConfigError(f"quality case {case_id} generation_settings must be an object")
    maximum = payload.get("maximum_predicted_tokens")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ArmTuneConfigError(f"quality case {case_id} maximum_predicted_tokens must be a positive integer")
    return QualityCase(
        case_id=case_id,
        category=_required_text(payload, "category"),
        prompt=_required_text(payload, "prompt"),
        response_format=str(payload.get("response_format", "json")),
        required=bool(payload.get("required", True)),
        expected=dict(expected),
        validators=validators,
        maximum_predicted_tokens=maximum,
        generation_settings=dict(generation),
    )


def _validator_from_payload(payload: Any, case_id: str) -> QualityValidatorSpec:
    if not isinstance(payload, dict):
        raise ArmTuneConfigError(f"quality case {case_id} validators must be objects")
    validator_type = _required_text(payload, "type")
    return QualityValidatorSpec(
        validator_type=validator_type,
        expected=to_plain(payload.get("expected")),
        field=str(payload["field"]) if "field" in payload and payload.get("field") is not None else None,
        critical=bool(payload.get("critical", True)),
        required=bool(payload.get("required", True)),
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArmTuneConfigError(f"{key} must be a non-empty string")
    return value.strip()
