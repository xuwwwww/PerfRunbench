from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotune.recommendation.fingerprints import collect_environment_fingerprint
from autotune.recommendation.guards import GuardResult, guard_dicts, guard_fail, guard_pass, utc_now
from autotune.recommendation.schemas import (
    APPLY_SCOPES,
    OBJECTIVE_DIRECTIONS,
    RECOMMENDATION_SCHEMA_VERSION,
    RECOMMENDATION_STATUS_ELIGIBLE,
    RecommendationError,
    atomic_write_json,
    file_sha256,
    load_json,
    require_versioned_top_level,
    stable_digest,
    validate_launch_spec,
)


RECOMMENDATIONS_DIR = Path(".autotuneai") / "recommendations"
DEFAULT_EXPIRY_DAYS = 14


def recommendation_path(run_id: str, *, root: str | Path = RECOMMENDATIONS_DIR) -> Path:
    _validate_run_id(run_id)
    return Path(root) / run_id / "recommendation.json"


def write_recommendation_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    validate_recommendation_bundle(bundle, validate_passport_digest=False)
    return atomic_write_json(path, bundle)


def load_recommendation_bundle(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise RecommendationError("recommendation bundle must be a JSON object")
    require_versioned_top_level(raw)
    return raw


def create_recommendation_bundle(
    *,
    source_run_id: str,
    source_passport_path: str | Path,
    status: str,
    objective: dict[str, Any],
    baseline_configuration: dict[str, Any],
    selected_configuration: dict[str, Any],
    environment_fingerprint: dict[str, Any] | None = None,
    workload_fingerprint: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    statistical_evidence: dict[str, Any] | None = None,
    apply_scope: str = "process",
    rollback_configuration: dict[str, Any] | None = None,
    expires_at: str | None = None,
    source_result_path: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_run_id(source_run_id)
    if status not in RECOMMENDATION_STATUS_ELIGIBLE:
        raise RecommendationError(f"recommendation status {status!r} is not applyable")
    if apply_scope not in APPLY_SCOPES:
        raise RecommendationError(f"apply_scope must be one of {sorted(APPLY_SCOPES)}")
    _validate_objective(objective)
    validate_selected_configuration(selected_configuration)
    validate_selected_configuration(baseline_configuration, selected=False)
    passport_path = Path(source_passport_path)
    if not passport_path.exists():
        raise RecommendationError(f"source passport not found: {passport_path}")
    source_passport_sha256 = file_sha256(passport_path)
    result_sha256 = file_sha256(source_result_path) if source_result_path and Path(source_result_path).exists() else None
    created_at = utc_now()
    bundle = {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": "",
        "source_run_id": source_run_id,
        "source_result_path": str(source_result_path) if source_result_path else None,
        "source_result_sha256": result_sha256,
        "source_passport_path": str(passport_path),
        "source_passport_sha256": source_passport_sha256,
        "status": status,
        "objective": objective,
        "baseline_configuration": baseline_configuration,
        "selected_configuration": selected_configuration,
        "environment_fingerprint": environment_fingerprint or collect_environment_fingerprint(),
        "workload_fingerprint": workload_fingerprint or {"schema_version": RECOMMENDATION_SCHEMA_VERSION, "digest": "not_applicable"},
        "constraints": constraints or {},
        "statistical_evidence": statistical_evidence or {},
        "apply_scope": apply_scope,
        "rollback_configuration": rollback_configuration or {"type": "process_exit", "system_restore_run_id": None},
        "created_at": created_at,
        "expires_at": expires_at,
        "provenance": provenance or {},
        "guard_results": [],
        "recommendation_wording": "Best observed eligible configuration under the tested target, workload, and measurement protocol.",
    }
    bundle["bundle_id"] = stable_digest(
        {
            "source_run_id": source_run_id,
            "source_passport_sha256": source_passport_sha256,
            "selected_configuration": selected_configuration,
            "objective": objective,
        }
    )[:24]
    validation = validate_recommendation_bundle(bundle)
    bundle["guard_results"] = validation["guard_results"]
    if validation["status"] != "pass":
        raise RecommendationError("recommendation bundle validation failed")
    return bundle


def generate_bundle_from_source(
    *,
    passport_path: str | Path,
    source_run_id: str,
    output_path: str | Path | None = None,
    source_result_path: str | Path | None = None,
    selected_configuration_path: str | Path | None = None,
    baseline_configuration_path: str | Path | None = None,
    objective_metric: str = "throughput",
    objective_direction: str = "maximize",
    apply_scope: str = "process",
) -> dict[str, Any]:
    passport = _load_versioned_or_plain_json(passport_path)
    source_result = _load_versioned_or_plain_json(source_result_path) if source_result_path else {}
    selected_configuration = _load_selected_configuration(source_result, selected_configuration_path)
    baseline_configuration = _load_baseline_configuration(source_result, baseline_configuration_path, selected_configuration)
    source_status = _source_status(passport, source_result)
    _reject_non_applyable_source(source_status, passport, source_result)
    bundle = create_recommendation_bundle(
        source_run_id=source_run_id,
        source_passport_path=passport_path,
        source_result_path=source_result_path,
        status="eligible_preliminary" if source_status == "eligible_preliminary" else "recommended",
        objective={"metric": objective_metric, "direction": objective_direction},
        baseline_configuration=baseline_configuration,
        selected_configuration=selected_configuration,
        environment_fingerprint=source_result.get("environment_fingerprint") or passport.get("environment_fingerprint") or collect_environment_fingerprint(),
        workload_fingerprint=source_result.get("workload_fingerprint") or passport.get("workload_fingerprint") or {"schema_version": RECOMMENDATION_SCHEMA_VERSION, "digest": "not_applicable"},
        constraints=source_result.get("constraints") or passport.get("constraints") or {},
        statistical_evidence=source_result.get("statistical_evidence") or passport.get("statistical_evidence") or {},
        apply_scope=apply_scope,
        rollback_configuration=source_result.get("rollback_configuration") or {"type": "process_exit", "system_restore_run_id": None},
        provenance=_source_provenance(passport, source_result),
    )
    if output_path:
        write_recommendation_bundle(bundle, output_path)
    return bundle


def validate_recommendation_bundle(
    bundle: dict[str, Any],
    *,
    validate_passport_digest: bool = True,
) -> dict[str, Any]:
    guards: list[GuardResult] = []
    try:
        require_versioned_top_level(bundle)
    except RecommendationError as exc:
        guards.append(guard_fail("preflight", "invalid_schema", str(exc)))
        return _validation_result(guards)
    status = bundle.get("status")
    if status not in RECOMMENDATION_STATUS_ELIGIBLE:
        guards.append(guard_fail("recommendation", "non_applyable_status", f"Recommendation status {status!r} is not applyable.", recoverable=True))
    else:
        guards.append(guard_pass("recommendation", "applyable_status", f"Recommendation status is {status}."))
    objective = bundle.get("objective", {})
    try:
        _validate_objective(objective)
    except RecommendationError as exc:
        guards.append(guard_fail("recommendation", "invalid_objective", str(exc)))
    else:
        guards.append(guard_pass("recommendation", "objective_valid", "Objective metric and direction are valid."))
    try:
        validate_selected_configuration(bundle.get("selected_configuration", {}))
    except RecommendationError as exc:
        guards.append(guard_fail("candidate", "selected_configuration_incomplete", str(exc), recoverable=True))
    else:
        guards.append(guard_pass("candidate", "selected_configuration_complete", "Selected configuration is executable."))
    try:
        validate_selected_configuration(bundle.get("baseline_configuration", {}), selected=False)
    except RecommendationError as exc:
        guards.append(guard_fail("candidate", "baseline_configuration_incomplete", str(exc), recoverable=True))
    else:
        guards.append(guard_pass("candidate", "baseline_configuration_complete", "Baseline configuration is executable."))
    statistical_evidence = bundle.get("statistical_evidence")
    if not isinstance(statistical_evidence, dict) or not statistical_evidence:
        guards.append(guard_fail("recommendation", "missing_statistical_evidence", "Applyable recommendations require statistical evidence.", recoverable=True))
    elif statistical_evidence.get("status") == "recommended" or statistical_evidence.get("classification") == "improved":
        guards.append(guard_pass("recommendation", "statistical_evidence_recommended", "Statistical evidence supports recommendation."))
    else:
        guards.append(
            guard_fail(
                "recommendation",
                "inconclusive_statistical_evidence",
                f"Statistical evidence status={statistical_evidence.get('status')!r} classification={statistical_evidence.get('classification')!r} is not applyable.",
                recoverable=True,
            )
        )
    source_passport_path = bundle.get("source_passport_path")
    expected_digest = bundle.get("source_passport_sha256")
    if validate_passport_digest:
        if not isinstance(source_passport_path, str) or not source_passport_path:
            guards.append(guard_fail("preflight", "missing_source_passport", "source_passport_path is missing."))
        else:
            path = Path(source_passport_path)
            if not path.exists():
                guards.append(guard_fail("preflight", "missing_source_passport", f"source Passport does not exist: {path}.", recoverable=True))
            elif expected_digest and file_sha256(path) == expected_digest:
                guards.append(guard_pass("preflight", "source_passport_digest_match", "source Passport digest matches."))
            else:
                guards.append(guard_fail("preflight", "source_passport_digest_mismatch", "source Passport digest mismatch.", recoverable=True))
    provenance = bundle.get("provenance", {})
    result_origin = provenance.get("result_origin") if isinstance(provenance, dict) else None
    if result_origin in {"fixture", "test_fixture", "synthetic", "example"}:
        guards.append(guard_fail("recommendation", "fixture_source_rejected", f"Source result_origin={result_origin} is not applyable."))
    elif result_origin != "measured":
        guards.append(guard_fail("recommendation", "measured_source_required", f"Source result_origin={result_origin or 'not_available'} is not measured.", recoverable=True))
    else:
        guards.append(guard_pass("recommendation", "source_origin_applyable", "Source result_origin is measured."))
    return _validation_result(guards)


def validate_selected_configuration(raw: dict[str, Any], *, selected: bool = True) -> None:
    if not isinstance(raw, dict):
        raise RecommendationError("configuration must be an object")
    launch_spec = raw.get("launch_spec")
    if not isinstance(launch_spec, dict):
        raise RecommendationError("configuration requires launch_spec")
    validate_launch_spec(launch_spec)
    if selected and not raw.get("configuration_id"):
        raise RecommendationError("selected configuration requires configuration_id")


def _validation_result(guards: list[GuardResult]) -> dict[str, Any]:
    status = "fail" if any(result.status == "fail" for result in guards) else "pass"
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "status": status,
        "guard_results": guard_dicts(guards),
    }


def _validate_objective(objective: dict[str, Any]) -> None:
    if not isinstance(objective, dict):
        raise RecommendationError("objective must be an object")
    metric = objective.get("metric")
    direction = objective.get("direction")
    if not isinstance(metric, str) or not metric:
        raise RecommendationError("objective.metric is required")
    if direction not in OBJECTIVE_DIRECTIONS:
        raise RecommendationError("objective.direction must be 'maximize' or 'minimize'")


def _load_selected_configuration(source_result: dict[str, Any], path: str | Path | None) -> dict[str, Any]:
    if path:
        raw = load_json(path)
    else:
        raw = source_result.get("selected_configuration") or source_result.get("recommendation", {}).get("selected_configuration")
    if not isinstance(raw, dict):
        raise RecommendationError("selected configuration is required to generate a bundle")
    return raw


def _load_baseline_configuration(source_result: dict[str, Any], path: str | Path | None, selected: dict[str, Any]) -> dict[str, Any]:
    if path:
        raw = load_json(path)
    else:
        raw = source_result.get("baseline_configuration") or source_result.get("recommendation", {}).get("baseline_configuration") or selected
    if not isinstance(raw, dict):
        raise RecommendationError("baseline configuration is required to generate a bundle")
    return raw


def _load_versioned_or_plain_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RecommendationError(f"source JSON must be an object: {path}")
    return raw


def _source_status(passport: dict[str, Any], source_result: dict[str, Any]) -> str:
    for key in ("recommendation_status", "status"):
        value = source_result.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("claim_eligibility_state", "verification_status", "recommendation_state"):
        value = passport.get(key)
        if isinstance(value, str) and value:
            return value
    return "invalid"


def _reject_non_applyable_source(status: str, passport: dict[str, Any], source_result: dict[str, Any]) -> None:
    if status in {"inconclusive", "insufficient-data", "invalid", "failed", "incomplete", "fixture", "test_fixture"}:
        raise RecommendationError(f"source status {status!r} is not applyable")
    origin = _source_provenance(passport, source_result).get("result_origin")
    if origin in {"fixture", "test_fixture", "synthetic", "example"}:
        raise RecommendationError(f"source result_origin {origin!r} is not applyable")
    verification = passport.get("verification_status")
    if verification and verification != "passed":
        raise RecommendationError(f"Passport verification_status {verification!r} is not applyable")


def _source_provenance(passport: dict[str, Any], source_result: dict[str, Any]) -> dict[str, Any]:
    provenance = {}
    for raw in (passport.get("provenance"), source_result.get("provenance")):
        if isinstance(raw, dict):
            provenance.update(raw)
    for key in ("result_origin", "target_architecture", "workload_adapter", "protocol_digest"):
        for raw in (source_result, passport):
            value = raw.get(key)
            if value is not None and key not in provenance:
                provenance[key] = value
    return provenance


def _validate_run_id(run_id: str) -> None:
    if not run_id or any(char in run_id for char in "\\/:*?\"<>|"):
        raise RecommendationError("run_id must be a non-empty path-safe identifier")
