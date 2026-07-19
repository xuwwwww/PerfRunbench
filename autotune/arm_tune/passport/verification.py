from __future__ import annotations

from pathlib import Path

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.passport.schemas import PerformancePassportVerification
from autotune.arm_tune.targets.protocol import read_json


CLAIM_STATES = {
    "eligible_preliminary",
    "ineligible_quality",
    "ineligible_provenance",
    "ineligible_statistics",
    "ineligible_cleanup",
    "fixture_only",
}


def verify_passport(path: str | Path) -> PerformancePassportVerification:
    issues: list[str] = []
    checked = 0
    try:
        payload = read_json(path)
    except Exception as exc:
        return PerformancePassportVerification(verification_status="failed", checked_items=0, issues=[f"passport JSON unreadable: {exc}"])
    checked += 1
    if payload.get("schema_version") != SCHEMA_VERSION:
        issues.append("unsupported schema_version")
    state = payload.get("claim_eligibility_state")
    checked += 1
    if state not in CLAIM_STATES:
        issues.append(f"unsupported claim_eligibility_state: {state!r}")
    result_origin = payload.get("result_origin")
    checked += 1
    if result_origin == "test_fixture" and state != "fixture_only":
        issues.append("test_fixture result_origin requires claim_eligibility_state fixture_only")
    comparison_path = Path(str(payload.get("comparison_result_path", "")))
    expected_comparison_digest = str(payload.get("comparison_result_sha256", ""))
    checked += 1
    if not comparison_path.exists():
        issues.append(f"comparison_result_path does not exist: {comparison_path}")
    else:
        actual = sha256_file(comparison_path)
        if actual.lower() != expected_comparison_digest.lower():
            issues.append("comparison_result_sha256 mismatch")
    manifest_text = str(payload.get("artifact_manifest_path", ""))
    expected_manifest_digest = str(payload.get("artifact_manifest_sha256", ""))
    if manifest_text:
        checked += 1
        manifest_path = Path(manifest_text)
        if not manifest_path.exists():
            issues.append(f"artifact_manifest_path does not exist: {manifest_path}")
        elif expected_manifest_digest and sha256_file(manifest_path).lower() != expected_manifest_digest.lower():
            issues.append("artifact_manifest_sha256 mismatch")
    claims = payload.get("benchmark_claims")
    checked += 1
    if not isinstance(claims, dict) or claims.get("schema_version") != SCHEMA_VERSION:
        issues.append("benchmark_claims must be a versioned object")
    return PerformancePassportVerification(
        verification_status="passed" if not issues else "failed",
        checked_items=checked,
        issues=issues,
    )
