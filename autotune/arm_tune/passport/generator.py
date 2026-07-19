from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.passport.markdown import passport_markdown
from autotune.arm_tune.passport.schemas import PerformancePassport
from autotune.arm_tune.passport.verification import verify_passport
from autotune.arm_tune.targets.protocol import atomic_write_json, read_json


def generate_passport(
    *,
    comparison_result_path: str | Path,
    run_id: str,
    output_dir: str | Path,
) -> tuple[PerformancePassport, Path, Path]:
    comparison_path = Path(comparison_result_path)
    comparison = read_json(comparison_path)
    manifest_path = comparison_path.parent / "artifact_manifest.json"
    state = _claim_state(comparison)
    passport = PerformancePassport(
        run_id=run_id,
        claim_eligibility_state=state,
        claim_summary=_claim_summary(comparison, state),
        result_origin=str(comparison.get("result_origin", "unverified")),
        comparison_result_path=str(comparison_path),
        comparison_result_sha256=sha256_file(comparison_path),
        artifact_manifest_path=str(manifest_path) if manifest_path.exists() else "",
        artifact_manifest_sha256=sha256_file(manifest_path) if manifest_path.exists() else "",
        benchmark_claims=_benchmark_claims(comparison),
        provenance_summary=_provenance_summary(comparison),
        limitations=_limitations(comparison, state),
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "performance_passport.json"
    markdown_path = root / "performance_passport.md"
    atomic_write_json(json_path, passport.to_dict())
    markdown_path.write_text(passport_markdown(passport), encoding="utf-8", newline="\n")
    verification = verify_passport(json_path)
    atomic_write_json(root / "performance_passport_verification.json", verification.to_dict())
    return replace(passport), json_path, markdown_path


def _claim_state(comparison: dict[str, Any]) -> str:
    if comparison.get("result_origin") == "test_fixture":
        return "fixture_only"
    recommendation = comparison.get("recommendation", {})
    if not isinstance(recommendation, dict):
        return "ineligible_provenance"
    status = recommendation.get("status")
    rejected = comparison.get("rejected_profiles", [])
    profiles = comparison.get("profiles", [])
    if any(isinstance(item, dict) and item.get("eligibility_status") == "rejected_quality" for item in rejected):
        return "ineligible_quality"
    if any(isinstance(item, dict) and item.get("eligibility_status") == "rejected_build_evidence" for item in rejected):
        return "ineligible_provenance"
    if status not in {"recommended", "inconclusive"}:
        return "ineligible_statistics"
    if not profiles:
        return "ineligible_statistics"
    return "eligible_preliminary"


def _claim_summary(comparison: dict[str, Any], state: str) -> str:
    recommendation = comparison.get("recommendation", {})
    profile = recommendation.get("profile_id") if isinstance(recommendation, dict) else None
    if state == "fixture_only":
        return "Fixture-only profile comparison; not a measured Arm performance claim."
    if state == "eligible_preliminary":
        return f"Preliminary quality-preserving Arm profile comparison; selected profile: {profile or 'none'}."
    return "Performance claim is not eligible because required evidence is incomplete or failed."


def _benchmark_claims(comparison: dict[str, Any]) -> dict[str, Any]:
    recommendation = comparison.get("recommendation", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": comparison.get("operation"),
        "baseline_profile_id": comparison.get("baseline_profile_id"),
        "recommended_profile_id": recommendation.get("profile_id") if isinstance(recommendation, dict) else None,
        "recommendation_status": recommendation.get("status") if isinstance(recommendation, dict) else None,
        "primary_metric": comparison.get("primary_metric"),
        "secondary_metric": comparison.get("secondary_metric"),
    }


def _provenance_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "result_origin": comparison.get("result_origin"),
        "profile_count": len(comparison.get("profiles", [])) if isinstance(comparison.get("profiles"), list) else 0,
        "rejected_profile_count": len(comparison.get("rejected_profiles", [])) if isinstance(comparison.get("rejected_profiles"), list) else 0,
    }


def _limitations(comparison: dict[str, Any], state: str) -> list[str]:
    limitations: list[str] = []
    if comparison.get("result_origin") == "test_fixture":
        limitations.append("Fixture metrics are not production Arm performance results.")
    if state != "eligible_preliminary":
        limitations.append("The passport must not be used as an optimization-result claim.")
    if comparison.get("schema_version") != SCHEMA_VERSION:
        limitations.append("Comparison schema version is unsupported.")
    return limitations
