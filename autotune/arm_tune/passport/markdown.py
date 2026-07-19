from __future__ import annotations

from autotune.arm_tune.passport.schemas import PerformancePassport


def passport_markdown(passport: PerformancePassport) -> str:
    claims = passport.benchmark_claims
    lines = [
        "# ArmTune Performance Passport",
        "",
        f"- schema_version: {passport.schema_version}",
        f"- run_id: {passport.run_id}",
        f"- claim_eligibility_state: {passport.claim_eligibility_state}",
        f"- result_origin: {passport.result_origin}",
        f"- claim_summary: {passport.claim_summary}",
        f"- baseline_profile_id: {claims.get('baseline_profile_id', 'unknown')}",
        f"- recommended_profile_id: {claims.get('recommended_profile_id', 'unknown')}",
        f"- recommendation_status: {claims.get('recommendation_status', 'unknown')}",
        f"- comparison_result_sha256: {passport.comparison_result_sha256}",
    ]
    if passport.artifact_manifest_sha256:
        lines.append(f"- artifact_manifest_sha256: {passport.artifact_manifest_sha256}")
    if passport.limitations:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in passport.limitations)
    lines.append("")
    return "\n".join(lines)
