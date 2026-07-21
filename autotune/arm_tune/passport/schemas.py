from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import to_plain, utc_now


@dataclass(frozen=True)
class PerformancePassport:
    run_id: str
    claim_eligibility_state: str
    claim_summary: str
    result_origin: str
    comparison_result_path: str
    comparison_result_sha256: str
    artifact_manifest_path: str = ""
    artifact_manifest_sha256: str = ""
    benchmark_claims: dict[str, Any] = field(default_factory=dict)
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    build_evidence: dict[str, Any] = field(default_factory=dict)
    host_system_evidence: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class PerformancePassportVerification:
    verification_status: str
    checked_items: int
    issues: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)
