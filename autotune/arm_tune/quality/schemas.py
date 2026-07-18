from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import to_plain


QUALITY_GATE_STATUSES = {
    "not_evaluated",
    "passed",
    "failed",
    "incomplete",
    "invalid_suite",
    "execution_failed",
}


@dataclass(frozen=True)
class QualityValidatorSpec:
    validator_type: str
    expected: Any = None
    field: str | None = None
    critical: bool = True
    required: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

@dataclass(frozen=True)
class QualityCase:
    case_id: str
    category: str
    prompt: str
    response_format: str
    required: bool
    expected: dict[str, Any]
    validators: list[QualityValidatorSpec]
    maximum_predicted_tokens: int
    generation_settings: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class QualitySuiteManifest:
    suite_id: str
    suite_digest: str
    case_count: int
    required_case_count: int
    source_path: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ValidatorResult:
    validator_type: str
    status: str
    expected: Any
    observed: Any
    message: str
    critical: bool = True
    required: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class QualityCaseResult:
    case_id: str
    category: str
    required: bool
    status: str
    request_status: str
    stream_parse_status: str
    validator_results: list[ValidatorResult]
    output_artifact: str = ""
    generated_output: str = ""
    repetition_index: int | None = None
    failure_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class QualityAggregateResult:
    total_cases: int
    required_cases: int
    passed_cases: int
    failed_cases: int
    critical_validator_failures: int
    json_validity_rate: float
    required_case_pass_rate: float
    evidence_citation_correctness: float
    quality_gate_status: str
    failure_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class CandidateQualityResult:
    candidate_id: str
    suite_manifest: QualitySuiteManifest
    aggregate: QualityAggregateResult
    cases: list[QualityCaseResult]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class QualityGateDecision:
    candidate_id: str
    quality_gate_status: str
    eligible: bool
    failure_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)
