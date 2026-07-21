from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import BenchmarkArtifactManifest, BenchmarkFileManifest, to_plain, utc_now
from autotune.arm_tune.search.statistics import MetricDistribution, NoiseBandDecision


@dataclass(frozen=True)
class BuildProfileDefinition:
    profile_id: str
    description: str
    cmake_definitions: dict[str, str]
    kleidiai_requested: bool = False
    cpu_target_policy: str = "upstream_default"
    backend_policy: str = "default_cpu"
    optimization_policy: str = "standard_release"
    portability_scope: str = "architecture_bound"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NativeBinaryScope:
    architecture: str
    portability_state: str
    source_commit: str = "unknown"
    compiler_identity: str = "unknown"
    effective_cpu_flags: list[str] = field(default_factory=list)
    required_cpu_features: list[str] = field(default_factory=list)
    cpu_model: str = "unknown"
    cpu_family: str = "unknown"
    binary_sha256: str = "unknown"
    build_target_fingerprint: str = "unknown"
    verified_target_fingerprints: list[str] = field(default_factory=list)
    startup_smoke_required: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NativeBinaryReuseDecision:
    compatibility_status: str
    launch_allowed: bool
    reasons: list[str] = field(default_factory=list)
    required_startup_smoke: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BuildSourceEvidence:
    source_repo: str
    source_status: str
    git_commit: str = "unknown"
    git_dirty_state: str = "unknown"
    git_remote: str = "unknown"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ToolchainEvidence:
    compiler_id: str = "unknown"
    compiler_version: str = "unknown"
    cmake_version: str = "unknown"
    linker_version: str = "unknown"
    generator: str = "unknown"
    build_type: str = "Release"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BinaryEvidence:
    profile_id: str
    binary_path: str
    manifest: BenchmarkFileManifest | None
    validation_status: str
    validation_message: str = ""
    executable_architecture: str = "unknown"
    executable_format: str = "unknown"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BuildProfileEvidence:
    run_id: str
    profile_id: str
    definition: BuildProfileDefinition
    source: BuildSourceEvidence
    toolchain: ToolchainEvidence
    requested_definitions: dict[str, str]
    effective_cmake_cache: dict[str, str] = field(default_factory=dict)
    compile_command_evidence: list[dict[str, Any]] = field(default_factory=list)
    effective_compiler_flags: list[str] = field(default_factory=list)
    effective_linker_flags: list[str] = field(default_factory=list)
    cpu_target_policy: str = "upstream_default"
    backend_policy: str = "default_cpu"
    optimization_policy: str = "standard_release"
    target_architecture: str = "unknown"
    binary: BinaryEvidence | None = None
    native_binary_scope: NativeBinaryScope | None = None
    linked_library_evidence: list[str] = field(default_factory=list)
    runtime_backend_evidence: dict[str, Any] = field(default_factory=dict)
    build_output: dict[str, Any] = field(default_factory=dict)
    build_duration_seconds: float | None = None
    build_status: str = "not_run"
    completeness: str = "incomplete"
    kleidiai_status: str = "unavailable"
    eligibility_state: str = "ineligible_provenance"
    result_origin: str = "unverified"
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BuildProfilePlan:
    run_id: str
    operation: str
    result_origin: str
    source: BuildSourceEvidence
    target_architecture: str
    toolchain: ToolchainEvidence
    profiles: list[BuildProfileDefinition]
    build_root: str = ""
    execute_build: bool = False
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BuildProfileRunResult:
    run_id: str
    operation: str
    result_origin: str
    profiles: list[BuildProfileEvidence]
    artifacts: BenchmarkArtifactManifest = field(default_factory=BenchmarkArtifactManifest)
    artifact_directory: str = ""
    artifact_validation: Any = None
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ProfileComparisonEntry:
    profile_id: str
    result_origin: str
    build_status: str
    kleidiai_status: str
    quality_gate_status: str
    eligibility_status: str
    eligible: bool
    ttft_ms: MetricDistribution
    end_to_end_latency_ms: MetricDistribution
    failure_rate: float
    metrics: dict[str, MetricDistribution] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ProfileComparisonRecommendation:
    status: str
    profile_id: str | None
    message: str
    noise_band_decision: NoiseBandDecision | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ProfileComparisonResult:
    run_id: str
    operation: str
    result_origin: str
    baseline_profile_id: str
    primary_metric: str
    secondary_metric: str
    recommendation: ProfileComparisonRecommendation
    profiles: list[ProfileComparisonEntry]
    rejected_profiles: list[dict[str, Any]] = field(default_factory=list)
    artifacts: BenchmarkArtifactManifest = field(default_factory=BenchmarkArtifactManifest)
    artifact_directory: str = ""
    artifact_validation: Any = None
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)
