from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION


UNKNOWN = "unknown"
RESULT_ORIGINS = {"measured", "test_fixture", "unverified"}
VALIDATION_STATUSES = {"passed", "failed"}
EXECUTION_STATUSES = {"completed", "failed", "timeout", "not_run"}
PARSE_STATUSES = {"parsed", "partially_parsed", "unsupported_format", "malformed_output", "no_metrics", "ambiguous_output"}
BENCHMARK_STATUSES = {"passed", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def require_schema_version(payload: dict[str, Any], *, expected: str = SCHEMA_VERSION) -> None:
    if payload.get("schema_version") != expected:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')!r}; expected {expected!r}")


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    return_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class TargetSpec:
    type: str
    host: str | None = None
    user: str | None = None
    port: int = 22
    identity_file: str | None = None
    remote_workdir: str = "~/armtune-runs"
    remote_python: str = "python3"
    remote_module: str = "autotune.cli"
    ssh_executable: str = "ssh"
    scp_executable: str = "scp"
    connect_timeout_seconds: int = 15
    command_timeout_seconds: int = 600
    keep_remote_artifacts: bool = False
    strict_host_key_checking: bool = True
    schema_version: str = SCHEMA_VERSION

    def sanitized_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        if self.identity_file:
            payload["identity_file"] = "<configured>"
        return payload

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ProductionValidation:
    valid: bool
    reason: str
    required_system: str = "Linux"
    required_machine: str = "aarch64"
    observed_system: str = UNKNOWN
    observed_machine: str = UNKNOWN
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class StatusCheck:
    status: str
    reason: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class DoctorStatuses:
    controller_status: StatusCheck
    local_target_status: StatusCheck
    configured_target_status: StatusCheck
    production_benchmark_ready: bool
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArchitectureInfo:
    system: str = UNKNOWN
    machine: str = UNKNOWN
    normalized_machine: str = UNKNOWN
    kernel_release: str = UNKNOWN
    is_linux: bool = False
    is_arm64: bool = False
    collected_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArmCpuIdentity:
    vendor: str = UNKNOWN
    model_name: str = UNKNOWN
    architecture: str = UNKNOWN
    implementer: str = UNKNOWN
    variant: str = UNKNOWN
    part: str = UNKNOWN
    revision: str = UNKNOWN
    source: str = UNKNOWN
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArmHostCapabilities:
    features: list[str] = field(default_factory=list)
    neon: bool | None = None
    dotprod: bool | None = None
    i8mm: bool | None = None
    bf16: bool | None = None
    sve: bool | None = None
    sve2: bool | None = None
    sme: bool | None = None
    sme2: bool | None = None
    evidence_source: str = UNKNOWN
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class CpuTopology:
    logical_cpus: int | None = None
    cores_per_socket: int | None = None
    sockets: int | None = None
    threads_per_core: int | None = None
    numa_nodes: int | None = None
    online_cpus: str = UNKNOWN
    caches: dict[str, str] = field(default_factory=dict)
    source: str = UNKNOWN
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class HostInfo:
    architecture: ArchitectureInfo
    cpu_identity: ArmCpuIdentity
    host_capabilities: ArmHostCapabilities
    topology: CpuTopology
    os_release: str = UNKNOWN
    cgroup_limits: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class WorkloadExecutionEvidence:
    status: str = "not_collected"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_artifacts: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkExecutableSpec:
    path: str
    expected_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkModelSpec:
    path: str
    expected_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LlamaBenchSettings:
    threads: int
    prompt_tokens: int
    generation_tokens: int
    repetitions: int
    timeout_seconds: float
    additional_supported_options: dict[str, str | int | float | bool] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkArtifactsSpec:
    preserve_raw_stdout: bool = True
    preserve_raw_stderr: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LlamaBenchJobSpec:
    protocol_version: str
    operation: str
    run_id: str
    target_run_directory: str
    executable: BenchmarkExecutableSpec
    model: BenchmarkModelSpec
    benchmark: LlamaBenchSettings
    artifacts: BenchmarkArtifactsSpec = field(default_factory=BenchmarkArtifactsSpec)
    target_type: str = "local"
    remote_module: str = "autotune.cli"
    result_origin: str = "measured"
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlamaBenchJobSpec":
        require_schema_version(payload)
        executable_payload = _required_object(payload, "executable")
        model_payload = _required_object(payload, "model")
        benchmark_payload = _required_object(payload, "benchmark")
        artifacts_payload = payload.get("artifacts", {})
        if not isinstance(artifacts_payload, dict):
            raise ValueError("artifacts must be an object")
        additional = benchmark_payload.get("additional_supported_options", {})
        if additional is None:
            additional = {}
        if not isinstance(additional, dict):
            raise ValueError("benchmark.additional_supported_options must be an object")
        result_origin = _optional_str(payload.get("result_origin", "unverified"), "result_origin")
        _ensure_enum(result_origin, RESULT_ORIGINS, "result_origin")
        return cls(
            protocol_version=_required_str(payload, "protocol_version"),
            operation=_required_str(payload, "operation"),
            run_id=_required_str(payload, "run_id"),
            target_run_directory=_required_str(payload, "target_run_directory"),
            executable=BenchmarkExecutableSpec(
                path=_required_str(executable_payload, "path"),
                expected_sha256=_optional_text(executable_payload.get("expected_sha256"), "expected_sha256"),
            ),
            model=BenchmarkModelSpec(
                path=_required_str(model_payload, "path"),
                expected_sha256=_optional_text(model_payload.get("expected_sha256"), "expected_sha256"),
            ),
            benchmark=LlamaBenchSettings(
                threads=_required_int(benchmark_payload, "threads"),
                prompt_tokens=_required_int(benchmark_payload, "prompt_tokens"),
                generation_tokens=_required_int(benchmark_payload, "generation_tokens"),
                repetitions=_required_int(benchmark_payload, "repetitions"),
                timeout_seconds=_required_float(benchmark_payload, "timeout_seconds"),
                additional_supported_options=dict(additional),
            ),
            artifacts=BenchmarkArtifactsSpec(
                preserve_raw_stdout=_optional_bool(artifacts_payload.get("preserve_raw_stdout", True), "preserve_raw_stdout"),
                preserve_raw_stderr=_optional_bool(artifacts_payload.get("preserve_raw_stderr", True), "preserve_raw_stderr"),
            ),
            target_type=_optional_str(payload.get("target_type", "local"), "target_type"),
            remote_module=_optional_str(payload.get("remote_module", "autotune.cli"), "remote_module"),
            result_origin=result_origin,
            created_at_utc=_optional_str(payload.get("created_at_utc", utc_now()), "created_at_utc"),
        )


@dataclass(frozen=True)
class BenchmarkFileManifest:
    path: str
    basename: str
    size_bytes: int
    sha256: str
    expected_sha256: str | None = None
    validation_status: str = "passed"
    validation_message: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LlamaBenchInspection:
    version_command: list[str]
    help_command: list[str]
    version_exit_code: int
    help_exit_code: int
    version_status: str
    help_status: str
    raw_version_stdout_path: str
    raw_version_stderr_path: str
    raw_help_stdout_path: str
    raw_help_stderr_path: str
    supported_flags: list[str] = field(default_factory=list)
    source_commit: str = UNKNOWN
    fixture_marker_detected: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LlamaBenchInvocation:
    command: list[str]
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    exit_code: int | None
    timed_out: bool
    raw_stdout_path: str
    raw_stderr_path: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ParsedMetric:
    name: str
    value: float
    unit: str
    source: str
    parser_strategy: str
    raw_source_line: str
    raw_line_reference: int | None = None
    validation_status: str = "passed"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LlamaBenchParseResult:
    parse_status: str
    metrics: list[ParsedMetric] = field(default_factory=list)
    parser_strategy: str = "none"
    unparsed_lines: list[str] = field(default_factory=list)
    message: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkTrialResult:
    trial_index: int
    validation_status: str
    execution_status: str
    parse_status: str
    benchmark_status: str
    message: str
    invocation: LlamaBenchInvocation | None = None
    parse_result: LlamaBenchParseResult | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkArtifactManifest:
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkProvenance:
    result_origin: str
    requested_origin: str
    decision_reason: str
    evidence: list[str] = field(default_factory=list)
    host_production_ready: bool = False
    fake_executable_marker: bool = False
    test_only_host_override: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArtifactValidationIssue:
    label: str
    path: str
    issue: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArtifactValidationResult:
    validation_status: str = "not_checked"
    checked_artifacts: int = 0
    issues: list[ArtifactValidationIssue] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArmBenchmarkResult:
    run_id: str
    operation: str
    target_type: str
    result_origin: str
    validation_status: str
    execution_status: str
    parse_status: str
    benchmark_status: str
    host: HostInfo
    executable: BenchmarkFileManifest | None = None
    model: BenchmarkFileManifest | None = None
    inspection: LlamaBenchInspection | None = None
    trial: BenchmarkTrialResult | None = None
    artifacts: BenchmarkArtifactManifest = field(default_factory=BenchmarkArtifactManifest)
    provenance: BenchmarkProvenance | None = None
    artifact_validation: ArtifactValidationResult = field(default_factory=ArtifactValidationResult)
    artifact_directory: str = ""
    message: str = ""
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArmBenchmarkResult":
        require_schema_version(payload)
        result_origin = _required_str(payload, "result_origin")
        validation_status = _required_str(payload, "validation_status")
        execution_status = _required_str(payload, "execution_status")
        parse_status = _required_str(payload, "parse_status")
        benchmark_status = _required_str(payload, "benchmark_status")
        _ensure_enum(result_origin, RESULT_ORIGINS, "result_origin")
        _ensure_enum(validation_status, VALIDATION_STATUSES, "validation_status")
        _ensure_enum(execution_status, EXECUTION_STATUSES, "execution_status")
        _ensure_enum(parse_status, PARSE_STATUSES, "parse_status")
        _ensure_enum(benchmark_status, BENCHMARK_STATUSES, "benchmark_status")
        if benchmark_status == "passed" and (
            validation_status != "passed" or execution_status != "completed" or parse_status != "parsed"
        ):
            raise ValueError("benchmark_status passed requires validation passed, execution completed, and parse parsed")
        host_payload = payload["host"]
        host = HostInfo(
            architecture=ArchitectureInfo(**_without_schema(host_payload["architecture"])),
            cpu_identity=ArmCpuIdentity(**_without_schema(host_payload["cpu_identity"])),
            host_capabilities=ArmHostCapabilities(**_without_schema(host_payload["host_capabilities"])),
            topology=CpuTopology(**_without_schema(host_payload["topology"])),
            os_release=host_payload.get("os_release", UNKNOWN),
            cgroup_limits=host_payload.get("cgroup_limits", {}),
        )
        return cls(
            run_id=_required_str(payload, "run_id"),
            operation=_required_str(payload, "operation"),
            target_type=_required_str(payload, "target_type"),
            result_origin=result_origin,
            validation_status=validation_status,
            execution_status=execution_status,
            parse_status=parse_status,
            benchmark_status=benchmark_status,
            host=host,
            executable=_file_manifest_from_payload(payload.get("executable")),
            model=_file_manifest_from_payload(payload.get("model")),
            inspection=_inspection_from_payload(payload.get("inspection")),
            trial=_trial_from_payload(payload.get("trial")),
            artifacts=BenchmarkArtifactManifest(artifacts=payload.get("artifacts", {}).get("artifacts", {})),
            provenance=_provenance_from_payload(payload.get("provenance")),
            artifact_validation=_artifact_validation_from_payload(payload.get("artifact_validation")),
            artifact_directory=str(payload.get("artifact_directory", "")),
            message=str(payload.get("message", "")),
            created_at_utc=str(payload.get("created_at_utc", utc_now())),
        )


@dataclass(frozen=True)
class RemotePrerequisiteResult:
    python_executable: str
    python_available: bool
    module: str
    module_importable: bool
    worker_protocol_compatible: bool
    remote_workdir_writable: bool
    linux: bool
    arm64: bool
    host: HostInfo
    production_validation: ProductionValidation
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemotePrerequisiteResult":
        require_schema_version(payload)
        host_payload = payload["host"]
        host = HostInfo(
            architecture=ArchitectureInfo(**_without_schema(host_payload["architecture"])),
            cpu_identity=ArmCpuIdentity(**_without_schema(host_payload["cpu_identity"])),
            host_capabilities=ArmHostCapabilities(**_without_schema(host_payload["host_capabilities"])),
            topology=CpuTopology(**_without_schema(host_payload["topology"])),
            os_release=host_payload.get("os_release", UNKNOWN),
            cgroup_limits=host_payload.get("cgroup_limits", {}),
        )
        return cls(
            python_executable=str(payload["python_executable"]),
            python_available=bool(payload["python_available"]),
            module=str(payload["module"]),
            module_importable=bool(payload["module_importable"]),
            worker_protocol_compatible=bool(payload["worker_protocol_compatible"]),
            remote_workdir_writable=bool(payload["remote_workdir_writable"]),
            linux=bool(payload["linux"]),
            arm64=bool(payload["arm64"]),
            host=host,
            production_validation=ProductionValidation(**_without_schema(payload["production_validation"])),
        )


@dataclass(frozen=True)
class HostInspectionResult:
    target_type: str
    host: HostInfo
    production_validation: ProductionValidation
    workload_execution_evidence: WorkloadExecutionEvidence = field(default_factory=WorkloadExecutionEvidence)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HostInspectionResult":
        require_schema_version(payload)
        host_payload = payload["host"]
        validation_payload = payload["production_validation"]
        workload_payload = payload.get("workload_execution_evidence", {})
        host = HostInfo(
            architecture=ArchitectureInfo(**_without_schema(host_payload["architecture"])),
            cpu_identity=ArmCpuIdentity(**_without_schema(host_payload["cpu_identity"])),
            host_capabilities=ArmHostCapabilities(**_without_schema(host_payload["host_capabilities"])),
            topology=CpuTopology(**_without_schema(host_payload["topology"])),
            os_release=host_payload.get("os_release", UNKNOWN),
            cgroup_limits=host_payload.get("cgroup_limits", {}),
        )
        return cls(
            target_type=str(payload["target_type"]),
            host=host,
            production_validation=ProductionValidation(**_without_schema(validation_payload)),
            workload_execution_evidence=WorkloadExecutionEvidence(**_without_schema(workload_payload)),
        )


@dataclass(frozen=True)
class WorkerJobSpec:
    operation: str
    target_type: str = "local"
    remote_module: str = "autotune.cli"
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class WorkerResult:
    operation: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def _without_schema(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "schema_version"}


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string when provided")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _ensure_enum(value: str, allowed: set[str], label: str) -> None:
    if value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}")


def _file_manifest_from_payload(payload: dict[str, Any] | None) -> BenchmarkFileManifest | None:
    if payload is None:
        return None
    return BenchmarkFileManifest(**_without_schema(payload))


def _inspection_from_payload(payload: dict[str, Any] | None) -> LlamaBenchInspection | None:
    if payload is None:
        return None
    return LlamaBenchInspection(**_without_schema(payload))


def _parse_result_from_payload(payload: dict[str, Any] | None) -> LlamaBenchParseResult | None:
    if payload is None:
        return None
    return LlamaBenchParseResult(
        parse_status=str(payload["parse_status"]),
        metrics=[ParsedMetric(**_without_schema(metric)) for metric in payload.get("metrics", [])],
        parser_strategy=str(payload.get("parser_strategy", "none")),
        unparsed_lines=list(payload.get("unparsed_lines", [])),
        message=str(payload.get("message", "")),
    )


def _invocation_from_payload(payload: dict[str, Any] | None) -> LlamaBenchInvocation | None:
    if payload is None:
        return None
    return LlamaBenchInvocation(**_without_schema(payload))


def _trial_from_payload(payload: dict[str, Any] | None) -> BenchmarkTrialResult | None:
    if payload is None:
        return None
    return BenchmarkTrialResult(
        trial_index=int(payload["trial_index"]),
        validation_status=str(payload["validation_status"]),
        execution_status=str(payload["execution_status"]),
        parse_status=str(payload["parse_status"]),
        benchmark_status=str(payload["benchmark_status"]),
        message=str(payload.get("message", "")),
        invocation=_invocation_from_payload(payload.get("invocation")),
        parse_result=_parse_result_from_payload(payload.get("parse_result")),
    )


def _provenance_from_payload(payload: dict[str, Any] | None) -> BenchmarkProvenance | None:
    if payload is None:
        return None
    return BenchmarkProvenance(**_without_schema(payload))


def _artifact_validation_from_payload(payload: dict[str, Any] | None) -> ArtifactValidationResult:
    if payload is None:
        return ArtifactValidationResult()
    return ArtifactValidationResult(
        validation_status=str(payload.get("validation_status", "not_checked")),
        checked_artifacts=int(payload.get("checked_artifacts", 0)),
        issues=[ArtifactValidationIssue(**_without_schema(issue)) for issue in payload.get("issues", [])],
    )
