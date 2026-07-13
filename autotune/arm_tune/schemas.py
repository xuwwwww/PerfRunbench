from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION


UNKNOWN = "unknown"


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
