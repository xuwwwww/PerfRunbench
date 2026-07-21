from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from autotune.recommendation.schemas import stable_digest


SCHEMA_VERSION = "1.0"
NOT_AVAILABLE = "not_available"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_after_minutes(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def versioned_digest(payload: dict[str, Any], *, ignore_keys: set[str] | None = None) -> str:
    ignored = ignore_keys or set()
    normalized = {key: value for key, value in payload.items() if key not in ignored}
    return stable_digest(normalized)


@dataclass(frozen=True)
class HostStateSnapshot:
    host_fingerprint: str
    identity: dict[str, Any]
    cpu_frequency: dict[str, Any]
    cpu_idle: dict[str, Any]
    scheduler_cgroup: dict[str, Any]
    interrupts_workqueues: dict[str, Any]
    memory: dict[str, Any]
    storage: dict[str, Any]
    network: dict[str, Any]
    security_recovery: dict[str, Any]
    snapshot_marker: str = NOT_AVAILABLE
    snapshot_digest: str = ""
    captured_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class HostCapabilityReport:
    snapshot_digest: str
    capabilities: dict[str, Any]
    profile_states: dict[str, str]
    generated_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ProfileChange:
    label: str
    interface: str
    path: str
    before_value: str
    requested_value: str
    effective_value: str = NOT_AVAILABLE
    requires_root: bool = False
    reversible: bool = True
    verification: str = "exact_value"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class HostSystemProfile:
    profile_id: str
    mechanism: str
    scope: str
    privilege_requirement: str
    reboot_requirement: bool
    required_capabilities: list[str]
    requested_changes: list[ProfileChange]
    prohibited_changes: list[str]
    verification_checks: list[str]
    rollback_plan: list[str]
    workload_protocols: list[str]
    expected_primary_metric: str
    possible_tradeoffs: list[str]
    eligibility_state: str
    state_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BootKernelProfile:
    profile_id: str
    mechanism: str
    kernel_flavor: str
    generated_command_line: str
    reboot_requirement: bool
    required_capabilities: list[str]
    verification_checks: list[str]
    rollback_plan: list[str]
    eligibility_state: str
    fallback_kernel: str = NOT_AVAILABLE
    one_time_boot: bool = True
    less_flexible_boot_isolation: bool = False
    state_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SystemTuningPlan:
    plan_id: str
    profile: HostSystemProfile
    snapshot_digest: str
    profile_state: str
    changes: list[ProfileChange]
    eligibility: str
    protocol_ids: list[str]
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class LeaseChangeRecord:
    label: str
    interface: str
    path: str
    before_value: str
    requested_value: str
    effective_value: str
    applied: bool
    error: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SystemTuningLease:
    lease_id: str
    profile_id: str
    snapshot_digest: str
    expires_at_utc: str
    changes: list[LeaseChangeRecord]
    active: bool = True
    watchdog_supported: bool = True
    watchdog_scope: str = "reversible_non_boot_settings_only"
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SystemTuningResult:
    profile_id: str
    apply_status: str
    lease: SystemTuningLease
    verification_status: str
    host_state_dirty: bool
    issues: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SystemRollbackReport:
    profile_id: str
    rollback_status: str
    restored_changes: list[LeaseChangeRecord]
    host_state_dirty: bool
    issues: list[str] = field(default_factory=list)
    completed_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BenchmarkProtocol:
    protocol_id: str
    primary_metric: str
    secondary_metrics: list[str]
    constraints: list[str]
    cannot_combine_with: list[str]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


class SystemProfileError(RuntimeError):
    pass
