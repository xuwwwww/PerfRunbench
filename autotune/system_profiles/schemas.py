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
class KernelConfigCapability:
    source_path: str
    digest: str
    capabilities: dict[str, str]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelCandidate:
    candidate_id: str
    release: str
    flavor: str
    package_name: str
    package_version: str
    architecture: str
    source_state: str
    vmlinuz_path: str
    initrd_path: str
    modules_path: str
    base_page_size: str
    config: KernelConfigCapability
    qualification_state: str
    blockers: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BootRecoveryCapability:
    snapshot_marker_state: str
    recovery_snapshot_state: str
    serial_console_state: str
    out_of_band_recovery_state: str
    fallback_kernel_state: str
    fallback_kernel: str
    root_volume_identity_state: str
    volume_recovery_state: str
    boot_menu_access_state: str
    manual_recovery_instructions_state: str
    one_time_boot_state: str
    ssh_post_boot_check_state: str
    rollback_command_state: str
    user_authorization_state: str
    maximum_health_check_timeout_seconds: int = 600
    maximum_reboot_count: int = 1
    blockers: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BootInventory:
    current_kernel_package: str
    current_kernel_release: str
    current_kernel_flavor: str
    current_base_page_size: str
    current_boot_id: str
    current_command_line: str
    current_command_line_digest: str
    current_default_boot_entry: str
    known_good_fallback_entry: str
    architecture: str
    logical_cpus: int | str
    distribution: dict[str, str]
    bootloader: str
    bootloader_evidence: dict[str, Any]
    kernel_inventory: list[dict[str, Any]]
    boot_files: dict[str, Any]
    filesystem_inventory: dict[str, Any]
    grub_environment: dict[str, Any]
    efi_state: dict[str, Any]
    initramfs_inventory: dict[str, Any]
    secure_boot_state: str
    lockdown_state: str
    current_kernel_config: KernelConfigCapability
    current_kernel: KernelCandidate
    fallback_kernel: KernelCandidate
    candidate_64k_kernel: KernelCandidate
    recovery: BootRecoveryCapability
    blocker_matrix: dict[str, dict[str, Any]]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class PostBootHealthPolicy:
    required_checks: list[str]
    timeout_seconds: int
    failure_action: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BootRollbackPlan:
    rollback_commands: list[str]
    fallback_mechanism: str
    maximum_reboot_count: int
    original_state_verification: list[str]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class OneTimeBootPlan:
    profile_id: str
    description: str
    profile_order: int
    current_kernel_package: str
    current_kernel_release: str
    current_kernel_flavor: str
    current_base_page_size: str
    current_boot_id: str
    current_command_line_digest: str
    current_default_boot_entry: str
    known_good_fallback_entry: str
    candidate_kernel_package: str
    candidate_kernel_release: str
    candidate_vmlinuz_path: str
    candidate_initrd_path: str
    candidate_modules_path: str
    candidate_base_page_size: str
    candidate_kernel_config_digest: str
    exact_candidate_command_line: str
    one_time_boot_mechanism: str
    fallback_mechanism: str
    expected_post_boot_identity: dict[str, Any]
    health_policy: PostBootHealthPolicy
    ssh_recovery_policy: str
    serial_console_recovery_policy: str
    rollback_plan: BootRollbackPlan
    maximum_reboot_count: int
    authorization_state: str
    eligibility_state: str
    blockers: list[str] = field(default_factory=list)
    advanced_less_flexible: bool = False
    temporary_entry_plan: dict[str, Any] = field(default_factory=dict)
    comparison_identities: list[dict[str, Any]] = field(default_factory=list)
    causal_baseline_identity: str = NOT_AVAILABLE
    capacity_baseline_identity: str = NOT_AVAILABLE
    experimental_identity: str = NOT_AVAILABLE
    kernel_or_boot_state_modified: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class BootExperimentResult:
    profile_id: str
    status: str
    plan_digest: str
    boot_attempted: bool
    reboot_performed: bool
    kernel_or_boot_state_modified: bool
    health_status: str
    rollback_status: str
    blockers: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageInventory:
    distribution_release: str
    architecture: str
    configured_repositories: list[dict[str, Any]]
    installed_kernel_packages: list[dict[str, Any]]
    package_manager_state: str
    boot_free_bytes: int | str
    root_free_bytes: int | str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageCandidate:
    profile_id: str
    meta_package: str
    image_package: str
    modules_packages: list[str]
    header_packages: list[str]
    candidate_version: str
    package_architecture: str
    package_origins: list[str]
    trust_source: str
    download_size_bytes: int | str
    installed_size_bytes: int | str
    expected_release: str
    expected_page_size: str
    qualification_state: str
    blockers: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageRollbackPlan:
    remove_packages: list[str]
    preserve_packages: list[str]
    forbidden_commands: list[str]
    dpkg_recovery_commands: list[str]
    final_original_state_verification: list[str]
    package_cache_cleanup_policy: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageTransactionPlan:
    profile_id: str
    package_inventory: KernelPackageInventory
    candidate: KernelPackageCandidate
    apt_simulation: dict[str, Any]
    dependency_changes: dict[str, list[str]]
    packages_newly_installed: list[str]
    packages_upgraded: list[str]
    packages_removed: list[str]
    packages_held: list[str]
    expected_artifacts: dict[str, str]
    expected_grub_entries: list[str]
    expected_page_size: str
    real_install_command: list[str]
    rollback_plan: KernelPackageRollbackPlan
    eligibility_state: str
    blockers: list[str] = field(default_factory=list)
    kernel_or_boot_state_modified: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageTransactionResult:
    plan_digest: str
    status: str
    install_attempted: bool
    kernel_or_boot_state_modified: bool
    installed_packages: list[str]
    removed_packages: list[str]
    issues: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class GrubDefaultResolution:
    status: str
    configuration_text_digest: str
    grub_environment_digest: str
    saved_entry: str
    next_entry: str
    generated_menu_order: list[str]
    resolved_default_entry_id: str
    resolved_default_kernel: str
    configuration_text_unchanged: bool
    resolved_default_boot_target_unchanged: bool
    safety_pin_required: bool
    safety_pin_plan: dict[str, Any]
    blockers: list[str] = field(default_factory=list)
    kernel_or_boot_state_modified: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelArtifactInventory:
    kernel_release: str
    vmlinuz_path: str
    initrd_path: str
    config_path: str
    system_map_path: str
    modules_path: str
    package_ownership: dict[str, str]
    page_size_expectation: str
    qualification_state: str
    blockers: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class KernelPackageCleanupResult:
    status: str
    removed_packages: list[str]
    preserved_packages: list[str]
    removed_artifacts: list[str]
    residual_artifacts: list[str]
    package_inventory_restored: bool
    grub_state_restored: bool
    issues: list[str] = field(default_factory=list)
    kernel_or_boot_state_modified: bool = False
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
