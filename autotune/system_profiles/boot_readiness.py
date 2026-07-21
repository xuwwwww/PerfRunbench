from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import file_sha256, stable_digest
from autotune.system_profiles.schemas import (
    BootExperimentResult,
    BootInventory,
    BootRecoveryCapability,
    BootRollbackPlan,
    KernelCandidate,
    KernelConfigCapability,
    NOT_AVAILABLE,
    OneTimeBootPlan,
    PostBootHealthPolicy,
    SystemProfileError,
    versioned_digest,
)
from autotune.system_profiles.snapshot import RootReader, collect_host_state_snapshot


PHASE21C0_BOOT_PROFILES = [
    "original-4k-default",
    "4k-nohz-rcu-irq",
    "4k-managed-irq-isolation",
    "4k-domain-isolation",
    "64k-default",
    "64k-nohz-rcu-irq",
    "64k-managed-irq-isolation",
]

CONFIG_KEYS = [
    "CONFIG_ARM64_4K_PAGES",
    "CONFIG_ARM64_64K_PAGES",
    "CONFIG_NO_HZ_FULL",
    "CONFIG_RCU_NOCB_CPU",
    "CONFIG_CPU_ISOLATION",
    "CONFIG_CGROUPS",
    "CONFIG_CPUSETS",
    "CONFIG_IRQ_TIME_ACCOUNTING",
    "CONFIG_SCHEDSTATS",
    "CONFIG_HUGETLBFS",
    "CONFIG_TRANSPARENT_HUGEPAGE",
    "CONFIG_PERF_EVENTS",
]

BLOCKER_CODES = [
    "missing_recovery_snapshot",
    "missing_snapshot_marker",
    "serial_console_not_verified",
    "no_out_of_band_recovery",
    "fallback_kernel_missing",
    "fallback_kernel_unverified",
    "bootloader_unknown",
    "one_time_boot_unavailable",
    "kernel_package_candidate_unknown",
    "kernel_config_support_unknown",
    "insufficient_boot_partition_space",
    "secure_boot_or_lockdown_unknown",
    "initramfs_generation_unknown",
    "ssh_post_boot_check_missing",
    "rollback_command_missing",
    "user_authorization_missing",
    "other",
]

HEALTH_CHECKS = [
    "management_channel_available",
    "expected_kernel_release",
    "expected_base_page_size",
    "expected_command_line_digest",
    "expected_cpu_count",
    "expected_architecture",
    "expected_root_filesystem",
    "expected_network_state",
    "expected_model_and_binary_files_available",
    "no_emergency_mode",
    "no_failed_critical_mounts",
    "no_kernel_panic_evidence",
    "no_unexpected_oom",
    "no_dirty_host_state",
    "benchmark_preflight_passes",
]


def collect_boot_inventory(root: str | Path | None = None) -> BootInventory:
    reader = RootReader(root)
    snapshot = collect_host_state_snapshot(root)
    identity = snapshot.identity
    current_release = str(identity.get("kernel_release", NOT_AVAILABLE))
    current_flavor = str(identity.get("kernel_flavor", NOT_AVAILABLE))
    current_page_size = str(identity.get("base_page_size", NOT_AVAILABLE))
    logical_cpus = identity.get("logical_cpus", NOT_AVAILABLE)
    cmdline = str(identity.get("kernel_command_line", NOT_AVAILABLE))
    cmdline_digest = stable_digest({"kernel_command_line": cmdline})
    boot_id = str(identity.get("boot_id", NOT_AVAILABLE))
    arch = str(identity.get("architecture", NOT_AVAILABLE))
    distro = identity.get("distribution_identity", {"state": NOT_AVAILABLE})
    if not isinstance(distro, dict):
        distro = {"state": NOT_AVAILABLE}

    current_config = read_kernel_config(reader, current_release)
    kernel_inventory = _kernel_package_inventory(reader)
    boot_files = _boot_files(reader)
    bootloader, bootloader_evidence = _detect_bootloader(reader)
    filesystem = _filesystem_inventory(reader)
    grub_environment = _grub_environment(reader)
    efi_state = _efi_state(reader)
    initramfs = _initramfs_inventory(reader)
    default_entry = _default_boot_entry(reader, grub_environment, bootloader_evidence, current_release)
    fallback_release = _fallback_release(boot_files, current_release)
    fallback_config = read_kernel_config(reader, fallback_release)
    recovery = _recovery_capability(reader, snapshot, bootloader, fallback_release, boot_files, initramfs)
    current_kernel = _current_kernel_candidate(reader, current_release, current_flavor, arch, current_page_size, current_config, kernel_inventory, boot_files, initramfs)
    fallback_kernel = _fallback_kernel_candidate(reader, fallback_release, arch, fallback_config, kernel_inventory, boot_files, initramfs, current_release)
    candidate_64k = _candidate_64k_kernel(reader, arch, distro)
    blockers = _blocker_matrix(
        reader=reader,
        bootloader=bootloader,
        recovery=recovery,
        current=current_kernel,
        fallback=fallback_kernel,
        candidate_64k=candidate_64k,
        filesystem=filesystem,
        secure_boot_state=_secure_boot_state(reader),
        lockdown_state=str(snapshot.security_recovery.get("lockdown_state", NOT_AVAILABLE)),
    )
    return BootInventory(
        current_kernel_package=current_kernel.package_name,
        current_kernel_release=current_release,
        current_kernel_flavor=current_flavor,
        current_base_page_size=current_page_size,
        current_boot_id=boot_id,
        current_command_line=cmdline,
        current_command_line_digest=cmdline_digest,
        current_default_boot_entry=default_entry,
        known_good_fallback_entry=fallback_release,
        architecture=arch,
        logical_cpus=logical_cpus if isinstance(logical_cpus, int) else str(logical_cpus),
        distribution={str(k): str(v) for k, v in distro.items()},
        bootloader=bootloader,
        bootloader_evidence=bootloader_evidence,
        kernel_inventory=kernel_inventory,
        boot_files=boot_files,
        filesystem_inventory=filesystem,
        grub_environment=grub_environment,
        efi_state=efi_state,
        initramfs_inventory=initramfs,
        secure_boot_state=_secure_boot_state(reader),
        lockdown_state=str(snapshot.security_recovery.get("lockdown_state", NOT_AVAILABLE)),
        current_kernel_config=current_config,
        current_kernel=current_kernel,
        fallback_kernel=fallback_kernel,
        candidate_64k_kernel=candidate_64k,
        recovery=recovery,
        blocker_matrix=blockers,
    )


def audit_boot_readiness(inventory: BootInventory) -> dict[str, Any]:
    blockers = [code for code, item in inventory.blocker_matrix.items() if item.get("state") == "blocked"]
    return {
        "schema_version": "1.0",
        "status": "blocked" if blockers else "ready_for_explicit_remote_readonly_audit",
        "phase": "21C0",
        "ready_for_phase21c": False,
        "kernel_or_boot_state_modified": False,
        "aws_accessed": False,
        "blockers": blockers,
        "blocker_matrix": inventory.blocker_matrix,
    }


def boot_inventory_from_dict(raw: dict[str, Any]) -> BootInventory:
    current_config = _kernel_config_from_dict(raw.get("current_kernel_config", {}))
    current_kernel = _kernel_candidate_from_dict(raw.get("current_kernel", {}))
    fallback_kernel = _kernel_candidate_from_dict(raw.get("fallback_kernel", {}))
    candidate_64k = _kernel_candidate_from_dict(raw.get("candidate_64k_kernel", {}))
    recovery = BootRecoveryCapability(**raw.get("recovery", {}))
    return BootInventory(
        current_kernel_package=str(raw.get("current_kernel_package", NOT_AVAILABLE)),
        current_kernel_release=str(raw.get("current_kernel_release", NOT_AVAILABLE)),
        current_kernel_flavor=str(raw.get("current_kernel_flavor", NOT_AVAILABLE)),
        current_base_page_size=str(raw.get("current_base_page_size", NOT_AVAILABLE)),
        current_boot_id=str(raw.get("current_boot_id", NOT_AVAILABLE)),
        current_command_line=str(raw.get("current_command_line", NOT_AVAILABLE)),
        current_command_line_digest=str(raw.get("current_command_line_digest", NOT_AVAILABLE)),
        current_default_boot_entry=str(raw.get("current_default_boot_entry", NOT_AVAILABLE)),
        known_good_fallback_entry=str(raw.get("known_good_fallback_entry", NOT_AVAILABLE)),
        architecture=str(raw.get("architecture", NOT_AVAILABLE)),
        logical_cpus=raw.get("logical_cpus", NOT_AVAILABLE),
        distribution=dict(raw.get("distribution", {})),
        bootloader=str(raw.get("bootloader", NOT_AVAILABLE)),
        bootloader_evidence=dict(raw.get("bootloader_evidence", {})),
        kernel_inventory=list(raw.get("kernel_inventory", [])),
        boot_files=dict(raw.get("boot_files", {})),
        filesystem_inventory=dict(raw.get("filesystem_inventory", {})),
        grub_environment=dict(raw.get("grub_environment", {})),
        efi_state=dict(raw.get("efi_state", {})),
        initramfs_inventory=dict(raw.get("initramfs_inventory", {})),
        secure_boot_state=str(raw.get("secure_boot_state", NOT_AVAILABLE)),
        lockdown_state=str(raw.get("lockdown_state", NOT_AVAILABLE)),
        current_kernel_config=current_config,
        current_kernel=current_kernel,
        fallback_kernel=fallback_kernel,
        candidate_64k_kernel=candidate_64k,
        recovery=recovery,
        blocker_matrix=dict(raw.get("blocker_matrix", {})),
        schema_version=str(raw.get("schema_version", "1.0")),
    )


def _kernel_config_from_dict(raw: dict[str, Any]) -> KernelConfigCapability:
    return KernelConfigCapability(
        source_path=str(raw.get("source_path", NOT_AVAILABLE)),
        digest=str(raw.get("digest", NOT_AVAILABLE)),
        capabilities={str(key): str(value) for key, value in dict(raw.get("capabilities", {})).items()},
        schema_version=str(raw.get("schema_version", "1.0")),
    )


def _kernel_candidate_from_dict(raw: dict[str, Any]) -> KernelCandidate:
    config = _kernel_config_from_dict(raw.get("config", {}))
    return KernelCandidate(
        candidate_id=str(raw.get("candidate_id", NOT_AVAILABLE)),
        release=str(raw.get("release", NOT_AVAILABLE)),
        flavor=str(raw.get("flavor", NOT_AVAILABLE)),
        package_name=str(raw.get("package_name", NOT_AVAILABLE)),
        package_version=str(raw.get("package_version", NOT_AVAILABLE)),
        architecture=str(raw.get("architecture", NOT_AVAILABLE)),
        source_state=str(raw.get("source_state", NOT_AVAILABLE)),
        vmlinuz_path=str(raw.get("vmlinuz_path", NOT_AVAILABLE)),
        initrd_path=str(raw.get("initrd_path", NOT_AVAILABLE)),
        modules_path=str(raw.get("modules_path", NOT_AVAILABLE)),
        base_page_size=str(raw.get("base_page_size", NOT_AVAILABLE)),
        config=config,
        qualification_state=str(raw.get("qualification_state", "blocked")),
        blockers=[str(item) for item in raw.get("blockers", [])],
        schema_version=str(raw.get("schema_version", "1.0")),
    )


def build_one_time_boot_plan(profile_id: str, inventory: BootInventory, *, housekeeping_cpus: str = "0-1", workload_cpus: str = "2-7") -> OneTimeBootPlan:
    if profile_id not in PHASE21C0_BOOT_PROFILES:
        raise SystemProfileError(f"unknown boot profile: {profile_id}")
    _validate_cpu_layout(housekeeping_cpus, workload_cpus)
    order = PHASE21C0_BOOT_PROFILES.index(profile_id)
    candidate = _candidate_for_profile(profile_id, inventory)
    additions = _profile_args(profile_id, inventory, housekeeping_cpus, workload_cpus)
    command_line = _append_non_duplicate_args(inventory.current_command_line, additions)
    expected = {
        "kernel_release": candidate.release,
        "base_page_size": candidate.base_page_size,
        "architecture": inventory.architecture,
        "command_line_digest": stable_digest({"kernel_command_line": command_line}),
        "cpu_count": _logical_cpu_count(inventory),
        "root_filesystem": inventory.filesystem_inventory.get("root_filesystem_type", NOT_AVAILABLE),
    }
    plan_blockers = _plan_blockers(profile_id, inventory, candidate, additions)
    eligibility = "planning_ready" if not plan_blockers else "blocked"
    if profile_id == "original-4k-default":
        eligibility = "baseline_known_good" if not plan_blockers else "blocked"
    health = PostBootHealthPolicy(required_checks=HEALTH_CHECKS, timeout_seconds=inventory.recovery.maximum_health_check_timeout_seconds, failure_action="execute_documented_recovery_path")
    rollback = BootRollbackPlan(
        rollback_commands=_rollback_commands(inventory),
        fallback_mechanism=_fallback_mechanism(inventory),
        maximum_reboot_count=inventory.recovery.maximum_reboot_count,
        original_state_verification=["kernel_release", "base_page_size", "command_line_digest", "default_boot_entry", "host_state_snapshot"],
    )
    return OneTimeBootPlan(
        profile_id=profile_id,
        description=_profile_description(profile_id),
        profile_order=order,
        current_kernel_package=inventory.current_kernel_package,
        current_kernel_release=inventory.current_kernel_release,
        current_kernel_flavor=inventory.current_kernel_flavor,
        current_base_page_size=inventory.current_base_page_size,
        current_boot_id=inventory.current_boot_id,
        current_command_line_digest=inventory.current_command_line_digest,
        current_default_boot_entry=inventory.current_default_boot_entry,
        known_good_fallback_entry=inventory.known_good_fallback_entry,
        candidate_kernel_package=candidate.package_name,
        candidate_kernel_release=candidate.release,
        candidate_vmlinuz_path=candidate.vmlinuz_path,
        candidate_initrd_path=candidate.initrd_path,
        candidate_modules_path=candidate.modules_path,
        candidate_base_page_size=candidate.base_page_size,
        candidate_kernel_config_digest=candidate.config.digest,
        exact_candidate_command_line=command_line,
        one_time_boot_mechanism=_one_time_boot_mechanism(inventory),
        fallback_mechanism=rollback.fallback_mechanism,
        expected_post_boot_identity=expected,
        health_policy=health,
        ssh_recovery_policy=inventory.recovery.ssh_post_boot_check_state,
        serial_console_recovery_policy=inventory.recovery.serial_console_state,
        rollback_plan=rollback,
        maximum_reboot_count=inventory.recovery.maximum_reboot_count,
        authorization_state=inventory.recovery.user_authorization_state,
        eligibility_state=eligibility,
        blockers=plan_blockers,
        advanced_less_flexible=profile_id == "4k-domain-isolation",
    )


def verify_one_time_boot_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if raw_plan.get("schema_version") != "1.0":
        issues.append("missing_or_unsupported_schema_version")
    if raw_plan.get("kernel_or_boot_state_modified") is True:
        issues.append("real_boot_state_modification_is_forbidden_in_phase21c0")
    profile_id = raw_plan.get("profile_id")
    if profile_id not in PHASE21C0_BOOT_PROFILES:
        issues.append("unknown_profile_id")
    if raw_plan.get("one_time_boot_mechanism") in {"", NOT_AVAILABLE, "unknown"}:
        issues.append("one_time_boot_unavailable")
    if raw_plan.get("current_default_boot_entry") in {"", NOT_AVAILABLE, None}:
        issues.append("current_default_boot_entry_missing")
    if raw_plan.get("known_good_fallback_entry") in {"", NOT_AVAILABLE, None}:
        issues.append("fallback_kernel_missing")
    if raw_plan.get("maximum_reboot_count", 0) > 1:
        issues.append("maximum_reboot_count_exceeds_phase21c0_limit")
    if raw_plan.get("authorization_state") != "explicit_user_authorization_required":
        issues.append("authorization_boundary_missing")
    rollback = raw_plan.get("rollback_plan", {})
    if not isinstance(rollback, dict) or not rollback.get("rollback_commands"):
        issues.append("rollback_command_missing")
    return {
        "schema_version": "1.0",
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "plan_digest": versioned_digest(raw_plan) if isinstance(raw_plan, dict) else NOT_AVAILABLE,
        "kernel_or_boot_state_modified": False,
    }


def build_recovery_plan(inventory: BootInventory) -> dict[str, Any]:
    blockers = [code for code in ["missing_recovery_snapshot", "serial_console_not_verified", "no_out_of_band_recovery", "fallback_kernel_missing", "rollback_command_missing"] if inventory.blocker_matrix[code]["state"] == "blocked"]
    return {
        "schema_version": "1.0",
        "status": "complete_placeholder_plan" if not blockers else "blocked",
        "blockers": blockers,
        "snapshot_verification": inventory.recovery.recovery_snapshot_state,
        "serial_console_verification": inventory.recovery.serial_console_state,
        "fallback_kernel": inventory.recovery.fallback_kernel,
        "volume_recovery": inventory.recovery.volume_recovery_state,
        "one_time_boot": inventory.recovery.one_time_boot_state,
        "health_check_commands": ["autotuneai system boot-doctor --json", "autotuneai system kernel-inventory --json"],
        "rollback_commands": _rollback_commands(inventory),
        "public_placeholders_only": True,
        "kernel_or_boot_state_modified": False,
    }


def fixture_boot_experiment_result(plan: OneTimeBootPlan, *, health_passed: bool, original_state_restored: bool) -> BootExperimentResult:
    status = "passed" if health_passed and original_state_restored and plan.eligibility_state not in {"blocked"} else "failed"
    blockers = [] if status == "passed" else list(plan.blockers or ["health_or_original_state_verification_failed"])
    return BootExperimentResult(
        profile_id=plan.profile_id,
        status=status,
        plan_digest=versioned_digest(plan.to_dict()),
        boot_attempted=False,
        reboot_performed=False,
        kernel_or_boot_state_modified=False,
        health_status="passed" if health_passed else "failed",
        rollback_status="passed" if original_state_restored else "failed",
        blockers=blockers,
    )


def verify_boot_experiment_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if raw_result.get("schema_version") != "1.0":
        issues.append("missing_or_unsupported_schema_version")
    if raw_result.get("kernel_or_boot_state_modified") is True:
        issues.append("kernel_or_boot_state_modified")
    if raw_result.get("boot_attempted") is True:
        issues.append("real_boot_attempt_is_out_of_scope_for_phase21c0")
    if raw_result.get("reboot_performed") is True:
        issues.append("real_reboot_is_out_of_scope_for_phase21c0")
    if raw_result.get("health_status") != "passed":
        issues.append("post_boot_health_check_failed")
    if raw_result.get("rollback_status") != "passed":
        issues.append("rollback_or_original_state_verification_failed")
    if raw_result.get("status") != "passed":
        issues.append("experiment_result_not_passed")
    return {
        "schema_version": "1.0",
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues), key=issues.index),
        "result_digest": versioned_digest(raw_result) if isinstance(raw_result, dict) else NOT_AVAILABLE,
        "kernel_or_boot_state_modified": False,
        "phase21c0_local_only": True,
    }


def read_kernel_config(reader: RootReader, release: str) -> KernelConfigCapability:
    paths = []
    if release and release != NOT_AVAILABLE:
        paths.append(f"/boot/config-{release}")
    paths.extend(["/proc/config.gz", "/run/perfrunbench-fixtures/kernel_config"])
    for item in paths:
        path = reader.path(item)
        if not path.exists():
            continue
        text = _read_config_path(path)
        digest = file_sha256(path)
        return KernelConfigCapability(source_path=item, digest=digest, capabilities=_config_capabilities(text))
    return KernelConfigCapability(source_path=NOT_AVAILABLE, digest=NOT_AVAILABLE, capabilities={key: "unknown" for key in CONFIG_KEYS})


def _read_config_path(path: Path) -> str:
    if path.suffix == ".gz":
        try:
            return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        except OSError:
            return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _config_capabilities(text: str) -> dict[str, str]:
    result = {}
    for key in CONFIG_KEYS:
        if re.search(rf"^{re.escape(key)}=y$", text, flags=re.MULTILINE):
            result[key] = "supported"
        elif re.search(rf"^# {re.escape(key)} is not set$", text, flags=re.MULTILINE) or re.search(rf"^{re.escape(key)}=n$", text, flags=re.MULTILINE):
            result[key] = "unsupported"
        else:
            result[key] = "unknown"
    return result


def _kernel_package_inventory(reader: RootReader) -> list[dict[str, Any]]:
    fixture = reader.read_text("/run/perfrunbench-fixtures/dpkg_kernel_inventory")
    text = fixture if fixture != NOT_AVAILABLE else _run_readonly(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\t${db:Status-Abbrev}\n", "linux-image*"], reader)
    records = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            records.append({"package": parts[0], "version": parts[1], "architecture": parts[2], "status": parts[3].strip()})
        elif len(parts) >= 2:
            records.append({"package": parts[0], "version": parts[1], "architecture": NOT_AVAILABLE, "status": NOT_AVAILABLE})
    return records


def _run_readonly(args: list[str], reader: RootReader) -> str:
    if reader.is_fake:
        return NOT_AVAILABLE
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return NOT_AVAILABLE
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else NOT_AVAILABLE


def _boot_files(reader: RootReader) -> dict[str, Any]:
    releases = sorted({path.name.removeprefix("vmlinuz-") for path in reader.glob("/boot/vmlinuz-*")})
    files = {}
    for release in releases:
        files[release] = {
            "vmlinuz_path": f"/boot/vmlinuz-{release}",
            "vmlinuz_present": reader.exists(f"/boot/vmlinuz-{release}"),
            "initrd_path": f"/boot/initrd.img-{release}",
            "initrd_present": reader.exists(f"/boot/initrd.img-{release}"),
            "config_path": f"/boot/config-{release}",
            "config_present": reader.exists(f"/boot/config-{release}"),
            "modules_path": f"/lib/modules/{release}",
            "modules_present": reader.exists(f"/lib/modules/{release}"),
        }
    return {"releases": releases, "files": files}


def _detect_bootloader(reader: RootReader) -> tuple[str, dict[str, Any]]:
    grub_cfg = reader.exists("/boot/grub/grub.cfg")
    efi = reader.exists("/sys/firmware/efi") or reader.exists("/boot/efi/EFI")
    systemd_boot = reader.exists("/boot/loader/loader.conf")
    direct_stub = efi and any(reader.glob("/boot/efi/EFI/**/vmlinuz*"))
    if systemd_boot:
        kind = "systemd-boot"
    elif grub_cfg and efi:
        kind = "grub_uefi"
    elif grub_cfg:
        kind = "grub_bios"
    elif direct_stub:
        kind = "direct_efi_stub"
    else:
        kind = "unknown"
    menu_entries = _grub_menu_entries(reader) if grub_cfg else _systemd_boot_entries(reader)
    return kind, {"grub_cfg_present": grub_cfg, "efi_present": efi, "systemd_boot_present": systemd_boot, "menu_entries": menu_entries}


def _grub_menu_entries(reader: RootReader) -> list[str]:
    text = reader.read_text("/boot/grub/grub.cfg")
    if text == NOT_AVAILABLE:
        return []
    return [match.group(1) for match in re.finditer(r"^\s*menuentry ['\"]([^'\"]+)['\"]", text, flags=re.MULTILINE)]


def _systemd_boot_entries(reader: RootReader) -> list[str]:
    entries = []
    for path in reader.glob("/boot/loader/entries/*.conf"):
        title = NOT_AVAILABLE
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("title "):
                title = line.split(None, 1)[1]
                break
        entries.append(title if title != NOT_AVAILABLE else path.name)
    return entries


def _filesystem_inventory(reader: RootReader) -> dict[str, Any]:
    boot_free = reader.read_text("/run/perfrunbench-fixtures/boot_free_bytes")
    root_free = reader.read_text("/run/perfrunbench-fixtures/root_free_bytes")
    if boot_free == NOT_AVAILABLE and not reader.is_fake:
        try:
            boot_free = str(shutil.disk_usage("/boot").free)
        except OSError:
            boot_free = NOT_AVAILABLE
    if root_free == NOT_AVAILABLE and not reader.is_fake:
        try:
            root_free = str(shutil.disk_usage("/").free)
        except OSError:
            root_free = NOT_AVAILABLE
    return {
        "root_filesystem_type": _fixture_or_unknown(reader, "root_filesystem_type"),
        "boot_filesystem_type": _fixture_or_unknown(reader, "boot_filesystem_type"),
        "boot_free_bytes": boot_free,
        "root_free_bytes": root_free,
        "required_boot_free_bytes": 512 * 1024 * 1024,
    }


def _fixture_or_unknown(reader: RootReader, name: str) -> str:
    return reader.read_text(f"/run/perfrunbench-fixtures/{name}")


def _grub_environment(reader: RootReader) -> dict[str, Any]:
    text = reader.read_text("/boot/grub/grubenv")
    values = {}
    if text != NOT_AVAILABLE:
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return {
        "path": "/boot/grub/grubenv" if text != NOT_AVAILABLE else NOT_AVAILABLE,
        "values": values,
        "next_entry": values.get("next_entry", NOT_AVAILABLE),
        "saved_entry": values.get("saved_entry", NOT_AVAILABLE),
    }


def _efi_state(reader: RootReader) -> dict[str, Any]:
    efiboot = reader.read_text("/run/perfrunbench-fixtures/efibootmgr")
    return {
        "efi_present": reader.exists("/sys/firmware/efi") or reader.exists("/boot/efi/EFI"),
        "efibootmgr_output": efiboot,
        "writes_allowed_in_phase21c0": False,
    }


def _initramfs_inventory(reader: RootReader) -> dict[str, Any]:
    releases = sorted(path.name.removeprefix("initrd.img-") for path in reader.glob("/boot/initrd.img-*"))
    return {"available_initrds": releases or NOT_AVAILABLE, "generation_allowed_in_phase21c0": False}


def _default_boot_entry(reader: RootReader, grub_environment: dict[str, Any], bootloader_evidence: dict[str, Any], current_release: str) -> str:
    fixture = reader.read_text("/run/perfrunbench-fixtures/default_boot_entry")
    if fixture != NOT_AVAILABLE:
        return fixture
    saved = grub_environment.get("saved_entry")
    if saved and saved != NOT_AVAILABLE:
        return str(saved)
    entries = bootloader_evidence.get("menu_entries", [])
    for entry in entries:
        if current_release in str(entry):
            return str(entry)
    return entries[0] if entries else NOT_AVAILABLE


def _fallback_release(boot_files: dict[str, Any], current_release: str) -> str:
    releases = [item for item in boot_files.get("releases", []) if item != current_release]
    return releases[0] if releases else NOT_AVAILABLE


def _recovery_capability(reader: RootReader, snapshot: Any, bootloader: str, fallback_release: str, boot_files: dict[str, Any], initramfs: dict[str, Any]) -> BootRecoveryCapability:
    snapshot_marker = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/recovery_snapshot_marker"), true_state="verified", false_state="missing")
    if snapshot_marker == "missing" and snapshot.snapshot_marker not in {"", NOT_AVAILABLE, None}:
        snapshot_marker = "private_marker_recorded"
    serial = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/serial_console_verified"), true_state="verified", false_state="not_verified")
    oob = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/out_of_band_recovery_verified"), true_state="verified", false_state="not_verified")
    volume = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/volume_recovery_path_verified"), true_state="verified", false_state="not_verified")
    boot_menu = "verified" if bootloader != "unknown" else "unknown"
    manual = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/manual_recovery_instructions_present"), true_state="present", false_state="missing")
    one_time = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/one_time_boot_verified"), true_state="verified", false_state="unknown")
    ssh = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/ssh_post_boot_check"), true_state="defined", false_state="missing")
    rollback = _state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/rollback_commands_defined"), true_state="defined", false_state="missing")
    authorization = "explicit_user_authorization_required"
    fallback_state = "verified" if fallback_release != NOT_AVAILABLE and _kernel_files_complete(boot_files, initramfs, fallback_release) else "missing"
    blockers = []
    if snapshot_marker == "missing":
        blockers.extend(["missing_recovery_snapshot", "missing_snapshot_marker"])
    if serial != "verified":
        blockers.append("serial_console_not_verified")
    if oob != "verified" and volume != "verified":
        blockers.append("no_out_of_band_recovery")
    if fallback_state == "missing":
        blockers.append("fallback_kernel_missing")
    if one_time != "verified":
        blockers.append("one_time_boot_unavailable")
    if ssh != "defined":
        blockers.append("ssh_post_boot_check_missing")
    if rollback != "defined":
        blockers.append("rollback_command_missing")
    blockers.append("user_authorization_missing")
    return BootRecoveryCapability(
        snapshot_marker_state=snapshot_marker,
        recovery_snapshot_state=snapshot_marker,
        serial_console_state=serial,
        out_of_band_recovery_state=oob,
        fallback_kernel_state=fallback_state,
        fallback_kernel=fallback_release,
        root_volume_identity_state=_state_from_boolish(reader.read_text("/run/perfrunbench-fixtures/root_volume_identity"), true_state="recorded", false_state="missing"),
        volume_recovery_state=volume,
        boot_menu_access_state=boot_menu,
        manual_recovery_instructions_state=manual,
        one_time_boot_state=one_time,
        ssh_post_boot_check_state=ssh,
        rollback_command_state=rollback,
        user_authorization_state=authorization,
        blockers=sorted(set(blockers), key=blockers.index),
    )


def _current_kernel_candidate(reader: RootReader, release: str, flavor: str, arch: str, page_size: str, config: KernelConfigCapability, packages: list[dict[str, Any]], boot_files: dict[str, Any], initramfs: dict[str, Any]) -> KernelCandidate:
    package = _package_for_release(packages, release)
    blockers = []
    if package.get("package") == NOT_AVAILABLE:
        blockers.append("kernel_package_owner_unknown")
    elif not _package_installed(package):
        blockers.append("kernel_package_not_installed")
    if not _kernel_files_complete(boot_files, initramfs, release):
        blockers.append("kernel_files_incomplete")
    if config.source_path == NOT_AVAILABLE:
        blockers.append("kernel_config_support_unknown")
    return KernelCandidate(
        candidate_id="current",
        release=release,
        flavor=flavor,
        package_name=package.get("package", NOT_AVAILABLE),
        package_version=package.get("version", NOT_AVAILABLE),
        architecture=package.get("architecture", arch),
        source_state="currently_booted",
        vmlinuz_path=f"/boot/vmlinuz-{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        initrd_path=f"/boot/initrd.img-{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        modules_path=f"/lib/modules/{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        base_page_size=page_size,
        config=config,
        qualification_state="qualified_known_good" if not blockers else "blocked",
        blockers=blockers,
    )


def _fallback_kernel_candidate(reader: RootReader, release: str, arch: str, config: KernelConfigCapability, packages: list[dict[str, Any]], boot_files: dict[str, Any], initramfs: dict[str, Any], current_release: str) -> KernelCandidate:
    package = _package_for_release(packages, release)
    blockers = []
    if release == NOT_AVAILABLE or release == current_release:
        blockers.append("fallback_kernel_missing")
    if package.get("package") == NOT_AVAILABLE:
        blockers.append("kernel_package_owner_unknown")
    elif not _package_installed(package):
        blockers.append("fallback_package_not_installed")
    if release != NOT_AVAILABLE and not _kernel_files_complete(boot_files, initramfs, release):
        blockers.append("fallback_files_incomplete")
    if config.source_path == NOT_AVAILABLE:
        blockers.append("kernel_config_support_unknown")
    page_size = _page_size_from_config(config)
    return KernelCandidate(
        candidate_id="fallback",
        release=release,
        flavor=_kernel_flavor(release),
        package_name=package.get("package", NOT_AVAILABLE),
        package_version=package.get("version", NOT_AVAILABLE),
        architecture=package.get("architecture", arch),
        source_state="installed_fallback",
        vmlinuz_path=f"/boot/vmlinuz-{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        initrd_path=f"/boot/initrd.img-{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        modules_path=f"/lib/modules/{release}" if release != NOT_AVAILABLE else NOT_AVAILABLE,
        base_page_size=page_size,
        config=config,
        qualification_state="qualified_fallback" if not blockers else "blocked",
        blockers=blockers,
    )


def _candidate_64k_kernel(reader: RootReader, arch: str, distro: dict[str, str]) -> KernelCandidate:
    metadata = _kernel_repository_metadata(reader)
    picked = next((item for item in metadata if _is_official_aws_arm64_64k(item, distro)), None)
    config = _candidate_64k_config(reader)
    blockers = []
    if picked is None:
        blockers.append("kernel_package_candidate_unknown")
    if config.capabilities.get("CONFIG_ARM64_64K_PAGES") == "unknown":
        blockers.append("kernel_config_support_unknown")
    elif config.capabilities.get("CONFIG_ARM64_64K_PAGES") != "supported":
        blockers.append("candidate_not_64k")
    return KernelCandidate(
        candidate_id="official-aws-arm64-64k",
        release=picked.get("release", NOT_AVAILABLE) if picked else NOT_AVAILABLE,
        flavor="aws-64k",
        package_name=picked.get("package", NOT_AVAILABLE) if picked else NOT_AVAILABLE,
        package_version=picked.get("version", NOT_AVAILABLE) if picked else NOT_AVAILABLE,
        architecture=picked.get("architecture", arch) if picked else arch,
        source_state="repository_metadata_only",
        vmlinuz_path=NOT_AVAILABLE,
        initrd_path=NOT_AVAILABLE,
        modules_path=NOT_AVAILABLE,
        base_page_size="65536" if "candidate_not_64k" not in blockers else NOT_AVAILABLE,
        config=config,
        qualification_state="metadata_qualified_not_installed" if not blockers else "blocked",
        blockers=blockers,
    )


def _kernel_repository_metadata(reader: RootReader) -> list[dict[str, str]]:
    fixture = reader.read_text("/run/perfrunbench-fixtures/kernel_repository_metadata")
    if fixture == NOT_AVAILABLE:
        fixture = _run_readonly(["apt-cache", "search", "--names-only", "linux-image"], reader)
    records = []
    for line in fixture.splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            parts = line.split("\t")
        else:
            parts = line.split()
        record = {
            "package": parts[0] if len(parts) > 0 else NOT_AVAILABLE,
            "version": parts[1] if len(parts) > 1 else NOT_AVAILABLE,
            "architecture": parts[2] if len(parts) > 2 else NOT_AVAILABLE,
            "origin": parts[3] if len(parts) > 3 else NOT_AVAILABLE,
            "release": parts[4] if len(parts) > 4 else NOT_AVAILABLE,
        }
        records.append(record)
    return records


def _is_official_aws_arm64_64k(item: dict[str, str], distro: dict[str, str]) -> bool:
    package = item.get("package", "").lower()
    arch = item.get("architecture", "").lower()
    origin = item.get("origin", "").lower()
    distro_id = distro.get("ID", "").lower()
    return "linux-image" in package and "aws" in package and "64k" in package and arch == "arm64" and (origin in {"ubuntu", "official", distro_id} or origin == NOT_AVAILABLE.lower())


def _candidate_64k_config(reader: RootReader) -> KernelConfigCapability:
    path = reader.path("/run/perfrunbench-fixtures/candidate_64k_kernel_config")
    if path.exists():
        text = _read_config_path(path)
        return KernelConfigCapability(source_path="/run/perfrunbench-fixtures/candidate_64k_kernel_config", digest=file_sha256(path), capabilities=_config_capabilities(text))
    return KernelConfigCapability(source_path=NOT_AVAILABLE, digest=NOT_AVAILABLE, capabilities={key: "unknown" for key in CONFIG_KEYS})


def _blocker_matrix(
    *,
    reader: RootReader,
    bootloader: str,
    recovery: BootRecoveryCapability,
    current: KernelCandidate,
    fallback: KernelCandidate,
    candidate_64k: KernelCandidate,
    filesystem: dict[str, Any],
    secure_boot_state: str,
    lockdown_state: str,
) -> dict[str, dict[str, Any]]:
    active = {code: {"state": "clear", "evidence": ""} for code in BLOCKER_CODES}
    for code in recovery.blockers:
        _block(active, code, "recovery prerequisite missing or unverified")
    if fallback.qualification_state == "blocked":
        _block(active, "fallback_kernel_unverified", ",".join(fallback.blockers))
    if current.qualification_state == "blocked" and not set(current.blockers).issubset({"kernel_config_support_unknown"}):
        _block(active, "other", ",".join(current.blockers))
    if bootloader == "unknown":
        _block(active, "bootloader_unknown", "boot arrangement could not be identified")
    if candidate_64k.qualification_state == "blocked":
        for blocker in candidate_64k.blockers:
            if blocker in active:
                _block(active, blocker, "64K candidate qualification did not pass")
            else:
                _block(active, "kernel_package_candidate_unknown", blocker)
    if current.config.source_path == NOT_AVAILABLE:
        _block(active, "kernel_config_support_unknown", "current kernel config file was not available")
    boot_free = _int_or_none(filesystem.get("boot_free_bytes"))
    required = _int_or_none(filesystem.get("required_boot_free_bytes"))
    if boot_free is None or required is None or boot_free < required:
        _block(active, "insufficient_boot_partition_space", "boot filesystem free space is unknown or below estimate")
    if secure_boot_state == NOT_AVAILABLE or lockdown_state == NOT_AVAILABLE:
        _block(active, "secure_boot_or_lockdown_unknown", "Secure Boot or lockdown state is unknown")
    if not _has_initramfs_tool_or_fixture(reader):
        _block(active, "initramfs_generation_unknown", "initramfs generation capability was not verified")
    return active


def _block(matrix: dict[str, dict[str, Any]], code: str, evidence: str) -> None:
    matrix.setdefault(code, {"state": "clear", "evidence": ""})
    matrix[code] = {"state": "blocked", "evidence": evidence}


def _has_initramfs_tool_or_fixture(reader: RootReader) -> bool:
    fixture = reader.read_text("/run/perfrunbench-fixtures/initramfs_generation_verified")
    if fixture != NOT_AVAILABLE:
        return _boolish(fixture)
    return not reader.is_fake and shutil.which("update-initramfs") is not None


def _plan_blockers(profile_id: str, inventory: BootInventory, candidate: KernelCandidate, additions: list[str]) -> list[str]:
    blockers = []
    duplicate_keys = _duplicate_kernel_parameter_keys(inventory.current_command_line, additions)
    if duplicate_keys:
        blockers.extend(f"duplicate_kernel_parameter:{key}" for key in duplicate_keys)
    if profile_id != "original-4k-default":
        for code in ["missing_recovery_snapshot", "missing_snapshot_marker", "serial_console_not_verified", "no_out_of_band_recovery", "fallback_kernel_missing", "fallback_kernel_unverified", "bootloader_unknown", "one_time_boot_unavailable", "ssh_post_boot_check_missing", "rollback_command_missing", "user_authorization_missing"]:
            if inventory.blocker_matrix[code]["state"] == "blocked":
                blockers.append(code)
    if candidate.qualification_state == "blocked":
        blockers.extend(candidate.blockers)
    if profile_id.startswith("64k"):
        for code in ["kernel_package_candidate_unknown", "kernel_config_support_unknown", "insufficient_boot_partition_space"]:
            if inventory.blocker_matrix[code]["state"] == "blocked":
                blockers.append(code)
    for arg in additions:
        if arg.startswith("nohz_full") and inventory.current_kernel_config.capabilities.get("CONFIG_NO_HZ_FULL") != "supported" and candidate.config.capabilities.get("CONFIG_NO_HZ_FULL") != "supported":
            blockers.append("CONFIG_NO_HZ_FULL")
        if arg.startswith("rcu_nocbs") and inventory.current_kernel_config.capabilities.get("CONFIG_RCU_NOCB_CPU") != "supported" and candidate.config.capabilities.get("CONFIG_RCU_NOCB_CPU") != "supported":
            blockers.append("CONFIG_RCU_NOCB_CPU")
        if arg.startswith("isolcpus") and inventory.current_kernel_config.capabilities.get("CONFIG_CPU_ISOLATION") != "supported" and candidate.config.capabilities.get("CONFIG_CPU_ISOLATION") != "supported":
            blockers.append("CONFIG_CPU_ISOLATION")
    return sorted(set(blockers), key=blockers.index)


def _candidate_for_profile(profile_id: str, inventory: BootInventory) -> KernelCandidate:
    if profile_id.startswith("64k"):
        return inventory.candidate_64k_kernel
    return inventory.current_kernel


def _profile_args(profile_id: str, inventory: BootInventory, housekeeping: str, workload: str) -> list[str]:
    if profile_id in {"original-4k-default", "64k-default"}:
        return []
    if profile_id.endswith("nohz-rcu-irq"):
        return [f"nohz_full={workload}", f"rcu_nocbs={workload}", f"irqaffinity={housekeeping}"]
    if profile_id.endswith("managed-irq-isolation"):
        return [f"isolcpus=managed_irq,{workload}"]
    if profile_id == "4k-domain-isolation":
        return [f"isolcpus=domain,managed_irq,{workload}", f"nohz_full={workload}", f"rcu_nocbs={workload}", f"irqaffinity={housekeeping}"]
    return []


def _profile_description(profile_id: str) -> str:
    return {
        "original-4k-default": "Current known-good AWS 4K kernel with original command line.",
        "4k-nohz-rcu-irq": "4K one-time bounded nohz_full, RCU callback, and IRQ affinity isolation plan.",
        "4k-managed-irq-isolation": "4K one-time managed IRQ isolation plan without domain isolation.",
        "4k-domain-isolation": "Advanced less-flexible 4K domain plus managed IRQ isolation plan.",
        "64k-default": "Official AWS ARM64 64K kernel with original non-experimental command line.",
        "64k-nohz-rcu-irq": "64K one-time bounded nohz_full, RCU callback, and IRQ affinity isolation plan.",
        "64k-managed-irq-isolation": "64K managed IRQ isolation plan gated on successful earlier 64K profiles.",
    }[profile_id]


def _append_non_duplicate_args(current: str, additions: list[str]) -> str:
    parts = [] if current in {"", NOT_AVAILABLE} else current.split()
    existing_keys = {item.split("=", 1)[0] for item in parts}
    filtered_additions = [item for item in additions if item.split("=", 1)[0] not in existing_keys]
    return " ".join([*parts, *filtered_additions]).strip()


def _duplicate_kernel_parameter_keys(current: str, additions: list[str]) -> list[str]:
    parts = [] if current in {"", NOT_AVAILABLE} else current.split()
    existing_keys = {item.split("=", 1)[0] for item in parts}
    duplicates = []
    for item in additions:
        key = item.split("=", 1)[0]
        if key in existing_keys:
            duplicates.append(key)
    return sorted(set(duplicates), key=duplicates.index)


def _validate_cpu_layout(housekeeping: str, workload: str) -> None:
    house = _expand_cpu_list(housekeeping)
    work = _expand_cpu_list(workload)
    if not house:
        raise SystemProfileError("housekeeping CPU range must not be empty")
    if not work:
        raise SystemProfileError("workload CPU range must not be empty")
    if set(house) & set(work):
        raise SystemProfileError("housekeeping and workload CPU ranges must be disjoint")


def _expand_cpu_list(text: str) -> list[int]:
    cpus = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise SystemProfileError("malformed CPU range")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise SystemProfileError("malformed CPU range")
            cpus.extend(range(start, end + 1))
        else:
            if not part.isdigit():
                raise SystemProfileError("malformed CPU range")
            cpus.append(int(part))
    return cpus


def _rollback_commands(inventory: BootInventory) -> list[str]:
    if inventory.bootloader in {"grub_bios", "grub_uefi"}:
        return ["grub-editenv list", "grub-reboot <known-good-entry>", "verify original default entry after reboot"]
    if inventory.bootloader == "systemd-boot":
        return ["bootctl list", "bootctl set-oneshot <known-good-entry>", "verify original default entry after reboot"]
    return []


def _one_time_boot_mechanism(inventory: BootInventory) -> str:
    if inventory.recovery.one_time_boot_state != "verified":
        return "unknown"
    if inventory.bootloader in {"grub_bios", "grub_uefi"}:
        return "grub-reboot next_entry"
    if inventory.bootloader == "systemd-boot":
        return "bootctl set-oneshot"
    return "unknown"


def _fallback_mechanism(inventory: BootInventory) -> str:
    if inventory.bootloader in {"grub_bios", "grub_uefi"}:
        return "GRUB one-time fallback boot"
    if inventory.bootloader == "systemd-boot":
        return "systemd-boot one-time fallback boot"
    return "unknown"


def _package_for_release(packages: list[dict[str, Any]], release: str) -> dict[str, Any]:
    for item in packages:
        if release and release != NOT_AVAILABLE and release in str(item.get("package", "")):
            return item
    return {"package": NOT_AVAILABLE, "version": NOT_AVAILABLE, "architecture": NOT_AVAILABLE}


def _package_installed(package: dict[str, Any]) -> bool:
    status = str(package.get("status", "")).strip().lower()
    return status.startswith("ii") or "install ok installed" in status


def _kernel_files_complete(boot_files: dict[str, Any], initramfs: dict[str, Any], release: str) -> bool:
    if release == NOT_AVAILABLE:
        return False
    entry = boot_files.get("files", {}).get(release, {})
    return bool(entry.get("vmlinuz_present") and entry.get("initrd_present") and entry.get("config_present") and entry.get("modules_present"))


def _kernel_flavor(release: str) -> str:
    lowered = release.lower()
    if "64k" in lowered:
        return "aws-64k" if "aws" in lowered else "64k"
    if "aws" in lowered:
        return "aws-4k"
    return NOT_AVAILABLE if release == NOT_AVAILABLE else "generic"


def _page_size_from_config(config: KernelConfigCapability) -> str:
    if config.capabilities.get("CONFIG_ARM64_64K_PAGES") == "supported":
        return "65536"
    if config.capabilities.get("CONFIG_ARM64_4K_PAGES") == "supported":
        return "4096"
    return NOT_AVAILABLE


def _secure_boot_state(reader: RootReader) -> str:
    fixture = reader.read_text("/run/perfrunbench-fixtures/secure_boot_state")
    if fixture != NOT_AVAILABLE:
        return fixture
    if reader.exists("/sys/firmware/efi/efivars"):
        return "unknown"
    return NOT_AVAILABLE


def _state_from_boolish(text: str, *, true_state: str, false_state: str) -> str:
    if text == NOT_AVAILABLE:
        return false_state
    return true_state if _boolish(text) else false_state


def _boolish(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "verified", "present", "defined", "ok"}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _logical_cpu_count(inventory: BootInventory) -> int | str:
    return inventory.logical_cpus
