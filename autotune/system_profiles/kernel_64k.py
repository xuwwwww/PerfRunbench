from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any

from autotune.recommendation.schemas import stable_digest
from autotune.system_profiles.schemas import (
    GrubDefaultResolution,
    KernelArtifactInventory,
    KernelPackageCandidate,
    KernelPackageCleanupResult,
    KernelPackageInventory,
    KernelPackageRollbackPlan,
    KernelPackageTransactionPlan,
    KernelPackageTransactionResult,
    NOT_AVAILABLE,
    SCHEMA_VERSION,
    SystemProfileError,
    versioned_digest,
)


PHASE21C2B_PROFILE_IDS = {
    "G0": "original-4k-default",
    "G4": "64k-default",
    "G5": "64k-nohz-rcu-irq",
    "G6": "64k-managed-irq-isolation",
    "G7": "64k-domain-isolation-interaction",
}

PHASE21C2B_PROFILE_NAMES = set(PHASE21C2B_PROFILE_IDS.values())

PHASE21C2B_REQUIRED_AUTHORIZATION_FLAGS = [
    "--allow-kernel-package-install",
    "--allow-temporary-grub-default-safety-pin",
    "--allow-guest-reboot",
    "--recovery-marker",
    "--serial-recovery-marker",
    "--maximum-reboots",
]

FORBIDDEN_PACKAGE_ACTIONS = [
    "add_repository",
    "use_ppa",
    "apt_upgrade",
    "autoremove",
    "remove_current_4k_kernel",
    "remove_fallback_kernel",
    "install_generic_non_aws_kernel",
    "change_architecture",
]

FORBIDDEN_ROLLBACK_COMMANDS = ["apt autoremove", "mkfs", "update-grub without verification"]


@dataclass(frozen=True)
class GrubMenuEntry:
    title: str
    entry_id: str
    linux_command: str
    linux_path: str
    linux_arguments: str
    initrd_command: str
    initrd_path: str
    loader_prelude: list[str]
    submenu_path: list[str]
    source_index: int
    block_text: str
    is_recovery: bool
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "entry_id": self.entry_id,
            "linux_command": self.linux_command,
            "linux_path": self.linux_path,
            "linux_arguments": self.linux_arguments,
            "initrd_command": self.initrd_command,
            "initrd_path": self.initrd_path,
            "loader_prelude": list(self.loader_prelude),
            "submenu_path": list(self.submenu_path),
            "source_index": self.source_index,
            "block_text": self.block_text,
            "is_recovery": self.is_recovery,
        }


def parse_grub_menu_entries(grub_cfg_text: str) -> list[GrubMenuEntry]:
    """Parse generated GRUB menu entries without guessing kernel paths."""

    lines = grub_cfg_text.splitlines()
    entries: list[GrubMenuEntry] = []
    submenu_stack: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("submenu "):
            submenu_stack.append(_parse_grub_title(stripped, keyword="submenu"))
            index += 1
            continue
        if stripped == "}" and submenu_stack:
            submenu_stack.pop()
            index += 1
            continue
        if not stripped.startswith("menuentry "):
            index += 1
            continue
        block, end_index = _collect_grub_block(lines, index)
        entries.append(_parse_menuentry_block(block, index, submenu_stack))
        index = end_index + 1
    return entries


def qualify_generated_grub_entry(
    entries: list[dict[str, Any]] | list[GrubMenuEntry],
    *,
    kernel_release: str,
    expected_page_size: str | None = None,
    require_initrd: bool = True,
) -> dict[str, Any]:
    normalized = [_coerce_entry(item) for item in entries]
    candidates = [
        item
        for item in normalized
        if kernel_release
        and kernel_release != NOT_AVAILABLE
        and kernel_release in item.linux_path
        and not item.is_recovery
    ]
    issues: list[str] = []
    if not candidates:
        issues.append("generated_kernel_entry_missing")
    if len(candidates) > 1:
        issues.append("ambiguous_duplicate_kernel_entries")
    entry = candidates[0] if len(candidates) == 1 else None
    if entry and not entry.linux_path:
        issues.append("malformed_linux_line")
    if entry and require_initrd and entry.initrd_path == NOT_AVAILABLE:
        issues.append("generated_entry_initrd_missing")
    if entry and expected_page_size and expected_page_size not in {"", NOT_AVAILABLE}:
        if expected_page_size == "65536" and "64k" not in kernel_release.lower():
            issues.append("page_size_expectation_does_not_match_release")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified" if not issues else "blocked",
        "issues": issues,
        "entry": entry.to_dict() if entry else {},
        "path_source": "generated_grub_menu_entry",
        "path_guess_allowed": False,
    }


def render_temporary_grub_entry_from_generated(
    *,
    title: str,
    entry_id: str,
    source_entry: dict[str, Any] | GrubMenuEntry,
    linux_arguments: str,
) -> str:
    entry = _coerce_entry(source_entry)
    if not entry.linux_path:
        raise SystemProfileError("source GRUB entry has no parsed linux path")
    source_lines = entry.block_text.splitlines()
    if not source_lines:
        raise SystemProfileError("source GRUB entry has no block text")
    rendered = [f"menuentry '{title}' --id '{entry_id}' {{"]
    linux_replaced = False
    for raw_line in source_lines[1:]:
        stripped = raw_line.strip()
        if stripped.startswith(("linux ", "linuxefi ")):
            parts = stripped.split(maxsplit=2)
            if len(parts) < 2:
                raise SystemProfileError("source GRUB entry has malformed linux line")
            command, path = parts[0], parts[1]
            rendered.append(f"    {command} {path} {linux_arguments}".rstrip())
            linux_replaced = True
        else:
            rendered.append(raw_line)
    if not linux_replaced:
        raise SystemProfileError("source GRUB entry has no linux line to replace")
    rendered.append("")
    return "\n".join(rendered)


def build_kernel_package_transaction_plan(
    inventory: dict[str, Any],
    *,
    apt_simulation_text: str | None = None,
    profile_id: str = "aws-arm64-64k",
) -> KernelPackageTransactionPlan:
    if profile_id != "aws-arm64-64k":
        raise SystemProfileError(f"unknown kernel package profile: {profile_id}")
    package_inventory = _package_inventory_from_boot_inventory(inventory)
    candidate = _package_candidate_from_inventory(inventory)
    simulation = parse_apt_simulation(apt_simulation_text or "")
    dependency_changes = {
        "new": simulation["new_packages"],
        "upgraded": simulation["upgraded_packages"],
        "removed": simulation["removed_packages"],
        "held": simulation["held_packages"],
    }
    blockers = _transaction_blockers(package_inventory, candidate, simulation)
    expected_release = candidate.expected_release
    expected = {
        "vmlinuz_path": f"/boot/vmlinuz-{expected_release}" if expected_release != NOT_AVAILABLE else NOT_AVAILABLE,
        "initrd_path": f"/boot/initrd.img-{expected_release}" if expected_release != NOT_AVAILABLE else NOT_AVAILABLE,
        "config_path": f"/boot/config-{expected_release}" if expected_release != NOT_AVAILABLE else NOT_AVAILABLE,
        "system_map_path": f"/boot/System.map-{expected_release}" if expected_release != NOT_AVAILABLE else NOT_AVAILABLE,
        "modules_path": f"/lib/modules/{expected_release}" if expected_release != NOT_AVAILABLE else NOT_AVAILABLE,
    }
    install_packages = _exact_install_packages(candidate, simulation)
    install_command = ["apt-get", "install", "--no-install-recommends", *[f"{name}={candidate.candidate_version}" for name in install_packages if candidate.candidate_version != NOT_AVAILABLE]]
    rollback = KernelPackageRollbackPlan(
        remove_packages=install_packages,
        preserve_packages=_installed_kernel_package_names(package_inventory),
        forbidden_commands=FORBIDDEN_ROLLBACK_COMMANDS,
        dpkg_recovery_commands=["dpkg --audit", "apt-get -f install --simulate"],
        final_original_state_verification=[
            "running original 4K kernel",
            "PAGE_SIZE 4096",
            "original resolved permanent default",
            "original kernel packages still installed",
            "no residual 64K package artifacts",
        ],
        package_cache_cleanup_policy="clean only exact transaction downloads; do not autoremove",
    )
    return KernelPackageTransactionPlan(
        profile_id=profile_id,
        package_inventory=package_inventory,
        candidate=candidate,
        apt_simulation=simulation,
        dependency_changes=dependency_changes,
        packages_newly_installed=simulation["new_packages"],
        packages_upgraded=simulation["upgraded_packages"],
        packages_removed=simulation["removed_packages"],
        packages_held=simulation["held_packages"],
        expected_artifacts=expected,
        expected_grub_entries=[expected_release] if expected_release != NOT_AVAILABLE else [],
        expected_page_size="65536",
        real_install_command=install_command,
        rollback_plan=rollback,
        eligibility_state="planning_ready" if not blockers else "blocked",
        blockers=blockers,
        kernel_or_boot_state_modified=False,
    )


def verify_kernel_package_transaction_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if raw_plan.get("schema_version") != SCHEMA_VERSION:
        issues.append("missing_or_unsupported_schema_version")
    if raw_plan.get("kernel_or_boot_state_modified") is not False:
        issues.append("kernel_or_boot_state_modified")
    command = raw_plan.get("real_install_command", [])
    if "upgrade" in [str(item) for item in command]:
        issues.append("apt_upgrade_forbidden")
    rollback = raw_plan.get("rollback_plan", {})
    if "apt autoremove" in rollback.get("forbidden_commands", []) and any("autoremove" in str(item) for item in rollback.get("remove_packages", [])):
        issues.append("autoremove_forbidden")
    if raw_plan.get("packages_removed"):
        issues.append("package_removal_in_simulation")
    candidate = raw_plan.get("candidate", {})
    if candidate.get("package_architecture") not in {"arm64", NOT_AVAILABLE}:
        issues.append("incorrect_architecture")
    if raw_plan.get("eligibility_state") == "planning_ready" and raw_plan.get("blockers"):
        issues.append("planning_ready_with_blockers")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues), key=issues.index),
        "plan_digest": versioned_digest(raw_plan) if isinstance(raw_plan, dict) else NOT_AVAILABLE,
        "kernel_or_boot_state_modified": False,
    }


def parse_apt_simulation(text: str) -> dict[str, Any]:
    new_packages: list[str] = []
    upgraded: list[str] = []
    removed: list[str] = []
    held: list[str] = []
    unauthenticated = False
    conflicts = False
    bootloader_replacement = False
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if "warning" in lowered and ("unauthenticated" in lowered or "not authenticated" in lowered or "cannot be authenticated" in lowered):
            unauthenticated = True
        if "conflict" in lowered or "held broken packages" in lowered:
            conflicts = True
        if lowered.startswith("inst "):
            package = stripped.split()[1]
            if _is_kernel_package(package):
                if "[" in stripped:
                    upgraded.append(package)
                else:
                    new_packages.append(package)
            elif package in {"grub-efi-arm64", "grub-pc", "shim-signed"}:
                bootloader_replacement = True
            else:
                upgraded.append(package)
        elif lowered.startswith("remv ") or lowered.startswith("remove "):
            parts = stripped.split()
            if len(parts) > 1:
                removed.append(parts[1])
        elif lowered.startswith("hold ") or "kept back" in lowered:
            parts = stripped.split()
            if len(parts) > 1:
                held.append(parts[1])
    return {
        "schema_version": SCHEMA_VERSION,
        "provided": bool(text.strip()),
        "new_packages": sorted(set(new_packages)),
        "upgraded_packages": sorted(set(upgraded)),
        "removed_packages": sorted(set(removed)),
        "held_packages": sorted(set(held)),
        "unauthenticated": unauthenticated,
        "dependency_conflicts": conflicts,
        "bootloader_replacement": bootloader_replacement,
    }


def build_grub_default_plan(inventory: dict[str, Any], *, safe_entry: str) -> GrubDefaultResolution:
    entries = _menu_entry_details(inventory)
    entry_ids = [str(item.get("entry_id", NOT_AVAILABLE)) for item in entries]
    titles = [str(item.get("title", NOT_AVAILABLE)) for item in entries]
    saved = str(inventory.get("grub_environment", {}).get("saved_entry", NOT_AVAILABLE))
    next_entry = str(inventory.get("grub_environment", {}).get("next_entry", NOT_AVAILABLE))
    default = str(inventory.get("current_default_boot_entry", NOT_AVAILABLE))
    resolved = _resolve_entry(entries, saved if saved != NOT_AVAILABLE else default)
    safe = _resolve_entry(entries, safe_entry)
    blockers = []
    if not safe:
        blockers.append("safe_entry_missing")
    elif safe.get("is_recovery"):
        blockers.append("safe_entry_is_recovery")
    if safe and "perfrunbench" in str(safe.get("entry_id", "")).lower():
        blockers.append("permanent_experimental_default_rejected")
    resolved_kernel = _kernel_release_from_entry(resolved) if resolved else NOT_AVAILABLE
    safe_kernel = _kernel_release_from_entry(safe) if safe else NOT_AVAILABLE
    safety_pin_required = bool(safe and resolved and resolved.get("entry_id") != safe.get("entry_id"))
    if not safe:
        blockers.append("safe_default_unavailable")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mechanism": "guarded_grub_saved_default_safety_pin",
        "apply_allowed_in_phase21c2b0": False,
        "safe_entry_id": safe.get("entry_id", NOT_AVAILABLE) if safe else NOT_AVAILABLE,
        "safe_kernel": safe_kernel,
        "backup_required": True,
        "restore_byte_for_byte": True,
        "never_pin_experimental_profile": True,
    }
    return GrubDefaultResolution(
        status="planning_ready" if not blockers else "blocked",
        configuration_text_digest=str(inventory.get("bootloader_evidence", {}).get("grub_cfg_digest", NOT_AVAILABLE)),
        grub_environment_digest=stable_digest(inventory.get("grub_environment", {})),
        saved_entry=saved,
        next_entry=next_entry,
        generated_menu_order=[*entry_ids, *[title for title in titles if title not in entry_ids]],
        resolved_default_entry_id=str(resolved.get("entry_id", NOT_AVAILABLE)) if resolved else NOT_AVAILABLE,
        resolved_default_kernel=resolved_kernel,
        configuration_text_unchanged=True,
        resolved_default_boot_target_unchanged=not safety_pin_required,
        safety_pin_required=safety_pin_required,
        safety_pin_plan=plan,
        blockers=sorted(set(blockers), key=blockers.index),
        kernel_or_boot_state_modified=False,
    )


def build_kernel_experiment_plan(
    profile_id: str,
    inventory: dict[str, Any],
    *,
    allow_domain_isolation_interaction: bool = False,
    completed_profiles: list[str] | None = None,
) -> dict[str, Any]:
    canonical = PHASE21C2B_PROFILE_IDS.get(profile_id, profile_id)
    if canonical not in PHASE21C2B_PROFILE_NAMES:
        raise SystemProfileError(f"unknown Phase 21C2B profile: {profile_id}")
    additions = _profile_additions(canonical)
    entries = _menu_entry_details(inventory)
    current = _qualify_from_inventory_entries(inventory, str(inventory.get("current_kernel_release", NOT_AVAILABLE)), require_initrd=True)
    candidate = _qualify_from_inventory_entries(inventory, str(inventory.get("candidate_64k_kernel", {}).get("release", NOT_AVAILABLE)), require_initrd=True)
    source = current if canonical == "original-4k-default" else candidate
    entry = source.get("entry", {})
    blockers = list(source.get("issues", []))
    completed = set(completed_profiles or [])
    if canonical == "64k-domain-isolation-interaction":
        if not allow_domain_isolation_interaction:
            blockers.append("domain_isolation_interaction_authorization_missing")
        for required in ["64k-default", "64k-nohz-rcu-irq", "64k-managed-irq-isolation"]:
            if required not in completed:
                blockers.append(f"required_prior_profile_missing:{required}")
    command_line = _append_profile_args(str(inventory.get("current_command_line", "")), additions)
    entry_id = f"perfrunbench-phase21c2b-{canonical}-{stable_digest({'profile': canonical, 'cmdline': command_line})[:12]}"
    menuentry_text = ""
    if entry:
        menuentry_text = render_temporary_grub_entry_from_generated(
            title=f"PerfRunbench Phase 21C2B {canonical}",
            entry_id=entry_id,
            source_entry=entry,
            linux_arguments=_linux_arguments(command_line),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "21C2B",
        "profile_id": canonical,
        "source_generated_entry": entry,
        "temporary_entry_plan": {
            "schema_version": SCHEMA_VERSION,
            "temporary_entry_required": canonical != "original-4k-default",
            "entry_id": entry_id if canonical != "original-4k-default" else NOT_AVAILABLE,
            "menuentry_text": menuentry_text,
            "path_source": "generated_grub_menu_entry",
            "path_guess_allowed": False,
            "appended_parameters": additions,
            "starts_from_profile": "original-4k-default",
            "returns_to_profile": "original-4k-default",
            "refuse_experimental_to_experimental_chain": True,
            "permanent_default_change_allowed": False,
        },
        "advanced_interaction_candidate": canonical == "64k-domain-isolation-interaction",
        "prior_4k_domain_isolation_regressed": canonical == "64k-domain-isolation-interaction",
        "authorization_required": PHASE21C2B_REQUIRED_AUTHORIZATION_FLAGS
        + (["--allow-domain-isolation-interaction"] if canonical == "64k-domain-isolation-interaction" else []),
        "eligibility_state": "planning_ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers), key=blockers.index),
        "comparison_identities": build_phase21c2b_comparison_identities(),
        "kernel_or_boot_state_modified": False,
    }


def verify_kernel_experiment_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if raw_plan.get("schema_version") != SCHEMA_VERSION:
        issues.append("missing_or_unsupported_schema_version")
    temp = raw_plan.get("temporary_entry_plan", {})
    if temp.get("path_guess_allowed") is not False:
        issues.append("guessed_kernel_path_allowed")
    if temp.get("permanent_default_change_allowed") is not False:
        issues.append("permanent_default_change_not_refused")
    if temp.get("starts_from_profile") != "original-4k-default" or temp.get("returns_to_profile") != "original-4k-default":
        issues.append("mandatory_return_to_g0_missing")
    if raw_plan.get("profile_id") == "64k-domain-isolation-interaction":
        auth = raw_plan.get("authorization_required", [])
        if "--allow-domain-isolation-interaction" not in auth:
            issues.append("g7_authorization_gate_missing")
        if raw_plan.get("advanced_interaction_candidate") is not True:
            issues.append("g7_advanced_marker_missing")
    if raw_plan.get("kernel_or_boot_state_modified") is not False:
        issues.append("kernel_or_boot_state_modified")
    return {"schema_version": SCHEMA_VERSION, "status": "passed" if not issues else "failed", "issues": issues, "kernel_or_boot_state_modified": False}


def build_phase21c2b_comparison_identities() -> list[dict[str, Any]]:
    return [
        _identity("H1-4k-full-capacity-baseline", "page_size_baseline", "original-4k-default", "4096", "0-7", 8, "page_size_effect_full_capacity"),
        _identity("H2-4k-matched-six-cpu-baseline", "page_size_baseline", "original-4k-default", "4096", "2-7", 6, "page_size_effect_matched_six_cpu"),
        _identity("H3-64k-full-capacity-baseline", "page_size_experimental", "64k-default", "65536", "0-7", 8, "page_size_effect_full_capacity"),
        _identity("H4-64k-matched-six-cpu-baseline", "isolation_baseline", "64k-default", "65536", "2-7", 6, "page_size_effect_matched_six_cpu_and_isolation_baseline"),
        _identity("H5-64k-isolated-six-cpu-profile", "isolation_experimental", "64k-nohz-rcu-irq|64k-managed-irq-isolation|64k-domain-isolation-interaction", "65536", "2-7", 6, "64k_isolation_effect"),
    ]


def validate_phase21c2b_comparison(identities: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item.get("identity_id"): item for item in identities}
    issues = []
    for left, right in [
        ("H1-4k-full-capacity-baseline", "H3-64k-full-capacity-baseline"),
        ("H2-4k-matched-six-cpu-baseline", "H4-64k-matched-six-cpu-baseline"),
        ("H4-64k-matched-six-cpu-baseline", "H5-64k-isolated-six-cpu-profile"),
    ]:
        if left not in by_id or right not in by_id:
            issues.append(f"comparison_pair_missing:{left}:{right}")
            continue
        a, b = by_id[left], by_id[right]
        if a.get("workload_cpus") != b.get("workload_cpus") or a.get("generation_threads") != b.get("generation_threads"):
            issues.append(f"mismatched_cpu_thread_identity:{left}:{right}")
    return {"schema_version": SCHEMA_VERSION, "status": "passed" if not issues else "failed", "issues": issues}


def build_kernel_artifact_inventory(inventory: dict[str, Any], *, kernel_release: str, expected_page_size: str) -> KernelArtifactInventory:
    boot_files = inventory.get("boot_files", {}).get("files", {})
    files = boot_files.get(kernel_release, {})
    packages = inventory.get("kernel_inventory", [])
    package = next((item for item in packages if kernel_release in str(item.get("package", ""))), {})
    blockers = []
    for key in ["vmlinuz_present", "initrd_present", "config_present", "modules_present"]:
        if not files.get(key):
            blockers.append(f"{key}_missing")
    entry = _qualify_from_inventory_entries(inventory, kernel_release, require_initrd=True)
    if entry["status"] != "qualified":
        blockers.extend(entry["issues"])
    return KernelArtifactInventory(
        kernel_release=kernel_release,
        vmlinuz_path=str(files.get("vmlinuz_path", NOT_AVAILABLE)),
        initrd_path=str(files.get("initrd_path", NOT_AVAILABLE)),
        config_path=str(files.get("config_path", NOT_AVAILABLE)),
        system_map_path=str(files.get("system_map_path", f"/boot/System.map-{kernel_release}" if kernel_release != NOT_AVAILABLE else NOT_AVAILABLE)),
        modules_path=str(files.get("modules_path", NOT_AVAILABLE)),
        package_ownership={str(package.get("package", NOT_AVAILABLE)): str(package.get("version", NOT_AVAILABLE))},
        page_size_expectation=expected_page_size,
        qualification_state="qualified" if not blockers else "blocked",
        blockers=sorted(set(blockers), key=blockers.index),
    )


def build_kernel_cleanup_plan(before: dict[str, Any], after: dict[str, Any]) -> KernelPackageCleanupResult:
    before_packages = {str(item.get("package")) for item in before.get("kernel_inventory", [])}
    after_packages = {str(item.get("package")) for item in after.get("kernel_inventory", [])}
    introduced = sorted(item for item in after_packages - before_packages if item and item != "None")
    preserved = sorted(item for item in before_packages if item and item != "None")
    residual = [item for item in introduced if "64k" in item.lower()]
    return KernelPackageCleanupResult(
        status="cleanup_plan_ready",
        removed_packages=introduced,
        preserved_packages=preserved,
        removed_artifacts=[],
        residual_artifacts=residual,
        package_inventory_restored=not residual,
        grub_state_restored=True,
        issues=[] if introduced else ["no_introduced_packages_detected"],
        kernel_or_boot_state_modified=False,
    )


def fake_kernel_package_transaction_result(plan: KernelPackageTransactionPlan, *, status: str = "passed") -> KernelPackageTransactionResult:
    return KernelPackageTransactionResult(
        plan_digest=versioned_digest(plan.to_dict()),
        status=status,
        install_attempted=False,
        kernel_or_boot_state_modified=False,
        installed_packages=[],
        removed_packages=[],
        issues=[] if status == "passed" else ["fixture_transaction_failed"],
    )


def verify_kernel_package_transaction_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if raw_result.get("schema_version") != SCHEMA_VERSION:
        issues.append("missing_or_unsupported_schema_version")
    if raw_result.get("install_attempted") is True:
        issues.append("real_install_is_out_of_scope_for_phase21c2b0")
    if raw_result.get("kernel_or_boot_state_modified") is True:
        issues.append("kernel_or_boot_state_modified")
    if raw_result.get("status") != "passed":
        issues.append("transaction_result_not_passed")
    return {"schema_version": SCHEMA_VERSION, "status": "passed" if not issues else "failed", "issues": issues, "kernel_or_boot_state_modified": False}


def _collect_grub_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block = []
    depth = 0
    seen_open = False
    for index in range(start, len(lines)):
        line = lines[index]
        block.append(line)
        depth += line.count("{")
        if "{" in line:
            seen_open = True
        depth -= line.count("}")
        if seen_open and depth <= 0:
            return block, index
    return block, len(lines) - 1


def _parse_menuentry_block(block: list[str], source_index: int, submenu_stack: list[str]) -> GrubMenuEntry:
    header = block[0].strip()
    title = _parse_grub_title(header, keyword="menuentry")
    entry_id = _parse_grub_id(header) or title
    linux_command = ""
    linux_path = ""
    linux_arguments = ""
    initrd_command = ""
    initrd_path = NOT_AVAILABLE
    prelude: list[str] = []
    linux_seen = False
    for raw in block[1:]:
        stripped = raw.strip()
        if not stripped or stripped == "}" or stripped.startswith("#"):
            continue
        parts = _split_grub_line(stripped)
        if not parts:
            continue
        command = parts[0]
        if command in {"linux", "linuxefi", "linux16"}:
            linux_seen = True
            linux_command = command
            linux_path = parts[1] if len(parts) > 1 else ""
            linux_arguments = " ".join(parts[2:]) if len(parts) > 2 else ""
            continue
        if command in {"initrd", "initrdefi", "initrd16"}:
            initrd_command = command
            initrd_path = parts[1] if len(parts) > 1 else NOT_AVAILABLE
            continue
        if not linux_seen:
            prelude.append(stripped)
    recovery_text = f"{title} {linux_arguments}".lower()
    return GrubMenuEntry(
        title=title,
        entry_id=entry_id,
        linux_command=linux_command or "linux",
        linux_path=linux_path,
        linux_arguments=linux_arguments,
        initrd_command=initrd_command or "initrd",
        initrd_path=initrd_path,
        loader_prelude=prelude,
        submenu_path=list(submenu_stack),
        source_index=source_index,
        block_text="\n".join(block),
        is_recovery="recovery" in recovery_text,
    )


def _parse_grub_title(line: str, *, keyword: str) -> str:
    rest = line[len(keyword) :].strip()
    if not rest:
        return NOT_AVAILABLE
    if rest[0] in {"'", '"'}:
        quote = rest[0]
        end = rest.find(quote, 1)
        return rest[1:end] if end > 0 else NOT_AVAILABLE
    return rest.split()[0]


def _parse_grub_id(line: str) -> str:
    match = re.search(r"--id(?:=|\s+)(?:'([^']+)'|\"([^\"]+)\"|([^\s{]+))", line)
    if not match:
        return ""
    return next(group for group in match.groups() if group)


def _split_grub_line(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.split()


def _coerce_entry(item: dict[str, Any] | GrubMenuEntry) -> GrubMenuEntry:
    if isinstance(item, GrubMenuEntry):
        return item
    return GrubMenuEntry(
        title=str(item.get("title", NOT_AVAILABLE)),
        entry_id=str(item.get("entry_id", item.get("title", NOT_AVAILABLE))),
        linux_command=str(item.get("linux_command", "linux")),
        linux_path=str(item.get("linux_path", "")),
        linux_arguments=str(item.get("linux_arguments", "")),
        initrd_command=str(item.get("initrd_command", "initrd")),
        initrd_path=str(item.get("initrd_path", NOT_AVAILABLE)),
        loader_prelude=[str(value) for value in item.get("loader_prelude", [])],
        submenu_path=[str(value) for value in item.get("submenu_path", [])],
        source_index=int(item.get("source_index", -1)),
        block_text=str(item.get("block_text", "")),
        is_recovery=bool(item.get("is_recovery", False)),
        schema_version=str(item.get("schema_version", SCHEMA_VERSION)),
    )


def _package_inventory_from_boot_inventory(inventory: dict[str, Any]) -> KernelPackageInventory:
    distribution = inventory.get("distribution", {})
    release = str(distribution.get("VERSION_ID", distribution.get("VERSION_CODENAME", NOT_AVAILABLE)))
    repositories = inventory.get("repository_inventory", [{"name": "configured-official-repositories", "state": "unknown_from_fixture"}])
    filesystem = inventory.get("filesystem_inventory", {})
    return KernelPackageInventory(
        distribution_release=release,
        architecture=str(inventory.get("architecture", NOT_AVAILABLE)),
        configured_repositories=list(repositories),
        installed_kernel_packages=list(inventory.get("kernel_inventory", [])),
        package_manager_state=str(inventory.get("package_manager_state", "unknown_not_modified")),
        boot_free_bytes=filesystem.get("boot_free_bytes", NOT_AVAILABLE),
        root_free_bytes=filesystem.get("root_free_bytes", NOT_AVAILABLE),
    )


def _package_candidate_from_inventory(inventory: dict[str, Any]) -> KernelPackageCandidate:
    kernel = inventory.get("candidate_64k_kernel", {})
    package = str(kernel.get("package_name", NOT_AVAILABLE))
    version = str(kernel.get("package_version", NOT_AVAILABLE))
    arch = str(kernel.get("architecture", NOT_AVAILABLE))
    release = str(kernel.get("release", NOT_AVAILABLE))
    blockers = []
    if package == NOT_AVAILABLE:
        blockers.append("kernel_package_candidate_unknown")
    if arch not in {"arm64", NOT_AVAILABLE}:
        blockers.append("incorrect_architecture")
    if "64k" not in package.lower() and "64k" not in release.lower():
        blockers.append("not_64k_package")
    if "aws" not in package.lower() and "aws" not in release.lower():
        blockers.append("not_aws_kernel_package")
    image_package = package
    return KernelPackageCandidate(
        profile_id="aws-arm64-64k",
        meta_package=_meta_package_from_image(package),
        image_package=image_package,
        modules_packages=[package.replace("linux-image", "linux-modules")] if package.startswith("linux-image") else [],
        header_packages=[],
        candidate_version=version,
        package_architecture=arch,
        package_origins=["ubuntu"],
        trust_source="official_ubuntu_repository_metadata",
        download_size_bytes=kernel.get("download_size_bytes", NOT_AVAILABLE),
        installed_size_bytes=kernel.get("installed_size_bytes", NOT_AVAILABLE),
        expected_release=release,
        expected_page_size="65536",
        qualification_state="metadata_qualified_not_installed" if not blockers else "blocked",
        blockers=blockers,
    )


def _transaction_blockers(package_inventory: KernelPackageInventory, candidate: KernelPackageCandidate, simulation: dict[str, Any]) -> list[str]:
    blockers = list(candidate.blockers)
    if package_inventory.architecture not in {"aarch64", "arm64"}:
        blockers.append("architecture_not_linux_arm64")
    if not simulation.get("provided"):
        blockers.append("apt_simulation_required")
    if simulation.get("unauthenticated"):
        blockers.append("unsigned_or_untrusted_packages")
    if simulation.get("dependency_conflicts"):
        blockers.append("dependency_conflicts")
    if simulation.get("bootloader_replacement"):
        blockers.append("bootloader_replacement")
    for package in simulation.get("removed_packages", []):
        if _is_kernel_package(package):
            blockers.append("kernel_removal_rejected")
        else:
            blockers.append("package_removal_rejected")
    for package in simulation.get("upgraded_packages", []):
        if not _is_kernel_package(package):
            blockers.append("unrelated_upgrade_rejected")
    boot_free = _int_or_none(package_inventory.boot_free_bytes)
    if boot_free is None or boot_free < 512 * 1024 * 1024:
        blockers.append("insufficient_boot_space")
    root_free = _int_or_none(package_inventory.root_free_bytes)
    if root_free is None or root_free < 2 * 1024 * 1024 * 1024:
        blockers.append("insufficient_root_space")
    return sorted(set(blockers), key=blockers.index)


def _exact_install_packages(candidate: KernelPackageCandidate, simulation: dict[str, Any]) -> list[str]:
    packages = [candidate.meta_package, candidate.image_package, *candidate.modules_packages, *candidate.header_packages]
    packages.extend(simulation.get("new_packages", []))
    return sorted({item for item in packages if item and item != NOT_AVAILABLE and _is_kernel_package(item)})


def _installed_kernel_package_names(inventory: KernelPackageInventory) -> list[str]:
    return sorted({str(item.get("package")) for item in inventory.installed_kernel_packages if _is_kernel_package(str(item.get("package")))})


def _meta_package_from_image(package: str) -> str:
    if package == NOT_AVAILABLE:
        return NOT_AVAILABLE
    if re.match(r"linux-image-\d", package):
        return "linux-aws-64k" if "aws" in package and "64k" in package else NOT_AVAILABLE
    return package


def _is_kernel_package(package: str) -> bool:
    lowered = package.lower()
    return lowered.startswith(("linux-image", "linux-modules", "linux-headers", "linux-aws"))


def _menu_entry_details(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = inventory.get("bootloader_evidence", {})
    details = evidence.get("menu_entry_details", [])
    return [dict(item) for item in details if isinstance(item, dict)]


def _resolve_entry(entries: list[dict[str, Any]], value: str) -> dict[str, Any] | None:
    for item in entries:
        if value in {str(item.get("entry_id", "")), str(item.get("title", ""))}:
            return item
    for item in entries:
        if value and value in str(item.get("title", "")):
            return item
    return None


def _kernel_release_from_entry(entry: dict[str, Any] | None) -> str:
    if not entry:
        return NOT_AVAILABLE
    path = str(entry.get("linux_path", ""))
    match = re.search(r"vmlinuz-(.+)$", path)
    return match.group(1) if match else NOT_AVAILABLE


def _qualify_from_inventory_entries(inventory: dict[str, Any], release: str, *, require_initrd: bool) -> dict[str, Any]:
    return qualify_generated_grub_entry(_menu_entry_details(inventory), kernel_release=release, expected_page_size="65536" if "64k" in release else None, require_initrd=require_initrd)


def _profile_additions(profile_id: str) -> list[str]:
    if profile_id in {"original-4k-default", "64k-default"}:
        return []
    if profile_id == "64k-nohz-rcu-irq":
        return ["nohz_full=2-7", "rcu_nocbs=2-7", "irqaffinity=0-1"]
    if profile_id == "64k-managed-irq-isolation":
        return ["nohz_full=2-7", "rcu_nocbs=2-7", "irqaffinity=0-1", "isolcpus=managed_irq,2-7"]
    if profile_id == "64k-domain-isolation-interaction":
        return ["nohz_full=2-7", "rcu_nocbs=2-7", "irqaffinity=0-1", "isolcpus=domain,managed_irq,2-7"]
    return []


def _append_profile_args(command_line: str, additions: list[str]) -> str:
    parts = [item for item in command_line.split() if not item.startswith("BOOT_IMAGE=")]
    existing = {item.split("=", 1)[0] for item in parts}
    parts.extend(item for item in additions if item.split("=", 1)[0] not in existing)
    return " ".join(parts).strip()


def _linux_arguments(command_line: str) -> str:
    return " ".join(item for item in command_line.split() if not item.startswith("BOOT_IMAGE="))


def _identity(identity_id: str, role: str, profile_id: str, page_size: str, cpus: str, threads: int, comparison_use: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_id": identity_id,
        "role": role,
        "profile_id": profile_id,
        "page_size": page_size,
        "workload_cpus": cpus,
        "workload_cpu_count": 8 if cpus == "0-7" else 6,
        "generation_threads": threads,
        "batch_threads": threads,
        "comparison_use": comparison_use,
        "primary_metric": "decode_tokens_per_second",
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
