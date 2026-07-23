from __future__ import annotations

from pathlib import Path

from autotune.system_profiles.schemas import BootKernelProfile, HostStateSnapshot, NOT_AVAILABLE, SystemProfileError
from autotune.system_profiles.snapshot import RootReader


BOOT_PROFILE_IDS = [
    "aws-4k-default",
    "aws-4k-nohz-rcu-irq",
    "aws-4k-managed-irq-isolation",
    "aws-4k-domain-isolation",
    "aws-64k-default",
    "aws-64k-nohz-rcu-irq",
]


def build_boot_profile(
    profile_id: str,
    snapshot: HostStateSnapshot,
    *,
    root: str | Path | None = None,
    allow_boot_change: bool = False,
    user_snapshot_marker: str | None = None,
) -> BootKernelProfile:
    if profile_id not in BOOT_PROFILE_IDS:
        raise SystemProfileError(f"unknown boot profile: {profile_id}")
    reader = RootReader(root)
    available = snapshot.security_recovery.get("available_kernels", NOT_AVAILABLE)
    fallback = _fallback_kernel(snapshot)
    marker = user_snapshot_marker or snapshot.snapshot_marker
    support = _kernel_config_support(reader, snapshot)
    current_cmdline = snapshot.identity.get("kernel_command_line", "")
    if profile_id == "aws-4k-default":
        return _boot(profile_id, "standard Ubuntu AWS 4K kernel without extra isolation", "aws-4k", str(current_cmdline), "supported", fallback=fallback)
    if profile_id == "aws-64k-default":
        state = "requires_reboot" if _has_64k_kernel(available) else "unsupported"
        reason = "" if state != "unsupported" else "no official 64K kernel flavor was found in the snapshot"
        return _boot(profile_id, "official Ubuntu AWS ARM64 64K kernel without extra isolation", "aws-64k", str(current_cmdline), state, fallback=fallback, reason=reason)
    if "nohz-rcu-irq" in profile_id:
        flavor = "aws-64k" if profile_id.startswith("aws-64k") else "aws-4k"
        missing = [key for key, ok in {"CONFIG_NO_HZ_FULL": support["nohz_full"], "CONFIG_RCU_NOCB_CPU": support["rcu_nocb"]}.items() if not ok]
        cmdline = _append_args(str(current_cmdline), ["nohz_full=2-7", "rcu_nocbs=2-7", "irqaffinity=0-1"])
        return _authorized_boot(profile_id, f"{flavor} one-time nohz_full, RCU callback, and IRQ-affinity isolation plan", flavor, cmdline, fallback, marker, allow_boot_change, missing)
    if profile_id == "aws-4k-managed-irq-isolation":
        missing = [] if support["isolcpus_managed_irq"] else ["isolcpus=managed_irq support"]
        cmdline = _append_args(str(current_cmdline), ["isolcpus=managed_irq,2-7"])
        return _authorized_boot(profile_id, "4K AWS one-time managed IRQ isolation plan", "aws-4k", cmdline, fallback, marker, allow_boot_change, missing)
    if profile_id == "aws-4k-domain-isolation":
        missing = [key for key, ok in {"CONFIG_NO_HZ_FULL": support["nohz_full"], "CONFIG_RCU_NOCB_CPU": support["rcu_nocb"], "isolcpus=managed_irq": support["isolcpus_managed_irq"]}.items() if not ok]
        cmdline = _append_args(str(current_cmdline), ["isolcpus=domain,managed_irq,2-7", "nohz_full=2-7", "rcu_nocbs=2-7", "irqaffinity=0-1"])
        result = _authorized_boot(profile_id, "advanced less-flexible domain and managed IRQ boot isolation plan", "aws-4k", cmdline, fallback, marker, allow_boot_change, missing)
        return BootKernelProfile(**{**result.to_dict(), "less_flexible_boot_isolation": True})
    raise SystemProfileError(f"boot profile not implemented: {profile_id}")


def list_boot_profiles(snapshot: HostStateSnapshot, *, root: str | Path | None = None) -> list[dict[str, object]]:
    return [build_boot_profile(profile_id, snapshot, root=root).to_dict() for profile_id in BOOT_PROFILE_IDS]


def _authorized_boot(
    profile_id: str,
    mechanism: str,
    flavor: str,
    cmdline: str,
    fallback: str,
    marker: str,
    allow_boot_change: bool,
    missing: list[str],
) -> BootKernelProfile:
    if missing:
        return _boot(profile_id, mechanism, flavor, cmdline, "unsupported", fallback=fallback, reason=f"missing support: {', '.join(missing)}")
    if fallback == NOT_AVAILABLE:
        return _boot(profile_id, mechanism, flavor, cmdline, "requires_user_snapshot", fallback=fallback, reason="fallback kernel is not available")
    if marker in {"", NOT_AVAILABLE, None}:
        return _boot(profile_id, mechanism, flavor, cmdline, "requires_user_snapshot", fallback=fallback, reason="user snapshot marker is required before boot changes")
    if not allow_boot_change:
        return _boot(profile_id, mechanism, flavor, cmdline, "requires_user_snapshot", fallback=fallback, reason="--allow-boot-change is required for a real boot change")
    return _boot(profile_id, mechanism, flavor, cmdline, "requires_reboot", fallback=fallback)


def _boot(profile_id: str, mechanism: str, flavor: str, cmdline: str, state: str, *, fallback: str, reason: str = "") -> BootKernelProfile:
    return BootKernelProfile(
        profile_id=profile_id,
        mechanism=mechanism,
        kernel_flavor=flavor,
        generated_command_line=cmdline,
        reboot_requirement=state in {"requires_reboot", "requires_user_snapshot"},
        required_capabilities=["kernel inventory", "kernel config", "fallback kernel", "SSH health check", "user snapshot marker"],
        verification_checks=["one-time boot", "post-reboot kernel flavor", "command-line digest", "SSH availability", "workload model availability"],
        rollback_plan=["boot fallback kernel once", "restore previous command line", "verify original kernel and command-line digest"],
        eligibility_state=state,
        fallback_kernel=fallback,
        state_reason=reason,
    )


def _kernel_config_support(reader: RootReader, snapshot: HostStateSnapshot) -> dict[str, bool]:
    release = snapshot.identity.get("kernel_release", "")
    candidates = [reader.path(f"/boot/config-{release}"), reader.path("/run/perfrunbench-fixtures/kernel_config")]
    text = ""
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            break
    return {
        "nohz_full": "CONFIG_NO_HZ_FULL=y" in text,
        "rcu_nocb": "CONFIG_RCU_NOCB_CPU=y" in text or "CONFIG_RCU_NOCB_CPU_DEFAULT_ALL=y" in text,
        "isolcpus_managed_irq": True if text else False,
    }


def _fallback_kernel(snapshot: HostStateSnapshot) -> str:
    kernels = snapshot.security_recovery.get("available_kernels", NOT_AVAILABLE)
    current = snapshot.identity.get("kernel_release", NOT_AVAILABLE)
    if not isinstance(kernels, list):
        return NOT_AVAILABLE
    for kernel in kernels:
        if kernel != current:
            return str(kernel)
    return NOT_AVAILABLE


def _has_64k_kernel(kernels: object) -> bool:
    return isinstance(kernels, list) and any("64k" in str(item).lower() for item in kernels)


def _append_args(current: str, args: list[str]) -> str:
    parts = [item for item in current.split() if not any(item.startswith(arg.split("=", 1)[0] + "=") for arg in args)]
    parts.extend(args)
    return " ".join(parts).strip()
