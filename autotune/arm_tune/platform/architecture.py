from __future__ import annotations

import platform as py_platform

from autotune.arm_tune.schemas import ArchitectureInfo, UNKNOWN


ARM64_ALIASES = {"aarch64", "arm64"}


def normalize_machine(machine: str | None) -> str:
    text = (machine or "").strip().lower()
    if not text:
        return UNKNOWN
    if text in ARM64_ALIASES:
        return "aarch64"
    return text


def collect_architecture_info(
    *,
    system: str | None = None,
    machine: str | None = None,
    kernel_release: str | None = None,
) -> ArchitectureInfo:
    observed_system = system if system is not None else py_platform.system()
    observed_machine = machine if machine is not None else py_platform.machine()
    observed_release = kernel_release if kernel_release is not None else py_platform.release()
    normalized = normalize_machine(observed_machine)
    return ArchitectureInfo(
        system=observed_system or UNKNOWN,
        machine=observed_machine or UNKNOWN,
        normalized_machine=normalized,
        kernel_release=observed_release or UNKNOWN,
        is_linux=(observed_system or "").lower() == "linux",
        is_arm64=normalized == "aarch64",
    )
