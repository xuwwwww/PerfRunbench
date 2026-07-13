from __future__ import annotations

import platform as py_platform
import subprocess
from pathlib import Path

from autotune.arm_tune.platform.architecture import collect_architecture_info
from autotune.arm_tune.platform.cpu_features import merge_features
from autotune.arm_tune.platform.cpu_identity import merge_cpu_identity, parse_cpuinfo_identity, parse_lscpu_identity
from autotune.arm_tune.platform.evidence import no_workload_evidence
from autotune.arm_tune.platform.topology import parse_lscpu_topology
from autotune.arm_tune.schemas import HostInfo, HostInspectionResult, ProductionValidation, UNKNOWN


def collect_host_inspection(*, target_type: str) -> HostInspectionResult:
    cpuinfo_text = _read_text(Path("/proc/cpuinfo"))
    lscpu_text = _command_stdout(["lscpu", "--json"])
    architecture = collect_architecture_info()
    identity = merge_cpu_identity(parse_cpuinfo_identity(cpuinfo_text), parse_lscpu_identity(lscpu_text))
    capabilities = merge_features(cpuinfo_text, lscpu_text)
    topology = parse_lscpu_topology(lscpu_text)
    host = HostInfo(
        architecture=architecture,
        cpu_identity=identity,
        host_capabilities=capabilities,
        topology=topology,
        os_release=_os_release(),
        cgroup_limits=_cgroup_limits(),
    )
    validation = ProductionValidation(
        valid=architecture.is_linux and architecture.is_arm64,
        reason=_production_reason(architecture.is_linux, architecture.is_arm64),
        observed_system=architecture.system,
        observed_machine=architecture.normalized_machine,
    )
    return HostInspectionResult(
        target_type=target_type,
        host=host,
        production_validation=validation,
        workload_execution_evidence=no_workload_evidence(),
    )


def _command_stdout(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _os_release() -> str:
    if py_platform.system().lower() != "linux":
        return UNKNOWN
    text = _read_text(Path("/etc/os-release"))
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return UNKNOWN


def _cgroup_limits() -> dict[str, str]:
    limits: dict[str, str] = {}
    for label, path in {
        "memory.max": Path("/sys/fs/cgroup/memory.max"),
        "cpu.max": Path("/sys/fs/cgroup/cpu.max"),
    }.items():
        value = _read_text(path).strip()
        if value:
            limits[label] = value
    return limits


def _production_reason(is_linux: bool, is_arm64: bool) -> str:
    if is_linux and is_arm64:
        return "target is Linux Arm64"
    if not is_linux and not is_arm64:
        return "production target must be Linux Arm64"
    if not is_linux:
        return "production target must run Linux"
    return "production target machine must be aarch64"
