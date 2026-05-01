from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from typing import Any

from autotune.profiler.hardware_info import is_wsl
from autotune.resource.budget import ResourceBudget
from autotune.resource.systemd_executor import probe_systemd_scope, read_systemd_state, sudo_credential_cached


def collect_executor_capabilities(
    *,
    probe_docker: bool = False,
    probe_systemd: bool = False,
    check_sudo_cache: bool = False,
) -> dict[str, Any]:
    system = platform.system()
    executors = {
        "local": _local_capability(),
        "systemd": _systemd_capability(
            system,
            check_sudo_cache=check_sudo_cache,
            probe=probe_systemd,
        ),
        "docker": _docker_capability(probe=probe_docker),
        "windows_job": _windows_job_capability(system),
        "macos": _macos_capability(system),
    }
    return {
        "platform": _platform_key(system),
        "system": system,
        "is_wsl": is_wsl(),
        "recommended_executor": recommend_executor(executors),
        "executors": executors,
    }


def recommend_executor(executors: dict[str, dict[str, Any]]) -> str:
    systemd = executors.get("systemd", {})
    if systemd.get("available") and systemd.get("hard_memory_limit"):
        return "systemd"
    docker = executors.get("docker", {})
    if docker.get("available") and docker.get("hard_memory_limit") and docker.get("implemented"):
        return "docker"
    return "local"


def _local_capability() -> dict[str, Any]:
    psutil_available = importlib.util.find_spec("psutil") is not None
    return {
        "available": True,
        "implemented": True,
        "requires_root": False,
        "requires_sudo": False,
        "hard_memory_limit": False,
        "hard_cpu_limit": False,
        "process_tree_monitoring": psutil_available,
        "cgroup_monitoring": False,
        "notes": [
            "Cross-platform fallback executor.",
            "Uses psutil process-tree monitoring when psutil is installed.",
            "Memory limits are soft unless hard-kill is enabled.",
        ],
    }


def _systemd_capability(system: str, *, check_sudo_cache: bool, probe: bool) -> dict[str, Any]:
    systemd_run = shutil.which("systemd-run")
    systemctl = shutil.which("systemctl")
    sudo = shutil.which("sudo")
    state = read_systemd_state() if system == "Linux" else None
    cgroup_root_exists = system == "Linux" and _path_exists("/sys/fs/cgroup")
    available = bool(system == "Linux" and systemd_run and state in {"running", "degraded"})
    sudo_cached = sudo_credential_cached() if check_sudo_cache and sudo else None
    transient_scope_without_sudo = None
    probe_output = None
    notes = []
    if system != "Linux":
        notes.append("systemd executor is Linux-only.")
    if system == "Linux" and not systemd_run:
        notes.append("systemd-run was not found on PATH.")
    if system == "Linux" and state not in {"running", "degraded"}:
        notes.append(f"systemd state is {state}; transient scopes may not be usable.")
    if available:
        notes.append("Supports systemd transient scopes with cgroup memory and CPU accounting.")
        notes.append("May require sudo or polkit permission to create transient scopes.")
    if available and probe:
        transient_scope_without_sudo, probe_output = probe_systemd_scope(ResourceBudget())
        if not transient_scope_without_sudo and _looks_like_interactive_auth_required(probe_output):
            notes.append("Non-sudo transient scope probe requires interactive authentication; use --sudo.")
    return {
        "available": available,
        "implemented": True,
        "requires_root": False,
        "requires_sudo": False if transient_scope_without_sudo else True if transient_scope_without_sudo is False else None,
        "sudo_available": sudo is not None,
        "sudo_cached": sudo_cached,
        "transient_scope_without_sudo": transient_scope_without_sudo,
        "probe_output": probe_output,
        "hard_memory_limit": available,
        "hard_cpu_limit": available,
        "process_tree_monitoring": True,
        "cgroup_monitoring": bool(available and cgroup_root_exists and systemctl),
        "systemd_run_path": systemd_run,
        "systemctl_path": systemctl,
        "systemd_state": state,
        "cgroup_root_exists": cgroup_root_exists,
        "notes": notes,
    }


def _docker_capability(*, probe: bool) -> dict[str, Any]:
    docker = shutil.which("docker")
    daemon_available = None
    if probe and docker:
        daemon_available = _docker_daemon_available()
    implemented = True
    available = bool(docker and (daemon_available is not False))
    notes = []
    if not docker:
        notes.append("docker CLI was not found on PATH.")
    elif daemon_available is False:
        notes.append("docker CLI exists, but the daemon is not reachable.")
    else:
        notes.append("Docker can provide cross-platform hard memory and CPU limits through containers.")
    return {
        "available": available,
        "implemented": implemented,
        "requires_root": False,
        "requires_sudo": False,
        "docker_path": docker,
        "docker_daemon_available": daemon_available,
        "hard_memory_limit": available,
        "hard_cpu_limit": available,
        "process_tree_monitoring": False,
        "cgroup_monitoring": False,
        "notes": notes,
    }


def _windows_job_capability(system: str) -> dict[str, Any]:
    platform_supported = system == "Windows"
    return {
        "available": False,
        "implemented": False,
        "platform_supported": platform_supported,
        "requires_root": False,
        "requires_sudo": False,
        "hard_memory_limit": platform_supported,
        "hard_cpu_limit": platform_supported,
        "process_tree_monitoring": platform_supported,
        "cgroup_monitoring": False,
        "notes": [
            "Windows Job Object support is planned, but not implemented yet."
            if platform_supported
            else "Windows Job Object executor is Windows-only."
        ],
    }


def _macos_capability(system: str) -> dict[str, Any]:
    platform_supported = system == "Darwin"
    return {
        "available": False,
        "implemented": False,
        "platform_supported": platform_supported,
        "requires_root": False,
        "requires_sudo": False,
        "hard_memory_limit": False,
        "hard_cpu_limit": False,
        "process_tree_monitoring": platform_supported,
        "cgroup_monitoring": False,
        "notes": [
            "macOS support is currently local monitoring only; hard memory limits need a different backend."
            if platform_supported
            else "macOS executor is Darwin-only."
        ],
    }


def _platform_key(system: str) -> str:
    if system == "Linux" and is_wsl():
        return "linux-wsl"
    if system == "Linux":
        return "linux"
    if system == "Windows":
        return "windows"
    if system == "Darwin":
        return "macos"
    return system.lower() or "unknown"


def _docker_daemon_available(timeout_seconds: float = 2.0) -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _looks_like_interactive_auth_required(output: str | None) -> bool:
    if not output:
        return False
    lowered = output.lower()
    return "interactive authentication required" in lowered or "authentication required" in lowered


def _path_exists(path: str) -> bool:
    try:
        from pathlib import Path

        return Path(path).exists()
    except OSError:
        return False
