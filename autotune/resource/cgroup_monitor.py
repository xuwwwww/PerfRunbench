from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class CgroupStats:
    timestamp: float
    cgroup_path: str
    memory_current_mb: float | None
    memory_peak_mb: float | None
    cpu_usage_usec: int | None
    cpu_user_usec: int | None
    cpu_system_usec: int | None


def read_systemd_control_group(unit_name: str, timeout_seconds: float = 2.0) -> str | None:
    try:
        result = subprocess.run(
            ["systemctl", "show", unit_name, "-p", "ControlGroup", "--value"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    control_group = result.stdout.strip()
    return control_group or None


def wait_for_systemd_control_group(
    unit_name: str,
    *,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.05,
) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        control_group = read_systemd_control_group(unit_name)
        if control_group:
            return control_group
        time.sleep(poll_interval_seconds)
    return None


def cgroup_path(control_group: str, root: Path = CGROUP_ROOT) -> Path:
    relative = control_group.lstrip("/")
    return root / relative


def read_cgroup_stats(control_group: str, root: Path = CGROUP_ROOT) -> CgroupStats | None:
    path = cgroup_path(control_group, root)
    if not path.exists():
        return None
    cpu_stat = _read_cpu_stat(path / "cpu.stat")
    memory_current = _read_int(path / "memory.current")
    memory_peak = _read_int(path / "memory.peak")
    if memory_peak is None:
        memory_peak = _read_int(path / "memory.max_usage_in_bytes")
    return CgroupStats(
        timestamp=time.time(),
        cgroup_path=str(path),
        memory_current_mb=_bytes_to_mb(memory_current),
        memory_peak_mb=_bytes_to_mb(memory_peak),
        cpu_usage_usec=cpu_stat.get("usage_usec"),
        cpu_user_usec=cpu_stat.get("user_usec"),
        cpu_system_usec=cpu_stat.get("system_usec"),
    )


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cpu_stat(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 * 1024)
