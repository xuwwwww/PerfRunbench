from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from autotune.resource.budget import ResourceBudget


@dataclass(frozen=True)
class DockerCommand:
    command: list[str]
    notes: list[str]


def build_docker_run_command(
    command: list[str],
    budget: ResourceBudget,
    *,
    image: str = "python:3.12-slim",
    workdir: str | Path | None = None,
    total_cores: int | None = None,
    total_memory_mb: float | None = None,
) -> DockerCommand:
    if not command:
        raise ValueError("command cannot be empty")
    if not image:
        raise ValueError("docker image cannot be empty")
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI was not found on PATH")
    host_workdir = Path(workdir or Path.cwd()).resolve()
    container_workdir = "/workspace"
    total_cores = total_cores or os.cpu_count()
    memory_budget = budget.effective_memory_budget_mb(total_memory_mb if total_memory_mb is not None else _visible_memory_mb())
    allowed_threads = budget.allowed_threads(total_cores)
    cpu_limit = _docker_cpu_limit(total_cores, allowed_threads, budget.cpu_quota_percent)
    normalized_command = _normalize_python_command(command)

    wrapped = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{host_workdir}:{container_workdir}",
        "-w",
        container_workdir,
    ]
    notes = [f"docker image={image}", f"docker workdir={host_workdir} mounted at {container_workdir}"]
    if memory_budget is not None and memory_budget > 0:
        memory_mb = int(memory_budget)
        wrapped.extend(["--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m"])
        notes.append(f"docker memory={memory_mb}m")
    if cpu_limit is not None:
        wrapped.extend(["--cpus", _format_cpus(cpu_limit)])
        notes.append(f"docker cpus={_format_cpus(cpu_limit)}")
    wrapped.extend([image, *normalized_command])
    notes.append("Docker executor enforces limits inside the container; host process sampling observes the docker client.")
    return DockerCommand(command=wrapped, notes=notes)


def _docker_cpu_limit(total_cores: int | None, allowed_threads: int | None, quota_percent: float | None) -> float | None:
    if not total_cores:
        return None
    limit = float(total_cores)
    if allowed_threads is not None:
        limit = min(limit, float(allowed_threads))
    if quota_percent is not None:
        limit = min(limit, max(0.01, total_cores * (quota_percent / 100.0)))
    return max(0.01, limit)


def _format_cpus(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "1"


def _normalize_python_command(command: list[str]) -> list[str]:
    executable = Path(command[0]).name.lower()
    if not Path(command[0]).is_absolute() or not executable.startswith("python"):
        return command
    if command[0] != sys.executable and "python" not in executable:
        return command
    return ["python3" if executable.startswith("python3") else "python", *command[1:]]


def _visible_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return None
