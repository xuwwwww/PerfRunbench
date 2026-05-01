from __future__ import annotations

import getpass
import shutil
from dataclasses import dataclass

from autotune.resource.budget import ResourceBudget


@dataclass(frozen=True)
class SystemdCommand:
    command: list[str]
    notes: list[str]


def systemd_available() -> bool:
    return shutil.which("systemd-run") is not None


def build_systemd_run_command(
    command: list[str],
    budget: ResourceBudget,
    *,
    use_sudo: bool = False,
    run_as_user: str | None = None,
) -> SystemdCommand:
    if not command:
        raise ValueError("command cannot be empty")
    if not systemd_available():
        raise RuntimeError("systemd-run was not found on PATH")

    notes: list[str] = []
    wrapped: list[str] = []
    if use_sudo:
        if shutil.which("sudo") is None:
            raise RuntimeError("sudo was requested but was not found on PATH")
        wrapped.append("sudo")

    wrapped.extend(["systemd-run", "--scope", "--quiet"])

    if use_sudo:
        user = run_as_user or getpass.getuser()
        wrapped.extend(["--uid", user])
        notes.append(f"systemd-run will be invoked through sudo and run workload as user {user}.")

    memory_budget = budget.memory_budget_mb
    if memory_budget is not None:
        wrapped.extend(["-p", f"MemoryMax={int(memory_budget)}M"])
        notes.append(f"systemd MemoryMax={int(memory_budget)}M")

    if budget.cpu_quota_percent is not None:
        wrapped.extend(["-p", f"CPUQuota={budget.cpu_quota_percent}%"])
        notes.append(f"systemd CPUQuota={budget.cpu_quota_percent}%")

    wrapped.extend(["--", *command])
    return SystemdCommand(command=wrapped, notes=notes)
