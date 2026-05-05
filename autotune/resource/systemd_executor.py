from __future__ import annotations

import getpass
import re
import shutil
import subprocess
from dataclasses import dataclass

from autotune.resource.budget import ResourceBudget


@dataclass(frozen=True)
class SystemdCommand:
    command: list[str]
    notes: list[str]


def make_systemd_scope_name(run_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-")
    if not safe:
        safe = "run"
    if not safe.startswith("autotuneai-"):
        safe = f"autotuneai-{safe}"
    if not safe.endswith(".scope"):
        safe = f"{safe}.scope"
    return safe


@dataclass(frozen=True)
class SystemdPreflight:
    runnable: bool
    use_sudo: bool
    user: str
    systemd_run_path: str | None
    sudo_path: str | None
    systemd_state: str | None
    sudo_cached: bool | None
    errors: list[str]
    warnings: list[str]
    notes: list[str]
    command_preview: list[str] | None
    probe_succeeded: bool | None = None
    probe_output: str | None = None

    def to_record(self) -> dict:
        return {
            "runnable": self.runnable,
            "use_sudo": self.use_sudo,
            "user": self.user,
            "systemd_run_path": self.systemd_run_path,
            "sudo_path": self.sudo_path,
            "systemd_state": self.systemd_state,
            "sudo_cached": self.sudo_cached,
            "errors": self.errors,
            "warnings": self.warnings,
            "notes": self.notes,
            "command_preview": self.command_preview,
            "probe_succeeded": self.probe_succeeded,
            "probe_output": self.probe_output,
        }


def systemd_available() -> bool:
    return shutil.which("systemd-run") is not None


def read_systemd_state(timeout_seconds: float = 3.0) -> str | None:
    if shutil.which("systemctl") is None:
        return None
    try:
        result = subprocess.run(
            ["systemctl", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    state = (result.stdout or result.stderr).strip()
    return state or None


def sudo_credential_cached(timeout_seconds: float = 3.0) -> bool:
    if shutil.which("sudo") is None:
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def probe_systemd_scope(
    budget: ResourceBudget,
    *,
    use_sudo: bool = False,
    run_as_user: str | None = None,
    unit_name: str | None = None,
    timeout_seconds: float = 10.0,
) -> tuple[bool, str]:
    command = build_systemd_run_command(
        ["true"],
        budget,
        use_sudo=use_sudo,
        run_as_user=run_as_user,
        unit_name=unit_name,
    ).command
    if use_sudo and command and command[0] == "sudo":
        command.insert(1, "-n")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def preflight_systemd_executor(
    command: list[str],
    budget: ResourceBudget,
    *,
    use_sudo: bool = False,
    run_as_user: str | None = None,
    check_sudo_cache: bool = False,
    probe: bool = False,
    unit_name: str | None = None,
) -> SystemdPreflight:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    user = run_as_user or getpass.getuser()
    systemd_run_path = shutil.which("systemd-run")
    sudo_path = shutil.which("sudo")
    systemd_state = read_systemd_state()
    sudo_cached: bool | None = None
    command_preview: list[str] | None = None
    probe_succeeded: bool | None = None
    probe_output: str | None = None

    if systemd_run_path is None:
        errors.append("systemd-run was not found on PATH.")
    if systemd_state not in {None, "running", "degraded"}:
        warnings.append(f"systemd state is {systemd_state}; transient scopes may fail.")
    if use_sudo and sudo_path is None:
        errors.append("sudo was requested but sudo was not found on PATH.")
    if use_sudo and check_sudo_cache and sudo_path is not None:
        sudo_cached = sudo_credential_cached()
        if not sudo_cached:
            warnings.append("sudo credential is not cached; run sudo -v first or expect an interactive password prompt.")

    if not errors:
        try:
            command_preview = build_systemd_run_command(
                command,
                budget,
                use_sudo=use_sudo,
                run_as_user=user,
                unit_name=unit_name,
            ).command
        except RuntimeError as exc:
            errors.append(str(exc))
    if probe and not errors:
        probe_succeeded, probe_output = probe_systemd_scope(
            budget,
            use_sudo=use_sudo,
            run_as_user=user,
            unit_name=unit_name,
        )
        if not probe_succeeded:
            errors.append(f"systemd scope probe failed: {probe_output}")

    if budget.enforce:
        memory_budget = budget.memory_budget_mb
        if memory_budget is not None:
            notes.append(f"MemoryMax would be set to {int(memory_budget)}M.")
        if budget.cpu_quota_percent is not None:
            notes.append(f"CPUQuota would be set to {budget.cpu_quota_percent}%.")
    elif budget.enabled:
        notes.append("Resource budget values were supplied, but enforcement is disabled.")
    if use_sudo:
        notes.append(f"sudo mode will request the systemd scope as root and run the workload as user {user}.")

    return SystemdPreflight(
        runnable=not errors,
        use_sudo=use_sudo,
        user=user,
        systemd_run_path=systemd_run_path,
        sudo_path=sudo_path,
        systemd_state=systemd_state,
        sudo_cached=sudo_cached,
        errors=errors,
        warnings=warnings,
        notes=notes,
        command_preview=command_preview,
        probe_succeeded=probe_succeeded,
        probe_output=probe_output,
    )


def build_systemd_run_command(
    command: list[str],
    budget: ResourceBudget,
    *,
    use_sudo: bool = False,
    run_as_user: str | None = None,
    unit_name: str | None = None,
    environment: dict[str, str] | None = None,
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
    if unit_name:
        wrapped.extend(["--unit", unit_name])
        notes.append(f"systemd unit={unit_name}")

    for key, value in sorted((environment or {}).items()):
        wrapped.extend(["--setenv", f"{key}={value}"])
        notes.append(f"systemd environment {key}=set")

    if use_sudo:
        user = run_as_user or getpass.getuser()
        wrapped.extend(["--uid", user])
        notes.append(f"systemd-run will be invoked through sudo and run workload as user {user}.")

    if not budget.enforce:
        notes.append("systemd resource limits disabled because resource budget enforcement is false.")
    memory_budget = budget.effective_memory_budget_mb(_visible_memory_mb()) if budget.enforce and budget.enabled else None
    if memory_budget is not None:
        wrapped.extend(["-p", f"MemoryMax={int(memory_budget)}M"])
        notes.append(f"systemd MemoryMax={int(memory_budget)}M")

    if budget.enforce and budget.cpu_quota_percent is not None:
        wrapped.extend(["-p", f"CPUQuota={budget.cpu_quota_percent}%"])
        notes.append(f"systemd CPUQuota={budget.cpu_quota_percent}%")

    wrapped.extend(["--", *command])
    return SystemdCommand(command=wrapped, notes=notes)


def _visible_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return None
