from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Any

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import RUNS_DIR, create_run, finish_run, write_json


class SystemTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSetting:
    key: str
    value: str
    reason: str
    require_existing: bool = True


@dataclass(frozen=True)
class SettingSnapshot:
    key: str
    value: str | None
    exists: bool
    error: str | None = None


@dataclass(frozen=True)
class SettingChange:
    key: str
    before: str | None
    target: str
    after: str | None
    changed: bool
    applied: bool
    reason: str
    error: str | None = None


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


PROFILES: dict[str, list[RuntimeSetting]] = {
    "linux-training-safe": [
        RuntimeSetting(
            key="vm.swappiness",
            value="10",
            reason="Reduce swap pressure so training memory pressure is visible before the system becomes unresponsive.",
        ),
        RuntimeSetting(
            key="kernel.numa_balancing",
            value="0",
            reason="Disable automatic NUMA balancing during a run to reduce runtime migration noise when the setting exists.",
        ),
        RuntimeSetting(
            key="vm.dirty_background_ratio",
            value="5",
            reason="Start background writeback earlier so checkpoint/log writes are less likely to create large flush spikes.",
        ),
        RuntimeSetting(
            key="vm.dirty_ratio",
            value="20",
            reason="Keep the maximum dirty page ratio at a conservative value during heavy training workloads.",
        ),
        RuntimeSetting(
            key="vm.zone_reclaim_mode",
            value="0",
            reason="Avoid local node reclaim stalls on systems that expose zone reclaim controls.",
        ),
    ],
}


def available_profiles() -> list[str]:
    return sorted(PROFILES)


def recommend_system_tuning(profile: str = "linux-training-safe") -> dict[str, Any]:
    settings = _profile_settings(profile)
    supported = platform.system() == "Linux"
    recommendations = []
    for setting in settings:
        snapshot = read_setting(setting.key)
        recommendations.append(
            {
                "key": setting.key,
                "current": snapshot.value,
                "target": setting.value,
                "exists": snapshot.exists,
                "would_change": bool(snapshot.exists and snapshot.value != setting.value),
                "reason": setting.reason,
                "error": snapshot.error,
            }
        )
    return {
        "profile": profile,
        "platform": platform.system(),
        "supported": supported,
        "apply_supported": supported,
        "settings": recommendations,
        "notes": _plan_notes(supported),
    }


def apply_system_tuning(
    profile: str = "linux-training-safe",
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
    runs_dir: Path = RUNS_DIR,
) -> tuple[Path, dict[str, Any]]:
    if platform.system() != "Linux":
        raise SystemTuningError("runtime system tuning is currently implemented only for Linux sysctl settings.")
    runner = runner or _run_command
    run_dir, manifest = create_run(["tune_system", "--profile", profile], ResourceBudget(), runs_dir=runs_dir)
    status = "completed"
    return_code = 0
    try:
        result = apply_system_tuning_to_run(run_dir, manifest, profile, use_sudo=use_sudo, runner=runner)
        if any(change.get("error") for change in result.get("changes", []) if change.get("applied") is False):
            status = "failed"
            return_code = 1
    except Exception:
        status = "failed"
        return_code = 1
        raise
    finally:
        finish_run(run_dir, manifest, status, return_code)
    return run_dir, result


def apply_system_tuning_to_run(
    run_dir: str | Path,
    manifest,
    profile: str = "linux-training-safe",
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise SystemTuningError("runtime system tuning is currently implemented only for Linux sysctl settings.")
    runner = runner or _run_command
    run_path = Path(run_dir)
    settings = _profile_settings(profile)
    before = snapshot_settings(settings)
    write_json(run_path / "system_tuning_plan.json", recommend_system_tuning(profile))
    write_json(run_path / "system_tuning_before.json", _snapshots_to_records(before))
    changes: list[SettingChange] = []
    try:
        for setting in settings:
            snapshot = before[setting.key]
            if setting.require_existing and not snapshot.exists:
                changes.append(
                    SettingChange(
                        key=setting.key,
                        before=snapshot.value,
                        target=setting.value,
                        after=snapshot.value,
                        changed=False,
                        applied=False,
                        reason=setting.reason,
                        error=snapshot.error or "setting does not exist",
                    )
                )
                continue
            result = runner(_sysctl_write_command(setting.key, setting.value, use_sudo=use_sudo))
            if result.returncode != 0:
                status = "failed"
                return_code = result.returncode
                changes.append(
                    SettingChange(
                        key=setting.key,
                        before=snapshot.value,
                        target=setting.value,
                        after=read_setting(setting.key).value,
                        changed=False,
                        applied=False,
                        reason=setting.reason,
                        error=(result.stderr or result.stdout or "").strip(),
                    )
                )
                break
            after_value = read_setting(setting.key).value
            changes.append(
                SettingChange(
                    key=setting.key,
                    before=snapshot.value,
                    target=setting.value,
                    after=after_value,
                    changed=snapshot.value != after_value,
                    applied=True,
                    reason=setting.reason,
                )
            )
    except Exception as exc:
        manifest.notes.append(f"system_tuning_error={exc}")
        raise
    finally:
        after = snapshot_settings(settings)
        write_json(run_path / "system_tuning_after.json", _snapshots_to_records(after))
        diff = [_change_to_record(change) for change in changes]
        write_json(run_path / "system_tuning_diff.json", diff)
        manifest.notes.extend(
            [
                f"system_tuning_profile={profile}",
                f"system_tuning_use_sudo={use_sudo}",
                f"system_tuning_changed={sum(1 for change in changes if change.changed)}",
            ]
        )
    return {"profile": profile, "changes": [_change_to_record(change) for change in changes]}


def restore_system_tuning(run_dir: str | Path, *, use_sudo: bool = False, runner: CommandRunner | None = None) -> list[dict]:
    path = Path(run_dir)
    before_path = path / "system_tuning_before.json"
    if not before_path.exists():
        return []
    runner = runner or _run_command
    before_records = load_json(before_path)
    restored = []
    for record in before_records:
        key = record["key"]
        value = record.get("value")
        if value is None:
            continue
        result = runner(_sysctl_write_command(key, value, use_sudo=use_sudo))
        after = read_setting(key).value
        restored.append(
            {
                "key": key,
                "restored_value": value,
                "after": after,
                "return_code": result.returncode,
                "error": (result.stderr or result.stdout or "").strip() if result.returncode != 0 else None,
            }
        )
    write_json(path / "system_tuning_restore_after.json", restored)
    return restored


def snapshot_settings(settings: list[RuntimeSetting]) -> dict[str, SettingSnapshot]:
    return {setting.key: read_setting(setting.key) for setting in settings}


def read_setting(key: str) -> SettingSnapshot:
    path = _sysctl_path(key)
    if not path.exists():
        return SettingSnapshot(key=key, value=None, exists=False, error=f"{path} does not exist")
    try:
        return SettingSnapshot(key=key, value=path.read_text(encoding="utf-8").strip(), exists=True)
    except OSError as exc:
        return SettingSnapshot(key=key, value=None, exists=True, error=str(exc))


def load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _profile_settings(profile: str) -> list[RuntimeSetting]:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise SystemTuningError(f"unknown system tuning profile: {profile}") from exc


def _sysctl_path(key: str) -> Path:
    return Path("/proc/sys") / Path(*key.split("."))


def _sysctl_write_command(key: str, value: str, *, use_sudo: bool) -> list[str]:
    command = ["sysctl", "-w", f"{key}={value}"]
    if use_sudo:
        return ["sudo", *command]
    return command


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _snapshots_to_records(snapshots: dict[str, SettingSnapshot]) -> list[dict[str, Any]]:
    return [asdict(snapshot) for snapshot in snapshots.values()]


def _change_to_record(change: SettingChange) -> dict[str, Any]:
    return asdict(change)


def _plan_notes(supported: bool) -> list[str]:
    if not supported:
        return ["Runtime sysctl tuning is Linux-only; no system settings will be applied on this platform."]
    return [
        "Only runtime sysctl values from an allowlist are considered.",
        "Persistent files such as /etc/sysctl.conf are not modified.",
        "Use restore_run.py with the run id to restore the before snapshot.",
    ]
