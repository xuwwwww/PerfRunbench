from __future__ import annotations

import platform
import re
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
    source: str = "sysctl"
    path: str | None = None


@dataclass(frozen=True)
class SettingSnapshot:
    key: str
    value: str | None
    exists: bool
    error: str | None = None
    source: str = "sysctl"
    path: str | None = None


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
    source: str = "sysctl"
    path: str | None = None


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
        RuntimeSetting(
            key="transparent_hugepage.enabled",
            value="madvise",
            reason="Use transparent huge pages only when runtimes explicitly request them, reducing surprise compaction stalls.",
            require_existing=False,
            source="file",
            path="/sys/kernel/mm/transparent_hugepage/enabled",
        ),
        RuntimeSetting(
            key="transparent_hugepage.defrag",
            value="madvise",
            reason="Limit transparent huge page defrag work to madvise regions when the kernel exposes the control.",
            require_existing=False,
            source="file",
            path="/sys/kernel/mm/transparent_hugepage/defrag",
        ),
    ],
    "linux-memory-conservative": [
        RuntimeSetting(
            key="vm.swappiness",
            value="1",
            reason="Avoid swap for memory-bound training until pressure is severe.",
        ),
        RuntimeSetting(
            key="vm.vfs_cache_pressure",
            value="200",
            reason="Reclaim inode/dentry cache more aggressively so training memory has more room.",
        ),
        RuntimeSetting(
            key="vm.page-cluster",
            value="0",
            reason="Reduce swap readahead burst size if swap is used under pressure.",
        ),
        RuntimeSetting(
            key="vm.dirty_background_ratio",
            value="3",
            reason="Start background writeback earlier to avoid large checkpoint-induced dirty page buildup.",
        ),
        RuntimeSetting(
            key="vm.dirty_ratio",
            value="10",
            reason="Cap dirty page accumulation more tightly for memory headroom.",
        ),
        RuntimeSetting(
            key="transparent_hugepage.enabled",
            value="madvise",
            reason="Avoid unconditional transparent huge pages for memory-constrained training runs.",
            require_existing=False,
            source="file",
            path="/sys/kernel/mm/transparent_hugepage/enabled",
        ),
    ],
    "linux-throughput": [
        RuntimeSetting(
            key="vm.swappiness",
            value="10",
            reason="Keep swap pressure low while allowing the kernel some flexibility.",
        ),
        RuntimeSetting(
            key="vm.vfs_cache_pressure",
            value="50",
            reason="Keep filesystem metadata cache warmer for dataset-heavy throughput runs.",
        ),
        RuntimeSetting(
            key="vm.dirty_background_ratio",
            value="10",
            reason="Allow more buffered writeback for throughput-oriented workloads.",
        ),
        RuntimeSetting(
            key="vm.dirty_ratio",
            value="30",
            reason="Allow larger dirty page bursts when throughput is preferred over latency.",
        ),
        RuntimeSetting(
            key="transparent_hugepage.enabled",
            value="madvise",
            reason="Allow runtimes to opt into transparent huge pages without forcing them globally.",
            require_existing=False,
            source="file",
            path="/sys/kernel/mm/transparent_hugepage/enabled",
        ),
    ],
    "linux-low-latency": [
        RuntimeSetting(
            key="vm.swappiness",
            value="1",
            reason="Avoid swap-induced latency spikes during interactive or latency-sensitive runs.",
        ),
        RuntimeSetting(
            key="vm.dirty_background_ratio",
            value="3",
            reason="Begin writeback early to reduce latency spikes from dirty page flushing.",
        ),
        RuntimeSetting(
            key="vm.dirty_ratio",
            value="10",
            reason="Keep maximum dirty pages low for more predictable latency.",
        ),
        RuntimeSetting(
            key="vm.dirty_expire_centisecs",
            value="500",
            reason="Expire dirty pages sooner so writeback work is spread out.",
        ),
        RuntimeSetting(
            key="vm.dirty_writeback_centisecs",
            value="100",
            reason="Run periodic writeback more frequently for smoother IO latency.",
        ),
        RuntimeSetting(
            key="transparent_hugepage.enabled",
            value="never",
            reason="Disable transparent huge pages to avoid allocation and compaction latency spikes.",
            require_existing=False,
            source="file",
            path="/sys/kernel/mm/transparent_hugepage/enabled",
        ),
    ],
    "windows-training-safe": [
        RuntimeSetting(
            key="power.active_scheme",
            value="SCHEME_MIN",
            reason="Use the Windows high performance power scheme during training to reduce CPU frequency scaling noise.",
            source="powercfg",
            path="powercfg://active-scheme",
        ),
    ],
    "windows-memory-conservative": [
        RuntimeSetting(
            key="power.active_scheme",
            value="SCHEME_MIN",
            reason="Windows does not expose Linux-style memory sysctls; keep CPU frequency stable while memory budget enforcement stays in AutoTuneAI.",
            source="powercfg",
            path="powercfg://active-scheme",
        ),
    ],
    "windows-throughput": [
        RuntimeSetting(
            key="power.active_scheme",
            value="SCHEME_MIN",
            reason="Use the Windows high performance power scheme during throughput runs to reduce CPU downclocking.",
            source="powercfg",
            path="powercfg://active-scheme",
        ),
    ],
    "windows-low-latency": [
        RuntimeSetting(
            key="power.active_scheme",
            value="SCHEME_MIN",
            reason="Use the Windows high performance power scheme during latency-sensitive runs to reduce frequency ramp-up delays.",
            source="powercfg",
            path="powercfg://active-scheme",
        ),
    ],
}


def available_profiles() -> list[str]:
    return sorted(PROFILES)


def recommend_system_tuning(profile: str = "linux-training-safe") -> dict[str, Any]:
    settings = _profile_settings(profile)
    current_platform = platform.system()
    supported = _profile_supported_on_platform(profile, current_platform)
    recommendations = []
    for setting in settings:
        snapshot = read_setting(setting)
        recommendations.append(
            {
                "key": setting.key,
                "source": setting.source,
                "path": _setting_location(setting),
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
        "platform": current_platform,
        "supported": supported,
        "apply_supported": supported,
        "settings": recommendations,
        "notes": _plan_notes(profile, current_platform, supported),
    }


def apply_system_tuning(
    profile: str = "linux-training-safe",
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
    runs_dir: Path = RUNS_DIR,
) -> tuple[Path, dict[str, Any]]:
    current_platform = platform.system()
    if not _profile_supported_on_platform(profile, current_platform):
        raise SystemTuningError(f"runtime system tuning profile {profile} is not supported on {current_platform}.")
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
    current_platform = platform.system()
    if not _profile_supported_on_platform(profile, current_platform):
        raise SystemTuningError(f"runtime system tuning profile {profile} is not supported on {current_platform}.")
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
            if not snapshot.exists:
                changes.append(
                    SettingChange(
                        key=setting.key,
                        before=snapshot.value,
                        target=setting.value,
                        after=snapshot.value,
                        changed=False,
                        applied=False,
                        reason=setting.reason,
                        error=(snapshot.error or "setting does not exist") if setting.require_existing else None,
                        source=setting.source,
                        path=_setting_location(setting),
                    )
                )
                continue
            result = runner(_write_command(setting, setting.value, use_sudo=use_sudo))
            if result.returncode != 0:
                status = "failed"
                return_code = result.returncode
                changes.append(
                    SettingChange(
                        key=setting.key,
                        before=snapshot.value,
                        target=setting.value,
                        after=read_setting(setting).value,
                        changed=False,
                        applied=False,
                        reason=setting.reason,
                        error=(result.stderr or result.stdout or "").strip(),
                        source=setting.source,
                        path=_setting_location(setting),
                    )
                )
                break
            after_value = read_setting(setting).value
            changes.append(
                SettingChange(
                    key=setting.key,
                    before=snapshot.value,
                    target=setting.value,
                    after=after_value,
                    changed=snapshot.value != after_value,
                    applied=True,
                    reason=setting.reason,
                    source=setting.source,
                    path=_setting_location(setting),
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
        setting = _setting_from_record(record)
        result = runner(_write_command(setting, value, use_sudo=use_sudo))
        after = read_setting(setting).value
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
    return {setting.key: read_setting(setting) for setting in settings}


def read_setting(setting: RuntimeSetting | str) -> SettingSnapshot:
    if isinstance(setting, str):
        setting = RuntimeSetting(key=setting, value="")
    if setting.source == "powercfg":
        return _read_powercfg_active_scheme(setting)
    path = _setting_path(setting)
    if not path.exists():
        return SettingSnapshot(
            key=setting.key,
            value=None,
            exists=False,
            error=f"{path.as_posix()} does not exist",
            source=setting.source,
            path=path.as_posix(),
        )
    try:
        return SettingSnapshot(
            key=setting.key,
            value=path.read_text(encoding="utf-8").strip(),
            exists=True,
            source=setting.source,
            path=path.as_posix(),
        )
    except OSError as exc:
        return SettingSnapshot(
            key=setting.key,
            value=None,
            exists=True,
            error=str(exc),
            source=setting.source,
            path=path.as_posix(),
        )


def load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _profile_settings(profile: str) -> list[RuntimeSetting]:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise SystemTuningError(f"unknown system tuning profile: {profile}") from exc


def _profile_supported_on_platform(profile: str, current_platform: str) -> bool:
    if profile.startswith("linux-"):
        return current_platform == "Linux"
    if profile.startswith("windows-"):
        return current_platform == "Windows"
    return False


def _sysctl_path(key: str) -> Path:
    return Path("/proc/sys") / Path(*key.split("."))


def _setting_path(setting: RuntimeSetting) -> Path:
    if setting.source == "sysctl":
        return _sysctl_path(setting.key)
    if setting.source == "file":
        if not setting.path:
            raise SystemTuningError(f"file setting {setting.key} is missing a path")
        return Path(setting.path)
    if setting.source == "powercfg":
        raise SystemTuningError("powercfg settings do not map to filesystem paths")
    raise SystemTuningError(f"unsupported runtime setting source: {setting.source}")


def _setting_location(setting: RuntimeSetting) -> str:
    if setting.source == "powercfg":
        return setting.path or "powercfg://active-scheme"
    return _setting_path(setting).as_posix()


def _write_command(setting: RuntimeSetting, value: str, *, use_sudo: bool) -> list[str]:
    if setting.source == "sysctl":
        command = ["sysctl", "-w", f"{setting.key}={value}"]
        if use_sudo:
            return ["sudo", *command]
        return command
    if setting.source == "file":
        path = _setting_path(setting).as_posix()
        command = ["sh", "-c", 'printf "%s\\n" "$1" > "$2"', "sh", value, path]
        if use_sudo:
            return ["sudo", *command]
        return command
    if setting.source == "powercfg":
        return ["powercfg", "/setactive", value]
    raise SystemTuningError(f"unsupported runtime setting source: {setting.source}")


def _setting_from_record(record: dict[str, Any]) -> RuntimeSetting:
    return RuntimeSetting(
        key=record["key"],
        value=str(record.get("value") or ""),
        reason="restore previous runtime setting value",
        source=record.get("source", "sysctl"),
        path=record.get("path"),
    )


def _sysctl_write_command(key: str, value: str, *, use_sudo: bool) -> list[str]:
    command = ["sysctl", "-w", f"{key}={value}"]
    if use_sudo:
        return ["sudo", *command]
    return command


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _read_powercfg_active_scheme(setting: RuntimeSetting) -> SettingSnapshot:
    try:
        result = _run_command(["powercfg", "/getactivescheme"])
    except (FileNotFoundError, OSError) as exc:
        return SettingSnapshot(
            key=setting.key,
            value=None,
            exists=False,
            error=str(exc),
            source=setting.source,
            path=_setting_location(setting),
        )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        return SettingSnapshot(
            key=setting.key,
            value=None,
            exists=False,
            error=output,
            source=setting.source,
            path=_setting_location(setting),
        )
    match = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", output)
    if not match:
        return SettingSnapshot(
            key=setting.key,
            value=None,
            exists=False,
            error=f"could not parse active power scheme from powercfg output: {output}",
            source=setting.source,
            path=_setting_location(setting),
        )
    return SettingSnapshot(
        key=setting.key,
        value=match.group(0),
        exists=True,
        source=setting.source,
        path=_setting_location(setting),
    )


def _snapshots_to_records(snapshots: dict[str, SettingSnapshot]) -> list[dict[str, Any]]:
    return [asdict(snapshot) for snapshot in snapshots.values()]


def _change_to_record(change: SettingChange) -> dict[str, Any]:
    return asdict(change)


def _plan_notes(profile: str, current_platform: str, supported: bool) -> list[str]:
    if not supported:
        return [f"Profile {profile} is not supported on {current_platform}; no system settings will be applied."]
    if profile.startswith("windows-"):
        return [
            "Only reversible Windows runtime settings from an allowlist are considered.",
            "The active power scheme is snapshotted before tuning and restored after the run.",
            "Persistent registry settings are not modified.",
        ]
    return [
        "Only runtime sysctl/sysfs values from an allowlist are considered.",
        "Persistent files such as /etc/sysctl.conf are not modified.",
        "Use restore_run.py with the run id to restore the before snapshot.",
    ]
