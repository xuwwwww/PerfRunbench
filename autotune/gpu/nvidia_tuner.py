from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import RUNS_DIR, create_run, finish_run, write_json


class NvidiaTuningError(RuntimeError):
    pass


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

QUERY_FIELDS = [
    "index",
    "name",
    "persistence_mode",
    "power.limit",
    "power.min_limit",
    "power.max_limit",
    "clocks.current.graphics",
    "clocks.current.memory",
]

PROFILES = {
    "nvidia-throughput": {
        "persistence_mode": "1",
        "power_limit": "max",
    },
    "nvidia-safe": {
        "persistence_mode": "1",
        "power_limit": None,
    },
}


def available_nvidia_profiles() -> list[str]:
    return sorted(PROFILES)


def recommend_nvidia_tuning(profile: str = "nvidia-throughput", runner: CommandRunner | None = None) -> dict[str, Any]:
    _profile(profile)
    return {
        "profile": profile,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "supported": shutil.which("nvidia-smi") is not None,
        "gpus": snapshot_nvidia(runner=runner),
        "planned_changes": _planned_changes(profile),
        "notes": [
            "NVIDIA runtime tuning uses nvidia-smi and may require sudo/admin privileges.",
            "Only runtime settings are changed; no driver config files are edited.",
            "Power-limit restore uses the before snapshot captured in the run directory.",
        ],
    }


def apply_nvidia_tuning(
    profile: str = "nvidia-throughput",
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
    runs_dir: Path = RUNS_DIR,
) -> tuple[Path, dict[str, Any]]:
    runner = runner or _run_command
    _require_nvidia_smi()
    run_dir, manifest = create_run(["tune_gpu", "--profile", profile], ResourceBudget(), runs_dir=runs_dir)
    status = "completed"
    return_code = 0
    try:
        result = apply_nvidia_tuning_to_run(run_dir, profile, use_sudo=use_sudo, runner=runner)
        if any(change.get("return_code") not in {0, None} for change in result.get("changes", [])):
            status = "failed"
            return_code = 1
    except Exception:
        status = "failed"
        return_code = 1
        raise
    finally:
        finish_run(run_dir, manifest, status, return_code)
    return run_dir, result


def apply_nvidia_tuning_to_run(
    run_dir: str | Path,
    profile: str = "nvidia-throughput",
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    runner = runner or _run_command
    _require_nvidia_smi()
    run_path = Path(run_dir)
    before = snapshot_nvidia(runner=runner)
    write_json(run_path / "gpu_tuning_plan.json", recommend_nvidia_tuning(profile, runner=runner))
    write_json(run_path / "gpu_tuning_before.json", before)
    changes = _apply_profile(profile, before, use_sudo=use_sudo, runner=runner)
    after = snapshot_nvidia(runner=runner)
    write_json(run_path / "gpu_tuning_after.json", after)
    write_json(run_path / "gpu_tuning_diff.json", changes)
    return {"profile": profile, "changes": changes}


def restore_nvidia_tuning(
    run_dir: str | Path,
    *,
    use_sudo: bool = False,
    runner: CommandRunner | None = None,
) -> list[dict[str, Any]]:
    runner = runner or _run_command
    before_path = Path(run_dir) / "gpu_tuning_before.json"
    if not before_path.exists() or shutil.which("nvidia-smi") is None:
        return []
    import json

    before = json.loads(before_path.read_text(encoding="utf-8"))
    restored = []
    for gpu in before.get("gpus", []):
        index = str(gpu.get("index"))
        persistence = gpu.get("persistence_mode")
        power_limit = gpu.get("power.limit")
        if persistence not in {None, "", "N/A", "[N/A]"}:
            restored.append(_run_change(["nvidia-smi", "-i", index, "-pm", _persistence_value(persistence)], use_sudo, runner))
        if power_limit not in {None, "", "N/A", "[N/A]"}:
            restored.append(_run_change(["nvidia-smi", "-i", index, "-pl", str(power_limit)], use_sudo, runner))
    after = snapshot_nvidia(runner=runner)
    write_json(Path(run_dir) / "gpu_tuning_restore_after.json", {"changes": restored, "after": after})
    return restored


def snapshot_nvidia(runner: CommandRunner | None = None) -> dict[str, Any]:
    runner = runner or _run_command
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "gpus": [], "error": "nvidia-smi was not found on PATH"}
    result = runner(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return {"available": False, "gpus": [], "error": (result.stderr or result.stdout or "").strip()}
    return {"available": True, "gpus": _parse_gpu_rows(result.stdout)}


def _apply_profile(
    profile: str,
    before: dict[str, Any],
    *,
    use_sudo: bool,
    runner: CommandRunner,
) -> list[dict[str, Any]]:
    config = _profile(profile)
    changes = []
    for gpu in before.get("gpus", []):
        index = str(gpu["index"])
        if config.get("persistence_mode") is not None and _setting_available(gpu.get("persistence_mode")):
            changes.append(
                _run_change(
                    ["nvidia-smi", "-i", index, "-pm", str(config["persistence_mode"])],
                    use_sudo,
                    runner,
                    key="persistence_mode",
                    target=str(config["persistence_mode"]),
                    before=gpu.get("persistence_mode"),
                )
            )
        if config.get("power_limit") == "max":
            max_limit = gpu.get("power.max_limit")
            if _setting_available(max_limit):
                changes.append(
                    _run_change(
                        ["nvidia-smi", "-i", index, "-pl", str(max_limit)],
                        use_sudo,
                        runner,
                        key="power.limit",
                        target=str(max_limit),
                        before=gpu.get("power.limit"),
                    )
                )
    return changes


def _run_change(
    command: list[str],
    use_sudo: bool,
    runner: CommandRunner,
    *,
    key: str | None = None,
    target: str | None = None,
    before: str | None = None,
) -> dict[str, Any]:
    actual = ["sudo", *command] if use_sudo else command
    result = runner(actual)
    return {
        "command": actual,
        "key": key,
        "before": before,
        "target": target,
        "return_code": result.returncode,
        "error": (result.stderr or result.stdout or "").strip() if result.returncode != 0 else None,
    }


def _parse_gpu_rows(stdout: str) -> list[dict[str, str]]:
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        rows.append({key: values[index] if index < len(values) else "" for index, key in enumerate(QUERY_FIELDS)})
    return rows


def _planned_changes(profile: str) -> list[dict[str, Any]]:
    config = _profile(profile)
    changes = [{"key": "persistence_mode", "target": config["persistence_mode"]}]
    if config.get("power_limit") == "max":
        changes.append({"key": "power.limit", "target": "power.max_limit"})
    return changes


def _profile(profile: str) -> dict[str, Any]:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise NvidiaTuningError(f"unknown NVIDIA tuning profile: {profile}") from exc


def _persistence_value(value: str) -> str:
    lowered = value.lower()
    if lowered in {"enabled", "1", "on"}:
        return "1"
    if lowered in {"disabled", "0", "off"}:
        return "0"
    return value


def _setting_available(value: str | None) -> bool:
    return value not in {None, "", "N/A", "[N/A]"}


def _require_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        raise NvidiaTuningError("nvidia-smi was not found on PATH")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
