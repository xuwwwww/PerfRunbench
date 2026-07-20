from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.golden_path.config import metrics_section, objective_from_config, workload_section
from autotune.recommendation.fingerprints import collect_environment_fingerprint
from autotune.recommendation.schemas import LaunchSpec, RecommendationError, file_sha256, stable_digest


SENSITIVE_ARG_MARKERS = ("token", "password", "secret", "credential", "private-key", "identity-file", "access-key", "api-key")


def build_workload_identity(command: list[str], config: dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    cwd = _cwd(config)
    executable_path = resolve_executable(command[0], cwd=cwd)
    script = _script_file(command, cwd)
    workload = workload_section(config)
    identity_files = [_file_record(Path(item), cwd) for item in workload.get("identity_files", []) if isinstance(item, str)]
    config_files = [_file_record(Path(item), cwd) for item in workload.get("config_files", []) if isinstance(item, str)]
    if config_path is not None:
        config_files.append(_file_record(Path(config_path), cwd))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "workload_identity",
        "adapter": str(workload.get("adapter") or "generic-command"),
        "normalized_argv": redact_argv(_normalized_argv(command, cwd)),
        "cwd_digest": stable_digest(str(cwd.resolve())),
        "executable": _file_record(executable_path, cwd) if executable_path else {"display": command[0], "sha256": "unknown"},
        "script": _file_record(script, cwd) if script else None,
        "identity_files": identity_files,
        "config_files": config_files,
        "objective": objective_from_config(config),
        "metric_protocol": _metric_protocol(config),
    }
    payload["digest"] = stable_digest(payload)
    return payload


def environment_fingerprint_for_command(command: list[str], config: dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    cwd = _cwd(config)
    executable_path = resolve_executable(command[0], cwd=cwd)
    workload = workload_section(config)
    return collect_environment_fingerprint(
        executable_path=executable_path,
        workload_config_path=config_path,
        runtime_identity=str(workload.get("adapter") or "generic-command"),
        runtime_version=str(workload.get("runtime_version") or platform.python_version()),
        allowed_cpu_mask=workload.get("allowed_cpu_mask") if isinstance(workload.get("allowed_cpu_mask"), str) else None,
    )


def launch_spec_for_command(
    command: list[str],
    config: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    cwd = _cwd(config)
    workload = workload_section(config)
    apply = config.get("apply", {}) if isinstance(config.get("apply"), dict) else {}
    timeout = float(workload.get("timeout_seconds") or apply.get("timeout_seconds") or 300.0)
    resolved = resolved_argv(command, cwd=cwd)
    metric = objective_from_config(config)["metric"]
    validators = apply.get("correctness_validators")
    if validators is None:
        validators = [
            {"type": "exit_code", "expected": 0},
            {"type": "required_metric", "metric": metric},
            {"type": "metric_finite", "metric": metric},
            {"type": "cleanup_passed"},
        ]
    launch = LaunchSpec(
        argv=resolved,
        adapter=str(workload.get("adapter") or "generic-command"),
        cwd=str(cwd),
        env={str(key): str(value) for key, value in (env or {}).items()},
        timeout_seconds=timeout,
        metadata={
            "runtime_identity": str(workload.get("adapter") or "generic-command"),
            "runtime_version": str(workload.get("runtime_version") or platform.python_version()),
            "workload_config_path": str(config_path) if config_path is not None else None,
            "metrics_output_path": apply.get("metrics_output_path") if isinstance(apply.get("metrics_output_path"), str) else None,
        },
        correctness_contract={"validators": validators},
    )
    return launch.to_dict()


def validate_env_overlay(env: dict[str, str]) -> None:
    LaunchSpec(argv=["noop"], env=env)


def resolved_argv(command: list[str], *, cwd: Path | None = None) -> list[str]:
    cwd = cwd or Path.cwd()
    first = resolve_executable(command[0], cwd=cwd)
    argv = [str(first) if first else command[0]]
    argv.extend(str(item) for item in command[1:])
    return argv


def resolve_executable(value: str, *, cwd: Path | None = None) -> Path | None:
    raw = Path(value)
    if raw.is_absolute() and raw.exists() and raw.is_file():
        return raw.resolve()
    cwd = cwd or Path.cwd()
    candidate = (cwd / raw)
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(value)
    return Path(resolved).resolve() if resolved else None


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in argv:
        lowered = item.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if any(marker in lowered for marker in SENSITIVE_ARG_MARKERS):
            if "=" in item:
                redacted.append(f"{item.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(item)
                redact_next = item.startswith("-")
            continue
        redacted.append(_sanitize_path_arg(item))
    return redacted


def _metric_protocol(config: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_section(config)
    return {
        "source": metrics.get("source") or "json_file",
        "path_reference": metrics.get("path") if isinstance(metrics.get("path"), str) else None,
        "prefix": metrics.get("prefix") if isinstance(metrics.get("prefix"), str) else None,
        "objective_metric": objective_from_config(config)["metric"],
        "objective_direction": objective_from_config(config)["direction"],
    }


def _normalized_argv(command: list[str], cwd: Path) -> list[str]:
    return [str(resolve_executable(command[0], cwd=cwd) or command[0]), *[str(item) for item in command[1:]]]


def _script_file(command: list[str], cwd: Path) -> Path | None:
    for item in command[1:]:
        if item.startswith("-"):
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _cwd(config: dict[str, Any]) -> Path:
    workload = workload_section(config)
    cwd = workload.get("cwd")
    return Path(str(cwd)).resolve() if isinstance(cwd, str) and cwd else Path.cwd().resolve()


def _file_record(path: Path, cwd: Path) -> dict[str, str]:
    target = path if path.is_absolute() else cwd / path
    try:
        resolved = target.resolve()
        digest = file_sha256(resolved) if resolved.exists() and resolved.is_file() else "missing"
    except OSError as exc:
        raise RecommendationError(f"failed to hash identity file: {path.name}") from exc
    return {"display": _display_path(resolved, cwd), "sha256": digest}


def _display_path(path: Path, cwd: Path) -> str:
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return path.name


def _sanitize_path_arg(item: str) -> str:
    path = Path(item)
    if path.is_absolute():
        return path.name
    if os.sep in item or "/" in item:
        return Path(item).as_posix()
    return item
