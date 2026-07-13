from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import TargetSpec
from autotune.utils.config import load_config


TOP_LEVEL_KEYS = {"schema_version", "target"}
TARGET_KEYS = {
    "type",
    "host",
    "user",
    "port",
    "identity_file",
    "remote_workdir",
    "remote_python",
    "remote_module",
    "ssh_executable",
    "scp_executable",
    "connect_timeout_seconds",
    "command_timeout_seconds",
    "keep_remote_artifacts",
    "strict_host_key_checking",
}
PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def load_target_spec(path: str | Path) -> TargetSpec:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("target config must be an object")
    return parse_target_spec(payload)


def parse_target_spec(payload: dict[str, Any]) -> TargetSpec:
    if "target" in payload:
        unknown = set(payload) - TOP_LEVEL_KEYS
        if unknown:
            raise ArmTuneConfigError(f"unknown top-level field(s): {sorted(unknown)}")
        schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
        raw_target = payload.get("target")
        if not isinstance(raw_target, dict):
            raise ArmTuneConfigError("target must be an object")
    else:
        schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
        raw_target = {key: value for key, value in payload.items() if key != "schema_version"}
    if schema_version != SCHEMA_VERSION:
        raise ArmTuneConfigError(f"unsupported schema_version: {schema_version!r}")
    unknown_target = set(raw_target) - TARGET_KEYS
    if unknown_target:
        raise ArmTuneConfigError(f"unknown target field(s): {sorted(unknown_target)}")
    target_type = str(raw_target.get("type", "")).strip().lower()
    if target_type not in {"local", "ssh"}:
        raise ArmTuneConfigError("target.type must be 'local' or 'ssh'")
    if target_type == "ssh" and not raw_target.get("host"):
        raise ArmTuneConfigError("ssh target requires target.host")
    port = _int_field(raw_target.get("port", 22), "target.port")
    if port <= 0 or port > 65535:
        raise ArmTuneConfigError("target.port must be between 1 and 65535")
    connect_timeout = _int_field(raw_target.get("connect_timeout_seconds", 15), "target.connect_timeout_seconds")
    command_timeout = _int_field(raw_target.get("command_timeout_seconds", 600), "target.command_timeout_seconds")
    if connect_timeout <= 0 or command_timeout <= 0:
        raise ArmTuneConfigError("target timeouts must be positive")
    return TargetSpec(
        type=target_type,
        host=_optional_str(raw_target.get("host")),
        user=_optional_str(raw_target.get("user")),
        port=port,
        identity_file=_expand_user(_optional_str(raw_target.get("identity_file"))),
        remote_workdir=str(raw_target.get("remote_workdir", "~/armtune-runs")),
        remote_python=_required_token(raw_target.get("remote_python", "python3"), "target.remote_python"),
        remote_module=_python_module(raw_target.get("remote_module", "autotune.cli")),
        ssh_executable=str(raw_target.get("ssh_executable", "ssh")),
        scp_executable=str(raw_target.get("scp_executable", "scp")),
        connect_timeout_seconds=connect_timeout,
        command_timeout_seconds=command_timeout,
        keep_remote_artifacts=bool(raw_target.get("keep_remote_artifacts", False)),
        strict_host_key_checking=bool(raw_target.get("strict_host_key_checking", True)),
    )


def default_local_target() -> TargetSpec:
    return TargetSpec(type="local")


def _int_field(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneConfigError(f"{label} must be an integer") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expand_user(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _required_token(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text or any(char.isspace() for char in text) or any(char in text for char in "\x00\r\n"):
        raise ArmTuneConfigError(f"{label} must be a non-empty token without whitespace")
    return text


def _python_module(value: Any) -> str:
    text = str(value).strip()
    if not PYTHON_MODULE_PATTERN.fullmatch(text):
        raise ArmTuneConfigError("target.remote_module must be a Python module path")
    return text
