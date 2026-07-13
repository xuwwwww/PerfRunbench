from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.errors import ArmTuneProtocolError


ALLOWED_WORKER_OPERATIONS = {"check_prerequisites", "inspect_host"}
WORKER_JOB_KEYS = {"schema_version", "operation", "target_type", "remote_module", "created_at_utc"}
REMOTE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._~/:@+-]+$")
PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def validate_worker_job(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArmTuneProtocolError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    unknown = set(payload) - WORKER_JOB_KEYS
    if unknown:
        raise ArmTuneProtocolError(f"unsupported worker job field(s): {sorted(unknown)}")
    operation = payload.get("operation")
    if operation not in ALLOWED_WORKER_OPERATIONS:
        raise ArmTuneProtocolError(f"unsupported worker operation: {operation!r}")
    target_type = payload.get("target_type")
    if target_type not in {"local", "ssh"}:
        raise ArmTuneProtocolError(f"unsupported target_type: {target_type!r}")
    remote_module = payload.get("remote_module", "autotune.cli")
    if not isinstance(remote_module, str) or not PYTHON_MODULE_PATTERN.fullmatch(remote_module):
        raise ArmTuneProtocolError("remote_module must be a Python module path")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        temp_name = handle.name
    os.replace(temp_name, target)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArmTuneProtocolError(f"JSON payload must be an object: {path}")
    return payload


def safe_remote_path(path: str) -> str:
    if not path or any(char in path for char in "\x00\r\n"):
        raise ArmTuneProtocolError("remote path must be non-empty and must not contain control characters")
    if not REMOTE_PATH_PATTERN.fullmatch(path):
        raise ArmTuneProtocolError("remote path contains unsupported characters")
    return path


def validate_remote_identity(value: str, *, label: str) -> str:
    if not value or any(char.isspace() for char in value) or any(char in value for char in "\x00\r\n"):
        raise ArmTuneProtocolError(f"{label} must not be empty or contain whitespace/control characters")
    return value


def validate_remote_module(value: str) -> str:
    if not PYTHON_MODULE_PATTERN.fullmatch(value):
        raise ArmTuneProtocolError("remote module must be a Python module path")
    return value
