from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.errors import ArmTuneProtocolError


ALLOWED_WORKER_OPERATIONS = {"benchmark.llama_bench", "benchmark.llama_server", "check_prerequisites", "inspect_host"}
LLAMA_BENCH_WORKER_JOB_KEYS = {
    "schema_version",
    "protocol_version",
    "operation",
    "run_id",
    "target_run_directory",
    "executable",
    "model",
    "benchmark",
    "artifacts",
    "target_type",
    "remote_module",
    "result_origin",
    "created_at_utc",
}
LLAMA_SERVER_WORKER_JOB_KEYS = {
    "schema_version",
    "protocol_version",
    "operation",
    "run_id",
    "target_run_directory",
    "executable",
    "model",
    "server",
    "workload",
    "artifacts",
    "target_type",
    "remote_module",
    "result_origin",
    "created_at_utc",
}
SIMPLE_WORKER_JOB_KEYS = {"schema_version", "operation", "target_type", "remote_module", "created_at_utc"}
EXECUTABLE_KEYS = {"schema_version", "path", "expected_sha256"}
MODEL_KEYS = {"schema_version", "path", "expected_sha256"}
BENCHMARK_KEYS = {
    "schema_version",
    "threads",
    "prompt_tokens",
    "generation_tokens",
    "repetitions",
    "timeout_seconds",
    "additional_supported_options",
    "cpu_affinity",
    "affinity_mode",
}
ARTIFACT_KEYS = {"schema_version", "preserve_raw_stdout", "preserve_raw_stderr"}
SERVER_KEYS = {
    "schema_version",
    "host",
    "port",
    "startup_timeout_seconds",
    "request_timeout_seconds",
    "shutdown_timeout_seconds",
    "threads",
    "context_size",
    "batch_size",
    "seed",
    "temperature",
}
SERVER_WORKLOAD_KEYS = {
    "schema_version",
    "prompt_suite_path",
    "predict_tokens",
    "repetitions",
    "warmup_requests",
    "streaming",
    "cooldown_seconds",
}
SERVER_ARTIFACT_KEYS = {
    "schema_version",
    "preserve_server_stdout",
    "preserve_server_stderr",
    "preserve_request_stream",
    "preserve_generated_output",
}
ADDITIONAL_SUPPORTED_OPTION_KEYS = {"threads_batch", "batch_size", "ubatch_size"}
REMOTE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._~/:@+-]+$")
PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def validate_worker_job(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArmTuneProtocolError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    operation = payload.get("operation")
    if operation == "benchmark.llama_bench":
        allowed_keys = LLAMA_BENCH_WORKER_JOB_KEYS
    elif operation == "benchmark.llama_server":
        allowed_keys = LLAMA_SERVER_WORKER_JOB_KEYS
    else:
        allowed_keys = SIMPLE_WORKER_JOB_KEYS
    unknown = set(payload) - allowed_keys
    if unknown:
        raise ArmTuneProtocolError(f"unsupported worker job field(s): {sorted(unknown)}")
    if operation not in ALLOWED_WORKER_OPERATIONS:
        raise ArmTuneProtocolError(f"unsupported worker operation: {operation!r}")
    target_type = payload.get("target_type")
    if target_type not in {"local", "ssh"}:
        raise ArmTuneProtocolError(f"unsupported target_type: {target_type!r}")
    remote_module = payload.get("remote_module", "autotune.cli")
    if not isinstance(remote_module, str) or not PYTHON_MODULE_PATTERN.fullmatch(remote_module):
        raise ArmTuneProtocolError("remote_module must be a Python module path")
    if operation == "benchmark.llama_bench":
        _validate_benchmark_job(payload)
    if operation == "benchmark.llama_server":
        _validate_server_benchmark_job(payload)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_name: str | None = None
    try:
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
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except OSError:
                pass
        raise
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


def _validate_benchmark_job(payload: dict[str, Any]) -> None:
    if payload.get("protocol_version") != "1.0":
        raise ArmTuneProtocolError(f"unsupported protocol_version: {payload.get('protocol_version')!r}")
    for key in ["run_id", "target_run_directory", "executable", "model", "benchmark", "artifacts"]:
        if key not in payload:
            raise ArmTuneProtocolError(f"benchmark job missing required field: {key}")
    if payload.get("result_origin", "unverified") not in {"measured", "test_fixture", "unverified"}:
        raise ArmTuneProtocolError("result_origin must be 'measured', 'test_fixture', or 'unverified'")
    _validate_object(payload["executable"], EXECUTABLE_KEYS, "executable")
    _validate_object(payload["model"], MODEL_KEYS, "model")
    _validate_object(payload["benchmark"], BENCHMARK_KEYS, "benchmark")
    _validate_object(payload["artifacts"], ARTIFACT_KEYS, "artifacts")
    if not _is_absolute_target_path(str(payload["executable"].get("path", ""))):
        raise ArmTuneProtocolError("executable.path must be absolute on the target")
    if not _is_absolute_target_path(str(payload["model"].get("path", ""))):
        raise ArmTuneProtocolError("model.path must be absolute on the target")
    benchmark = payload["benchmark"]
    for key in ["threads", "prompt_tokens", "generation_tokens", "repetitions"]:
        _positive_int(benchmark.get(key), f"benchmark.{key}")
    _positive_float(benchmark.get("timeout_seconds"), "benchmark.timeout_seconds")
    additional = benchmark.get("additional_supported_options", {})
    if not isinstance(additional, dict):
        raise ArmTuneProtocolError("benchmark.additional_supported_options must be an object")
    unknown_options = set(additional) - ADDITIONAL_SUPPORTED_OPTION_KEYS
    if unknown_options:
        raise ArmTuneProtocolError(f"unsupported benchmark.additional_supported_options field(s): {sorted(unknown_options)}")
    for key, value in additional.items():
        _positive_int(value, f"benchmark.additional_supported_options.{key}")
    _optional_cpu_affinity(benchmark.get("cpu_affinity"), "benchmark.cpu_affinity")
    affinity_mode = str(benchmark.get("affinity_mode", "scheduler-default"))
    if affinity_mode not in {"scheduler-default", "compact", "spread", "reserve-one-core", "numa-local"}:
        raise ArmTuneProtocolError("benchmark.affinity_mode is unsupported")


def _validate_server_benchmark_job(payload: dict[str, Any]) -> None:
    if payload.get("protocol_version") != "1.0":
        raise ArmTuneProtocolError(f"unsupported protocol_version: {payload.get('protocol_version')!r}")
    for key in ["run_id", "target_run_directory", "executable", "model", "server", "workload", "artifacts"]:
        if key not in payload:
            raise ArmTuneProtocolError(f"server benchmark job missing required field: {key}")
    if payload.get("result_origin", "unverified") not in {"measured", "test_fixture", "unverified"}:
        raise ArmTuneProtocolError("result_origin must be 'measured', 'test_fixture', or 'unverified'")
    _validate_object(payload["executable"], EXECUTABLE_KEYS, "executable")
    _validate_object(payload["model"], MODEL_KEYS, "model")
    _validate_object(payload["server"], SERVER_KEYS, "server")
    _validate_object(payload["workload"], SERVER_WORKLOAD_KEYS, "workload")
    _validate_object(payload["artifacts"], SERVER_ARTIFACT_KEYS, "artifacts")
    if not _is_absolute_target_path(str(payload["executable"].get("path", ""))):
        raise ArmTuneProtocolError("executable.path must be absolute on the target")
    if not _is_absolute_target_path(str(payload["model"].get("path", ""))):
        raise ArmTuneProtocolError("model.path must be absolute on the target")
    server = payload["server"]
    host = str(server.get("host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost"}:
        raise ArmTuneProtocolError("server.host must be loopback-only")
    port = server.get("port", "auto")
    if port != "auto":
        _positive_int(port, "server.port")
        if int(port) > 65535:
            raise ArmTuneProtocolError("server.port must be between 1 and 65535")
    for key in ["threads", "context_size"]:
        _positive_int(server.get(key), f"server.{key}")
    for key in ["startup_timeout_seconds", "request_timeout_seconds", "shutdown_timeout_seconds"]:
        _positive_float(server.get(key), f"server.{key}")
    if server.get("batch_size") is not None:
        _positive_int(server.get("batch_size"), "server.batch_size")
    if isinstance(server.get("seed"), bool) or not isinstance(server.get("seed"), int):
        raise ArmTuneProtocolError("server.seed must be an integer")
    temperature = server.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) < 0:
        raise ArmTuneProtocolError("server.temperature must be a non-negative number")
    workload = payload["workload"]
    if not _is_absolute_target_path(str(workload.get("prompt_suite_path", ""))):
        raise ArmTuneProtocolError("workload.prompt_suite_path must be absolute on the target")
    for key in ["predict_tokens", "repetitions"]:
        _positive_int(workload.get(key), f"workload.{key}")
    _non_negative_int(workload.get("warmup_requests"), "workload.warmup_requests")
    _non_negative_float(workload.get("cooldown_seconds"), "workload.cooldown_seconds")
    if workload.get("streaming") is not True:
        raise ArmTuneProtocolError("workload.streaming must be true")


def _validate_object(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ArmTuneProtocolError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ArmTuneProtocolError(f"unsupported {label} field(s): {sorted(unknown)}")


def _is_absolute_target_path(path: str) -> bool:
    return bool(path) and (Path(path).is_absolute() or path.startswith("/"))


def _positive_int(value: Any, label: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneProtocolError(f"{label} must be an integer") from exc
    if parsed < 1:
        raise ArmTuneProtocolError(f"{label} must be >= 1")


def _non_negative_int(value: Any, label: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneProtocolError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ArmTuneProtocolError(f"{label} must be >= 0")


def _positive_float(value: Any, label: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneProtocolError(f"{label} must be a number") from exc
    if parsed <= 0:
        raise ArmTuneProtocolError(f"{label} must be > 0")


def _non_negative_float(value: Any, label: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneProtocolError(f"{label} must be a number") from exc
    if parsed < 0:
        raise ArmTuneProtocolError(f"{label} must be >= 0")


def _optional_cpu_affinity(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ArmTuneProtocolError(f"{label} must be a list of CPU indexes")
    seen: set[int] = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ArmTuneProtocolError(f"{label} must contain only integers") from exc
        if parsed < 0:
            raise ArmTuneProtocolError(f"{label} must contain non-negative CPU indexes")
        if parsed in seen:
            raise ArmTuneProtocolError(f"{label} must not contain duplicate CPU indexes")
        seen.add(parsed)
