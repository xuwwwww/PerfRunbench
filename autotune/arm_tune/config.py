from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.llama_bench import BENCHMARK_OPERATION, PROTOCOL_VERSION
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import (
    BenchmarkArtifactsSpec,
    BenchmarkExecutableSpec,
    BenchmarkModelSpec,
    LlamaBenchJobSpec,
    LlamaBenchSettings,
    TargetSpec,
)
from autotune.utils.config import load_config


TOP_LEVEL_KEYS = {"schema_version", "target"}
BENCHMARK_TOP_LEVEL_KEYS = {
    "schema_version",
    "target",
    "result_origin",
    "executable",
    "model",
    "benchmark",
    "artifacts",
}
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
EXECUTABLE_KEYS = {"path", "expected_sha256"}
MODEL_KEYS = {"path", "expected_sha256"}
BENCHMARK_KEYS = {
    "threads",
    "prompt_tokens",
    "generation_tokens",
    "repetitions",
    "timeout_seconds",
    "additional_supported_options",
}
ARTIFACT_KEYS = {"preserve_raw_stdout", "preserve_raw_stderr"}
ADDITIONAL_SUPPORTED_OPTION_KEYS = {"batch_size", "ubatch_size"}
PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def load_target_spec(path: str | Path) -> TargetSpec:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("target config must be an object")
    return parse_target_spec(payload)


def load_benchmark_config(path: str | Path) -> tuple[TargetSpec, LlamaBenchJobSpec]:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("benchmark config must be an object")
    return parse_benchmark_config(payload)


def parse_benchmark_config(payload: dict[str, Any]) -> tuple[TargetSpec, LlamaBenchJobSpec]:
    unknown = set(payload) - BENCHMARK_TOP_LEVEL_KEYS
    if unknown:
        raise ArmTuneConfigError(f"unknown benchmark config field(s): {sorted(unknown)}")
    schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise ArmTuneConfigError(f"unsupported schema_version: {schema_version!r}")
    raw_target = payload.get("target")
    if not isinstance(raw_target, dict):
        raise ArmTuneConfigError("target must be an object")
    target = parse_target_spec({"schema_version": schema_version, "target": raw_target})
    executable = _object_field(payload, "executable")
    model = _object_field(payload, "model")
    benchmark = _object_field(payload, "benchmark")
    artifacts = payload.get("artifacts", {})
    if artifacts is None:
        artifacts = {}
    if not isinstance(artifacts, dict):
        raise ArmTuneConfigError("artifacts must be an object")
    _reject_unknown(executable, EXECUTABLE_KEYS, "executable")
    _reject_unknown(model, MODEL_KEYS, "model")
    _reject_unknown(benchmark, BENCHMARK_KEYS, "benchmark")
    _reject_unknown(artifacts, ARTIFACT_KEYS, "artifacts")
    executable_path = _absolute_path_field(executable.get("path"), "executable.path")
    model_path = _absolute_path_field(model.get("path"), "model.path")
    additional = benchmark.get("additional_supported_options", {})
    if additional is None:
        additional = {}
    if not isinstance(additional, dict):
        raise ArmTuneConfigError("benchmark.additional_supported_options must be an object")
    extra_unknown = set(additional) - ADDITIONAL_SUPPORTED_OPTION_KEYS
    if extra_unknown:
        raise ArmTuneConfigError(f"unsupported benchmark.additional_supported_options field(s): {sorted(extra_unknown)}")
    settings = LlamaBenchSettings(
        threads=_positive_int(benchmark.get("threads"), "benchmark.threads"),
        prompt_tokens=_positive_int(benchmark.get("prompt_tokens"), "benchmark.prompt_tokens"),
        generation_tokens=_positive_int(benchmark.get("generation_tokens"), "benchmark.generation_tokens"),
        repetitions=_positive_int(benchmark.get("repetitions"), "benchmark.repetitions"),
        timeout_seconds=_positive_float(benchmark.get("timeout_seconds"), "benchmark.timeout_seconds"),
        additional_supported_options={key: _positive_int(value, f"benchmark.additional_supported_options.{key}") for key, value in additional.items()},
    )
    result_origin = str(payload.get("result_origin", "measured"))
    if result_origin not in {"measured", "test_fixture"}:
        raise ArmTuneConfigError("result_origin must be 'measured' or 'test_fixture'")
    job = LlamaBenchJobSpec(
        protocol_version=PROTOCOL_VERSION,
        operation=BENCHMARK_OPERATION,
        run_id="pending",
        target_run_directory="pending",
        executable=BenchmarkExecutableSpec(
            path=executable_path,
            expected_sha256=_optional_sha256(executable.get("expected_sha256"), "executable.expected_sha256"),
        ),
        model=BenchmarkModelSpec(
            path=model_path,
            expected_sha256=_optional_sha256(model.get("expected_sha256"), "model.expected_sha256"),
        ),
        benchmark=settings,
        artifacts=BenchmarkArtifactsSpec(
            preserve_raw_stdout=bool(artifacts.get("preserve_raw_stdout", True)),
            preserve_raw_stderr=bool(artifacts.get("preserve_raw_stderr", True)),
        ),
        target_type=target.type,
        remote_module=target.remote_module,
        result_origin=result_origin,
    )
    return target, job


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


def _object_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArmTuneConfigError(f"{key} must be an object")
    return value


def _reject_unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ArmTuneConfigError(f"unknown {label} field(s): {sorted(unknown)}")


def _absolute_path_field(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ArmTuneConfigError(f"{label} is required")
    if not (Path(text).is_absolute() or text.startswith("/")):
        raise ArmTuneConfigError(f"{label} must be absolute on the target")
    return text


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneConfigError(f"{label} must be an integer") from exc
    if parsed < 1:
        raise ArmTuneConfigError(f"{label} must be >= 1")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmTuneConfigError(f"{label} must be a number") from exc
    if parsed <= 0:
        raise ArmTuneConfigError(f"{label} must be > 0")
    return parsed


def _optional_sha256(value: Any, label: str) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ArmTuneConfigError(f"{label} must be a 64-character hex SHA-256 digest")
    return text
