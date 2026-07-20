from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autotune.golden_path.config import GoldenPathConfigError, objective_from_config, tuning_section, workload_section
from autotune.training_adapter import ADAPTER_TYPE, TRAINING_ADAPTER_SCHEMA_VERSION
from autotune.training_adapter.errors import TrainingConfigError


ALLOWLISTED_SEMANTICS = {
    "micro_batch_size",
    "gradient_accumulation",
    "dataloader_workers",
    "dataloader_prefetch",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "cpu_threads",
    "interop_threads",
    "intraop_threads",
    "amp_mode",
    "compile_mode",
}
DEFAULT_RESULT_PREFIX = "AUTOTUNEAI_TRAIN_RESULT "
DEFAULT_STEP_PREFIX = "AUTOTUNEAI_TRAIN_STEP "
ALLOWED_BINDING_TYPES = {
    "cli_argument",
    "env_var",
    "environment",
    "json_config_path",
    "json_path",
    "yaml_config_path",
    "yaml_path",
    "boolean_cli_flag",
    "adapter_environment",
}
ALLOWED_WORKLOAD_KEYS = {
    "adapter",
    "cwd",
    "timeout_seconds",
    "identity_files",
    "config_files",
    "runtime_version",
    "allowed_cpu_mask",
}
ALLOWED_TRAINING_KEYS = {
    "schema_version",
    "world_size",
    "slice",
    "progress",
    "semantics",
    "knobs",
    "capabilities",
    "search",
}
ALLOWED_SLICE_KEYS = {
    "mode",
    "flag",
    "env_var",
    "file",
    "path",
    "argv_flag",
    "warmup_steps",
    "measured_steps",
    "minimum_completed_steps",
    "max_wall_time_seconds",
    "step_count_value",
    "omit_from_full_run",
}
ALLOWED_PROGRESS_KEYS = {
    "source",
    "step_prefix",
    "result_prefix",
    "path",
    "completed_steps_field",
    "startup_timeout_seconds",
    "no_progress_timeout_seconds",
}
ALLOWED_SEMANTICS_KEYS = {"effective_batch_policy", "semantic_change_allowed"}
ALLOWED_CAPABILITY_KEYS = {"amp_modes", "compile_modes", "cpu_only", "cuda_allowed"}
ALLOWED_SEARCH_KEYS = {"max_candidates", "shuffle_seed", "stages", "cooldown_seconds"}
ALLOWED_KNOB_KEYS = {"name", "baseline", "candidates", "binding", "stage", "description"}
ALLOWED_BINDING_KEYS = {"type", "flag", "name", "env_name", "file", "path", "argv_flag"}


def is_training_adapter(config: dict[str, Any]) -> bool:
    try:
        workload = workload_section(config)
    except GoldenPathConfigError:
        return False
    return workload.get("adapter") == ADAPTER_TYPE


def validate_training_config(config: dict[str, Any]) -> dict[str, Any]:
    workload = workload_section(config)
    if workload.get("adapter") != ADAPTER_TYPE:
        raise TrainingConfigError(f"workload.adapter must be {ADAPTER_TYPE!r}")
    _reject_unknown(workload, ALLOWED_WORKLOAD_KEYS, "workload")
    training = training_section(config)
    _reject_unknown(training, ALLOWED_TRAINING_KEYS, "training")
    if training.get("schema_version", TRAINING_ADAPTER_SCHEMA_VERSION) != TRAINING_ADAPTER_SCHEMA_VERSION:
        raise TrainingConfigError(f"training.schema_version must be {TRAINING_ADAPTER_SCHEMA_VERSION!r}")
    _validate_world_size(training)
    _validate_slice(training)
    _validate_progress(training)
    _validate_semantics(training)
    _validate_capabilities(training)
    _validate_search(training)
    _validate_knobs(training)
    objective_from_config(config)
    return config


def training_section(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("training", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training must be an object")
    return raw


def workload_cwd(config: dict[str, Any]) -> Path:
    workload = workload_section(config)
    cwd = workload.get("cwd")
    return Path(str(cwd)).resolve() if isinstance(cwd, str) and cwd else Path.cwd().resolve()


def world_size(config: dict[str, Any]) -> int:
    return int(training_section(config).get("world_size", 1))


def slice_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(training_section(config).get("slice", {}))
    raw.setdefault("mode", "cli_argument")
    raw.setdefault("warmup_steps", 1)
    raw.setdefault("measured_steps", 8)
    raw.setdefault("minimum_completed_steps", int(raw.get("measured_steps", 8)))
    raw.setdefault("omit_from_full_run", True)
    return raw


def progress_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(training_section(config).get("progress", {}))
    raw.setdefault("source", "stdout_json_prefix")
    raw.setdefault("step_prefix", DEFAULT_STEP_PREFIX)
    raw.setdefault("result_prefix", DEFAULT_RESULT_PREFIX)
    raw.setdefault("completed_steps_field", "completed_steps")
    raw.setdefault("startup_timeout_seconds", 30)
    raw.setdefault("no_progress_timeout_seconds", 30)
    return raw


def semantics_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(training_section(config).get("semantics", {}))
    raw.setdefault("effective_batch_policy", "preserve")
    raw.setdefault("semantic_change_allowed", False)
    return raw


def capabilities_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(training_section(config).get("capabilities", {}))
    raw.setdefault("amp_modes", ["disabled"])
    raw.setdefault("compile_modes", ["disabled"])
    raw.setdefault("cpu_only", True)
    return raw


def search_config(config: dict[str, Any]) -> dict[str, Any]:
    training = training_section(config)
    raw = dict(training.get("search", {}))
    tuning = tuning_section(config)
    raw.setdefault("max_candidates", int(tuning.get("max_candidates", 8)))
    raw.setdefault("shuffle_seed", int(tuning.get("shuffle_seed", 0)))
    raw.setdefault("stages", ["input_pipeline", "batch_config", "cpu_runtime", "experimental"])
    raw.setdefault("cooldown_seconds", float(tuning.get("cooldown_seconds", 0.0)))
    return raw


def knob_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    knobs = training_section(config).get("knobs", [])
    if not isinstance(knobs, list):
        raise TrainingConfigError("training.knobs must be a list")
    return [dict(item) for item in knobs]


def has_bounded_slice(config: dict[str, Any]) -> bool:
    try:
        raw = training_section(config).get("slice")
    except TrainingConfigError:
        return False
    if not isinstance(raw, dict):
        return False
    mode = str(raw.get("mode") or "")
    measured = raw.get("measured_steps")
    return mode in {"cli_argument", "env_var", "json_config_path", "json_path", "yaml_config_path", "yaml_path"} and isinstance(measured, int) and measured > 0


def visible_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _validate_world_size(training: dict[str, Any]) -> None:
    value = training.get("world_size", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigError("training.world_size must be a positive integer")


def _validate_slice(training: dict[str, Any]) -> None:
    raw = training.get("slice", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training.slice must be an object")
    _reject_unknown(raw, ALLOWED_SLICE_KEYS, "training.slice")
    mode = str(raw.get("mode") or "cli_argument")
    if mode not in {"cli_argument", "env_var", "json_config_path", "json_path", "yaml_config_path", "yaml_path", "none"}:
        raise TrainingConfigError(f"unsupported training.slice.mode: {mode}")
    if mode == "cli_argument" and not isinstance(raw.get("flag"), str):
        raise TrainingConfigError("training.slice.flag is required when mode is cli_argument")
    if mode == "env_var" and not isinstance(raw.get("env_var"), str):
        raise TrainingConfigError("training.slice.env_var is required when mode is env_var")
    if mode in {"json_config_path", "json_path", "yaml_config_path", "yaml_path"}:
        if not isinstance(raw.get("file"), str) or not isinstance(raw.get("path"), str):
            raise TrainingConfigError("config-path slice modes require training.slice.file and training.slice.path")
    for key in ("warmup_steps", "measured_steps", "minimum_completed_steps"):
        value = raw.get(key, 0 if key == "warmup_steps" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingConfigError(f"training.slice.{key} must be a non-negative integer")
    if int(raw.get("measured_steps", 0)) <= 0:
        raise TrainingConfigError("training.slice.measured_steps must be positive")
    timeout = raw.get("max_wall_time_seconds", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise TrainingConfigError("training.slice.max_wall_time_seconds must be > 0")


def _validate_progress(training: dict[str, Any]) -> None:
    raw = training.get("progress", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training.progress must be an object")
    _reject_unknown(raw, ALLOWED_PROGRESS_KEYS, "training.progress")
    source = str(raw.get("source") or "stdout_json_prefix")
    if source not in {"stdout_json_prefix", "jsonl_file", "monotonic_metric"}:
        raise TrainingConfigError(f"unsupported training.progress.source: {source}")
    if source == "stdout_json_prefix":
        for key in ("step_prefix", "result_prefix"):
            value = raw.get(key, DEFAULT_STEP_PREFIX if key == "step_prefix" else DEFAULT_RESULT_PREFIX)
            if not isinstance(value, str) or not value:
                raise TrainingConfigError(f"training.progress.{key} must be a non-empty string")
    if source == "jsonl_file" and not isinstance(raw.get("path"), str):
        raise TrainingConfigError("training.progress.path is required when source is jsonl_file")


def _validate_semantics(training: dict[str, Any]) -> None:
    raw = training.get("semantics", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training.semantics must be an object")
    _reject_unknown(raw, ALLOWED_SEMANTICS_KEYS, "training.semantics")
    policy = str(raw.get("effective_batch_policy") or "preserve")
    if policy not in {"preserve", "allow_change"}:
        raise TrainingConfigError("training.semantics.effective_batch_policy must be preserve or allow_change")
    if policy == "allow_change" and raw.get("semantic_change_allowed") is not True:
        raise TrainingConfigError("allow_change requires training.semantics.semantic_change_allowed: true")


def _validate_capabilities(training: dict[str, Any]) -> None:
    raw = training.get("capabilities", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training.capabilities must be an object")
    _reject_unknown(raw, ALLOWED_CAPABILITY_KEYS, "training.capabilities")
    for key in ("amp_modes", "compile_modes"):
        values = raw.get(key, ["disabled"])
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise TrainingConfigError(f"training.capabilities.{key} must be a list of strings")


def _validate_search(training: dict[str, Any]) -> None:
    raw = training.get("search", {})
    if not isinstance(raw, dict):
        raise TrainingConfigError("training.search must be an object")
    _reject_unknown(raw, ALLOWED_SEARCH_KEYS, "training.search")
    max_candidates = raw.get("max_candidates", 8)
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 0:
        raise TrainingConfigError("training.search.max_candidates must be a non-negative integer")


def _validate_knobs(training: dict[str, Any]) -> None:
    raw = training.get("knobs", [])
    if not isinstance(raw, list):
        raise TrainingConfigError("training.knobs must be a list")
    seen: set[str] = set()
    for index, knob in enumerate(raw):
        if not isinstance(knob, dict):
            raise TrainingConfigError(f"training.knobs[{index}] must be an object")
        _reject_unknown(knob, ALLOWED_KNOB_KEYS, f"training.knobs[{index}]")
        name = knob.get("name")
        if name not in ALLOWLISTED_SEMANTICS:
            raise TrainingConfigError(f"training.knobs[{index}].name is not allowlisted: {name!r}")
        if name in seen:
            raise TrainingConfigError(f"duplicate training knob: {name}")
        seen.add(str(name))
        if "baseline" not in knob:
            raise TrainingConfigError(f"training knob {name} requires baseline")
        candidates = knob.get("candidates", [])
        if not isinstance(candidates, list):
            raise TrainingConfigError(f"training knob {name} candidates must be a list")
        binding = knob.get("binding")
        if not isinstance(binding, dict):
            raise TrainingConfigError(f"training knob {name} requires binding")
        _reject_unknown(binding, ALLOWED_BINDING_KEYS, f"training knob {name}.binding")
        binding_type = str(binding.get("type") or "")
        if binding_type not in ALLOWED_BINDING_TYPES:
            raise TrainingConfigError(f"unsupported binding type for {name}: {binding_type}")
        if binding_type == "cli_argument" and not isinstance(binding.get("flag"), str):
            raise TrainingConfigError(f"cli_argument binding for {name} requires flag")
        if binding_type == "boolean_cli_flag" and not isinstance(binding.get("flag"), str):
            raise TrainingConfigError(f"boolean_cli_flag binding for {name} requires flag")
        if binding_type in {"env_var", "environment", "adapter_environment"} and not (
            isinstance(binding.get("name"), str) or isinstance(binding.get("env_name"), str)
        ):
            raise TrainingConfigError(f"environment binding for {name} requires name or env_name")
        if binding_type in {"json_config_path", "json_path", "yaml_config_path", "yaml_path"}:
            if not isinstance(binding.get("file"), str) or not isinstance(binding.get("path"), str):
                raise TrainingConfigError(f"config-path binding for {name} requires file and path")


def _reject_unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise TrainingConfigError(f"{context} has unknown field(s): {', '.join(unknown)}")
