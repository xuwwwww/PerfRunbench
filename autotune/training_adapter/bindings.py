from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import LaunchSpec, file_sha256, stable_digest
from autotune.training_adapter import ADAPTER_TYPE
from autotune.training_adapter.config import slice_config, workload_cwd
from autotune.training_adapter.errors import TrainingBindingError
from autotune.utils.config import load_config

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on optional local install
    yaml = None


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    stage: str
    knob_values: dict[str, Any]
    slice_argv: list[str]
    full_argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    overlays: list[dict[str, Any]] = field(default_factory=list)
    effective_batch_size: int | None = None
    baseline_effective_batch_size: int | None = None
    semantic_change_state: str = "preserved"
    recommendation_eligible: bool = True
    guard_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "id": self.candidate_id,
            "stage": self.stage,
            "knob_values": self.knob_values,
            "env": self.env,
            "slice_argv_redacted": _public_argv(self.slice_argv),
            "full_argv_redacted": _public_argv(self.full_argv),
            "overlays": self.overlays,
            "effective_batch_size": self.effective_batch_size,
            "baseline_effective_batch_size": self.baseline_effective_batch_size,
            "semantic_change_state": self.semantic_change_state,
            "recommendation_eligible": self.recommendation_eligible,
            "guard_results": self.guard_results,
        }


def build_candidate_plan(
    command: list[str],
    config: dict[str, Any],
    candidate: dict[str, Any],
    *,
    plan_root: str | Path,
) -> CandidatePlan:
    root = Path(plan_root)
    root.mkdir(parents=True, exist_ok=True)
    cwd = workload_cwd(config)
    slice_argv = [str(item) for item in command]
    full_argv = [str(item) for item in command]
    env: dict[str, str] = {}
    overlays: list[dict[str, Any]] = []
    for knob in candidate.get("knobs", []):
        value = candidate["knob_values"][str(knob["name"])]
        binding = knob["binding"]
        slice_argv, env, knob_overlay = _apply_binding(slice_argv, env, binding, value, cwd=cwd, root=root, candidate_id=candidate["id"])
        full_argv, env, full_overlay = _apply_binding(full_argv, env, binding, value, cwd=cwd, root=root, candidate_id=candidate["id"], full_run=True)
        overlays.extend(knob_overlay)
        overlays.extend(full_overlay)
    slice_argv, env, slice_overlay = _apply_slice(slice_argv, env, config, cwd=cwd, root=root, candidate_id=candidate["id"])
    overlays.extend(slice_overlay)
    return CandidatePlan(
        candidate_id=str(candidate["id"]),
        stage=str(candidate.get("stage") or "baseline"),
        knob_values=dict(candidate.get("knob_values", {})),
        slice_argv=slice_argv,
        full_argv=full_argv,
        env=env,
        overlays=overlays,
        effective_batch_size=candidate.get("effective_batch_size"),
        baseline_effective_batch_size=candidate.get("baseline_effective_batch_size"),
        semantic_change_state=str(candidate.get("semantic_change_state") or "preserved"),
        recommendation_eligible=bool(candidate.get("recommendation_eligible", True)),
        guard_results=list(candidate.get("guard_results", [])),
    )


def launch_spec_from_plan(plan: CandidatePlan, config: dict[str, Any], *, full_run: bool, config_path: str | Path | None) -> dict[str, Any]:
    from autotune.golden_path.identity import launch_spec_for_command

    argv = plan.full_argv if full_run else plan.slice_argv
    spec = launch_spec_for_command(argv, config, env=plan.env, config_path=config_path)
    spec["adapter"] = ADAPTER_TYPE
    metadata = spec.get("metadata", {}) if isinstance(spec.get("metadata"), dict) else {}
    metadata.update(
        {
            "candidate_id": plan.candidate_id,
            "training_knob_values": plan.knob_values,
            "effective_batch_size": plan.effective_batch_size,
            "baseline_effective_batch_size": plan.baseline_effective_batch_size,
            "semantic_change_state": plan.semantic_change_state,
            "recommendation_eligible": plan.recommendation_eligible,
            "launch_mode": "full_run" if full_run else "bounded_slice",
        }
    )
    spec["metadata"] = metadata
    LaunchSpec(**{key: spec[key] for key in ("argv", "adapter", "cwd", "env", "timeout_seconds", "metadata", "expected_artifacts", "correctness_contract", "schema_version") if key in spec})
    return spec


def _apply_binding(
    argv: list[str],
    env: dict[str, str],
    binding: dict[str, Any],
    value: Any,
    *,
    cwd: Path,
    root: Path,
    candidate_id: str,
    full_run: bool = False,
) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    binding_type = _canonical_binding_type(str(binding.get("type")))
    overlays: list[dict[str, Any]] = []
    if binding_type == "cli_argument":
        return _replace_or_append_arg(argv, str(binding["flag"]), str(value)), env, overlays
    if binding_type == "boolean_cli_flag":
        return _set_boolean_flag(argv, str(binding["flag"]), bool(value)), env, overlays
    if binding_type in {"env_var", "adapter_environment"}:
        name = str(binding.get("env_name") or binding.get("name"))
        next_env = dict(env)
        next_env[name] = _stringify_env(value)
        return argv, next_env, overlays
    if binding_type in {"json_config_path", "yaml_config_path"}:
        next_argv, overlay = _overlay_config_path(
            argv,
            binding,
            value,
            cwd=cwd,
            root=root,
            candidate_id=candidate_id,
            full_run=full_run,
        )
        overlays.append(overlay)
        return next_argv, env, overlays
    raise TrainingBindingError(f"unsupported binding type: {binding_type}")


def _apply_slice(
    argv: list[str],
    env: dict[str, str],
    config: dict[str, Any],
    *,
    cwd: Path,
    root: Path,
    candidate_id: str,
) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    raw = slice_config(config)
    mode = _canonical_binding_type(str(raw.get("mode") or "cli_argument"))
    total_steps = int(raw.get("step_count_value") or int(raw.get("warmup_steps", 0)) + int(raw["measured_steps"]))
    overlays: list[dict[str, Any]] = []
    if mode == "cli_argument":
        return _replace_or_append_arg(argv, str(raw["flag"]), str(total_steps)), env, overlays
    if mode == "env_var":
        next_env = dict(env)
        next_env[str(raw["env_var"])] = str(total_steps)
        return argv, next_env, overlays
    if mode in {"json_config_path", "yaml_config_path"}:
        next_argv, overlay = _overlay_config_path(argv, raw, total_steps, cwd=cwd, root=root, candidate_id=f"{candidate_id}_slice", full_run=False)
        overlay["purpose"] = "slice_control"
        overlays.append(overlay)
        return next_argv, env, overlays
    if mode == "none":
        raise TrainingBindingError("training adapter tuning requires bounded slice control")
    raise TrainingBindingError(f"unsupported slice mode: {mode}")


def _overlay_config_path(
    argv: list[str],
    binding: dict[str, Any],
    value: Any,
    *,
    cwd: Path,
    root: Path,
    candidate_id: str,
    full_run: bool,
) -> tuple[list[str], dict[str, Any]]:
    source = Path(str(binding["file"]))
    if not source.is_absolute():
        source = cwd / source
    source = source.resolve()
    if not source.exists() or not source.is_file():
        raise TrainingBindingError(f"overlay source config not found: {source.name}")
    relative_name = f"{candidate_id}_{'full' if full_run else 'slice'}_{source.name}"
    target = (root / "overlays" / relative_name).resolve()
    if root.resolve() not in [target, *target.parents]:
        raise TrainingBindingError("overlay target escaped the candidate artifact directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = load_config(source)
    if not isinstance(payload, dict):
        raise TrainingBindingError("overlay config must be a JSON/YAML object")
    mutated = json.loads(json.dumps(payload))
    _set_path(mutated, str(binding["path"]), value)
    _write_config(target, mutated)
    next_argv = _replace_config_reference(argv, source, target, str(binding.get("argv_flag") or ""), cwd=cwd)
    return next_argv, {
        "schema_version": "1.0",
        "source_display": source.name,
        "source_sha256": file_sha256(source),
        "overlay_relative_path": str(target.relative_to(root.resolve())).replace("\\", "/"),
        "overlay_sha256": file_sha256(target),
        "path": str(binding["path"]),
        "full_run": full_run,
    }


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = _path_parts(path)
    cursor: Any = payload
    for part in parts[:-1]:
        if isinstance(part, int):
            if not isinstance(cursor, list) or part >= len(cursor):
                raise TrainingBindingError(f"overlay path does not exist: {path}")
            cursor = cursor[part]
        else:
            if not isinstance(cursor, dict) or part not in cursor:
                raise TrainingBindingError(f"overlay path does not exist: {path}")
            cursor = cursor[part]
    last = parts[-1]
    if isinstance(last, int):
        if not isinstance(cursor, list) or last >= len(cursor):
            raise TrainingBindingError(f"overlay path does not exist: {path}")
        existing = cursor[last]
        _reject_type_change(existing, value, path)
        cursor[last] = value
    else:
        if not isinstance(cursor, dict) or last not in cursor:
            raise TrainingBindingError(f"overlay path does not exist: {path}")
        existing = cursor[last]
        _reject_type_change(existing, value, path)
        cursor[last] = value


def _path_parts(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for item in path.split("."):
        if not item or item in {".", ".."} or "/" in item or "\\" in item:
            raise TrainingBindingError(f"unsafe overlay path: {path}")
        if item.isdigit():
            parts.append(int(item))
        else:
            parts.append(item)
    if not parts:
        raise TrainingBindingError("overlay path cannot be empty")
    return parts


def _reject_type_change(existing: Any, value: Any, path: str) -> None:
    if isinstance(existing, bool) or isinstance(value, bool):
        if type(existing) is not type(value):
            raise TrainingBindingError(f"overlay path {path} would change value type")
        return
    if isinstance(existing, (int, float)) and isinstance(value, (int, float)):
        return
    if type(existing) is not type(value):
        raise TrainingBindingError(f"overlay path {path} would change value type")


def _replace_or_append_arg(argv: list[str], flag: str, value: str) -> list[str]:
    result = list(argv)
    for index, item in enumerate(result):
        if item == flag:
            if index + 1 >= len(result):
                raise TrainingBindingError(f"flag {flag} has no value")
            result[index + 1] = value
            return result
        if item.startswith(f"{flag}="):
            result[index] = f"{flag}={value}"
            return result
    result.extend([flag, value])
    return result


def _set_boolean_flag(argv: list[str], flag: str, enabled: bool) -> list[str]:
    result = [item for item in argv if item != flag and not item.startswith(f"{flag}=")]
    if enabled:
        result.append(flag)
    return result


def _replace_config_reference(argv: list[str], source: Path, target: Path, argv_flag: str, *, cwd: Path) -> list[str]:
    result = list(argv)
    if argv_flag:
        for index, item in enumerate(result):
            if item == argv_flag:
                if index + 1 >= len(result):
                    raise TrainingBindingError(f"argv_flag {argv_flag} has no value")
                result[index + 1] = str(target)
                return result
            if item.startswith(f"{argv_flag}="):
                result[index] = f"{argv_flag}={target}"
                return result
        raise TrainingBindingError(f"argv_flag {argv_flag} was not present in command")
    for index, item in enumerate(result):
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            if candidate.resolve() == source:
                result[index] = str(target)
                return result
        except OSError:
            continue
    raise TrainingBindingError(f"source config {source.name} is not referenced by the command")


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            path.write_text(_simple_yaml_dump(payload), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return
    raise TrainingBindingError(f"unsupported overlay config suffix: {path.suffix}")


def _simple_yaml_dump(payload: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(payload, dict):
        lines = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.append(_simple_yaml_dump(value, indent + 2).rstrip())
            else:
                lines.append(f"{spaces}{key}: {json.dumps(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(payload, list):
        lines = []
        for value in payload:
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}-")
                lines.append(_simple_yaml_dump(value, indent + 2).rstrip())
            else:
                lines.append(f"{spaces}- {json.dumps(value)}")
        return "\n".join(lines) + "\n"
    return f"{spaces}{json.dumps(payload)}\n"


def _canonical_binding_type(value: str) -> str:
    if value == "environment":
        return "env_var"
    if value == "json_path":
        return "json_config_path"
    if value == "yaml_path":
        return "yaml_config_path"
    return value


def _stringify_env(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _public_argv(argv: list[str]) -> list[str]:
    from autotune.golden_path.identity import redact_argv

    return redact_argv(argv)
