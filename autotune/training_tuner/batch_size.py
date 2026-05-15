from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest, write_json
from autotune.source_tuner.tuned_runner import SourceEdit, run_tuned_with_budget


class BatchSizeTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class NumericConfigTrial:
    key: str
    value: int
    run_id: str
    run_dir: str
    return_code: int
    safe: bool
    reason: str
    resource_summary: dict[str, Any]


def tune_batch_size(
    config_file: str | Path,
    key: str,
    values: list[int],
    command: list[str],
    budget: ResourceBudget,
    output: str | Path,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    docker_image: str = "python:3.12-slim",
) -> dict[str, Any]:
    return tune_numeric_config_key(
        config_file,
        key,
        values,
        command,
        budget,
        output,
        sample_interval_seconds=sample_interval_seconds,
        hard_kill=hard_kill,
        executor=executor,
        use_sudo=use_sudo,
        allow_sudo_auto=allow_sudo_auto,
        docker_image=docker_image,
        compatibility_batch_fields=True,
    )


def tune_numeric_config_key(
    config_file: str | Path,
    key: str,
    values: list[int],
    command: list[str],
    budget: ResourceBudget,
    output: str | Path,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
    docker_image: str = "python:3.12-slim",
    compatibility_batch_fields: bool = False,
) -> dict[str, Any]:
    if not values:
        raise BatchSizeTuningError("at least one candidate value is required")
    if not command:
        raise BatchSizeTuningError("command cannot be empty")
    config_path = Path(config_file)
    current_value, find_text = find_numeric_assignment(config_path, key)
    trials: list[NumericConfigTrial] = []
    for value in values:
        replace_text = replace_assignment_value(find_text, value)
        return_code, run_dir = run_tuned_with_budget(
            command,
            [SourceEdit(str(config_path), find_text, replace_text)],
            budget,
            sample_interval_seconds=sample_interval_seconds,
            hard_kill=hard_kill,
            executor=executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
            docker_image=docker_image,
            auto_restore=True,
        )
        summary = _load_resource_summary(run_dir)
        safe, reason = _trial_safety(return_code, summary)
        manifest = load_manifest(run_dir)
        trials.append(
            NumericConfigTrial(
                key=key,
                value=value,
                run_id=manifest["run_id"],
                run_dir=str(run_dir),
                return_code=return_code,
                safe=safe,
                reason=reason,
                resource_summary=summary,
            )
        )
    safe_trials = [trial for trial in trials if trial.safe]
    recommended = max(safe_trials, key=lambda trial: trial.value) if safe_trials else None
    result = {
        "config_file": str(config_path),
        "key": key,
        "original_value": current_value,
        "candidate_values": values,
        "recommended_value": recommended.value if recommended else None,
        "recommended_run_id": recommended.run_id if recommended else None,
        "trials": [_trial_to_dict(trial) for trial in trials],
    }
    if compatibility_batch_fields:
        result.update(
            {
                "original_batch_size": current_value,
                "recommended_batch_size": recommended.value if recommended else None,
            }
        )
    write_json(Path(output), result)
    return result


def find_numeric_assignment(config_file: str | Path, key: str) -> tuple[int, str]:
    value, assignment = find_scalar_assignment(config_file, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BatchSizeTuningError(f"assignment for key {key!r} is not an integer in {config_file}")
    return value, assignment


def find_scalar_assignment(config_file: str | Path, key: str) -> tuple[int | float | bool | str, str]:
    path = Path(config_file)
    if not path.exists():
        raise BatchSizeTuningError(f"config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^([ \t]*{re.escape(key)}[ \t]*[:=][ \t]*)([^#\r\n]+?)([ \t]*(?:#.*)?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise BatchSizeTuningError(f"could not find an assignment for key {key!r} in {path}")
    if len(matches) > 1:
        raise BatchSizeTuningError(f"found {len(matches)} assignments for key {key!r}; refusing ambiguous edit")
    match = matches[0]
    return parse_scalar_value(match.group(2).strip()), match.group(0)


def find_batch_size_assignment(config_file: str | Path, key: str) -> tuple[int, str]:
    return find_numeric_assignment(config_file, key)


def replace_assignment_value(assignment_line: str, value: Any) -> str:
    pattern = re.compile(r"^([ \t]*[^:=]+?[ \t]*[:=][ \t]*)([^#\r\n]+?)([ \t]*(?:#.*)?)$")
    match = pattern.match(assignment_line)
    if not match:
        raise BatchSizeTuningError(f"unsupported assignment line: {assignment_line}")
    return f"{match.group(1)}{format_scalar_value(value)}{match.group(3)}"


def parse_scalar_value(text: str) -> int | float | bool | str:
    normalized = text.strip()
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", normalized):
        return int(normalized)
    if re.fullmatch(r"-?(?:\d+\.\d+|\d+\.|\.\d+)", normalized):
        return float(normalized)
    if (normalized.startswith('"') and normalized.endswith('"')) or (
        normalized.startswith("'") and normalized.endswith("'")
    ):
        return normalized[1:-1]
    return normalized


def format_scalar_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_resource_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resource_summary.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _trial_safety(return_code: int, summary: dict[str, Any]) -> tuple[bool, str]:
    if return_code != 0:
        return False, f"command returned {return_code}"
    if summary.get("memory_budget_exceeded") is True:
        return False, "memory budget exceeded"
    return True, "completed within budget"


def _trial_to_dict(trial: NumericConfigTrial) -> dict[str, Any]:
    return {
        "key": trial.key,
        "value": trial.value,
        "batch_size": trial.value,
        "run_id": trial.run_id,
        "run_dir": trial.run_dir,
        "return_code": trial.return_code,
        "safe": trial.safe,
        "reason": trial.reason,
        "resource_summary": trial.resource_summary,
    }
