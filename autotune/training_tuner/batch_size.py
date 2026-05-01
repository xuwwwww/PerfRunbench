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
class BatchSizeTrial:
    batch_size: int
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
) -> dict[str, Any]:
    if not values:
        raise BatchSizeTuningError("at least one batch size value is required")
    if not command:
        raise BatchSizeTuningError("command cannot be empty")
    config_path = Path(config_file)
    current_value, find_text = find_batch_size_assignment(config_path, key)
    trials: list[BatchSizeTrial] = []
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
            auto_restore=True,
        )
        summary = _load_resource_summary(run_dir)
        safe, reason = _trial_safety(return_code, summary)
        manifest = load_manifest(run_dir)
        trials.append(
            BatchSizeTrial(
                batch_size=value,
                run_id=manifest["run_id"],
                run_dir=str(run_dir),
                return_code=return_code,
                safe=safe,
                reason=reason,
                resource_summary=summary,
            )
        )
    safe_trials = [trial for trial in trials if trial.safe]
    recommended = max(safe_trials, key=lambda trial: trial.batch_size) if safe_trials else None
    result = {
        "config_file": str(config_path),
        "key": key,
        "original_batch_size": current_value,
        "candidate_values": values,
        "recommended_batch_size": recommended.batch_size if recommended else None,
        "recommended_run_id": recommended.run_id if recommended else None,
        "trials": [_trial_to_dict(trial) for trial in trials],
    }
    write_json(Path(output), result)
    return result


def find_batch_size_assignment(config_file: str | Path, key: str) -> tuple[int, str]:
    path = Path(config_file)
    if not path.exists():
        raise BatchSizeTuningError(f"config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^([ \t]*{re.escape(key)}[ \t]*[:=][ \t]*)(\d+)([ \t]*(?:#.*)?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise BatchSizeTuningError(f"could not find a numeric assignment for key {key!r} in {path}")
    if len(matches) > 1:
        raise BatchSizeTuningError(f"found {len(matches)} assignments for key {key!r}; refusing ambiguous edit")
    match = matches[0]
    return int(match.group(2)), match.group(0)


def replace_assignment_value(assignment_line: str, value: int) -> str:
    pattern = re.compile(r"^([ \t]*[^:=]+?[ \t]*[:=][ \t]*)(\d+)([ \t]*(?:#.*)?)$")
    match = pattern.match(assignment_line)
    if not match:
        raise BatchSizeTuningError(f"unsupported assignment line: {assignment_line}")
    return f"{match.group(1)}{value}{match.group(3)}"


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


def _trial_to_dict(trial: BatchSizeTrial) -> dict[str, Any]:
    return {
        "batch_size": trial.batch_size,
        "run_id": trial.run_id,
        "run_dir": trial.run_dir,
        "return_code": trial.return_code,
        "safe": trial.safe,
        "reason": trial.reason,
        "resource_summary": trial.resource_summary,
    }
