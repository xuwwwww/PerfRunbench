from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autotune.golden_path.config import objective_from_config
from autotune.recommendation.schemas import stable_digest
from autotune.training_adapter.config import progress_config, slice_config
from autotune.training_adapter.errors import TrainingMetricError


CONVENTIONAL_FIELDS = {
    "completed_steps",
    "samples_per_second",
    "step_time_ms",
    "final_loss",
    "peak_memory_mb",
    "examples_processed",
    "tokens_per_second",
    "tokens_processed",
}
OOM_MARKERS = ("out of memory", "cuda oom", "cublas_status_alloc_failed", "memoryerror")
KILLED_MARKERS = ("killed", "sigkill")


def load_training_result_for_run(run_dir: str | Path, config: dict[str, Any], *, return_code: int) -> dict[str, Any]:
    run_path = Path(run_dir)
    progress = progress_config(config)
    stdout_path = run_path / "stdout.txt"
    stderr_path = run_path / "stderr.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    diagnostics: dict[str, Any] = {
        "schema_version": "1.0",
        "progress_source": progress["source"],
        "stdout_path": str(stdout_path) if stdout_path.exists() else None,
        "stderr_path": str(stderr_path) if stderr_path.exists() else None,
        "step_record_count": 0,
        "result_record_count": 0,
        "failure_class": None,
        "semantic_failure_categories": [],
    }
    try:
        if progress["source"] == "stdout_json_prefix":
            parsed = _records_from_stdout(stdout, step_prefix=str(progress["step_prefix"]), result_prefix=str(progress["result_prefix"]))
        elif progress["source"] == "jsonl_file":
            parsed = _records_from_jsonl(_progress_path(run_path, progress))
        else:
            parsed = _records_from_stdout(stdout, step_prefix=str(progress["step_prefix"]), result_prefix=str(progress["result_prefix"]))
        diagnostics["step_record_count"] = len(parsed["steps"])
        diagnostics["result_record_count"] = len(parsed["results"])
        metrics = _final_metrics(parsed, config)
        diagnostics["completed_steps"] = metrics.get("completed_steps")
        diagnostics["metric_fields"] = sorted(str(key) for key in metrics if key in CONVENTIONAL_FIELDS)
        _validate_metrics(metrics, config)
    except TrainingMetricError as exc:
        diagnostics["failure_class"] = _classify_failure(return_code, stdout, stderr, metric_error=str(exc))
        diagnostics["semantic_failure_categories"].append(str(exc))
        raise TrainingMetricError(json.dumps(diagnostics, sort_keys=True)) from exc
    if return_code != 0:
        diagnostics["failure_class"] = _classify_failure(return_code, stdout, stderr)
        raise TrainingMetricError(json.dumps(diagnostics, sort_keys=True))
    diagnostics["failure_class"] = None
    return {"metrics": metrics, "diagnostics": diagnostics}


def protocol_digest(config: dict[str, Any]) -> str:
    objective = objective_from_config(config)
    return stable_digest(
        {
            "schema_version": "1.0",
            "adapter": "pytorch-training",
            "slice": slice_config(config),
            "progress": progress_config(config),
            "objective": objective,
        }
    )


def failure_diagnostics_from_exception(exc: Exception) -> dict[str, Any]:
    try:
        raw = json.loads(str(exc))
    except json.JSONDecodeError:
        return {"schema_version": "1.0", "failure_class": "invalid_metric", "semantic_failure_categories": [str(exc)]}
    return raw if isinstance(raw, dict) else {"schema_version": "1.0", "failure_class": "invalid_metric"}


def _records_from_stdout(stdout: str, *, step_prefix: str, result_prefix: str) -> dict[str, list[dict[str, Any]]]:
    steps = []
    results = []
    for line in stdout.splitlines():
        if line.startswith(step_prefix):
            steps.append(_loads_record(line[len(step_prefix) :], "step"))
        elif line.startswith(result_prefix):
            results.append(_loads_record(line[len(result_prefix) :], "result"))
    return {"steps": steps, "results": results}


def _records_from_jsonl(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise TrainingMetricError(f"progress file missing: {path.name}")
    steps = []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = _loads_record(line, "jsonl")
        kind = record.get("kind") or record.get("type") or "step"
        if kind == "result":
            results.append(record)
        else:
            steps.append(record)
    return {"steps": steps, "results": results}


def _loads_record(text: str, context: str) -> dict[str, Any]:
    try:
        raw = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise TrainingMetricError(f"{context} record is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise TrainingMetricError(f"{context} record must be an object")
    return raw


def _final_metrics(parsed: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    results = parsed["results"]
    if len(results) != 1:
        raise TrainingMetricError(f"expected exactly one training result record, found {len(results)}")
    result = results[0]
    raw_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else result
    metrics = dict(raw_metrics)
    completed_field = str(progress_config(config)["completed_steps_field"])
    if "completed_steps" not in metrics and completed_field in result:
        metrics["completed_steps"] = result[completed_field]
    return metrics


def _validate_metrics(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    objective = objective_from_config(config)
    metric_name = objective["metric"]
    minimum_steps = int(slice_config(config).get("minimum_completed_steps", 1))
    completed = metrics.get("completed_steps")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise TrainingMetricError("completed_steps missing")
    if completed < minimum_steps:
        raise TrainingMetricError("incomplete_steps")
    value = metrics.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainingMetricError("objective metric missing or nonfinite")
    final_loss = metrics.get("final_loss")
    if final_loss is not None and (isinstance(final_loss, bool) or not isinstance(final_loss, (int, float)) or not math.isfinite(float(final_loss))):
        raise TrainingMetricError("nonfinite_loss")
    processed = [metrics.get("examples_processed"), metrics.get("tokens_processed")]
    if all(not isinstance(item, (int, float)) or isinstance(item, bool) or float(item) <= 0 for item in processed) and completed <= 0:
        raise TrainingMetricError("zero_work")


def _classify_failure(return_code: int, stdout: str, stderr: str, *, metric_error: str | None = None) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if return_code == 124:
        return "timeout"
    if any(marker in combined for marker in OOM_MARKERS):
        return "out_of_memory"
    if return_code in {-9, 137} or any(marker in combined for marker in KILLED_MARKERS):
        return "killed_by_system"
    if metric_error:
        if "incomplete_steps" in metric_error:
            return "incomplete_steps"
        if "nonfinite_loss" in metric_error:
            return "nonfinite_loss"
        if "expected exactly one training result record, found 0" in metric_error:
            return "stalled"
        return "invalid_metric"
    if return_code != 0:
        return "nonzero_exit"
    return "invalid_metric"


def _progress_path(run_path: Path, progress: dict[str, Any]) -> Path:
    raw_path = progress.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TrainingMetricError("jsonl_file progress requires path")
    path = Path(raw_path.format(run_dir=str(run_path)))
    return path if path.is_absolute() else run_path / path
