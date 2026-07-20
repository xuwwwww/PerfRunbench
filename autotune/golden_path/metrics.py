from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from autotune.golden_path.config import metrics_section


ALLOWED_METRIC_SOURCES = {"json_file", "jsonl_file", "stdout_json_prefix", "adapter_result_file"}


class MetricSourceError(RuntimeError):
    pass


def load_metrics_for_run(run_dir: str | Path, config: dict[str, Any], *, cwd: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics = metrics_section(config)
    source = str(metrics.get("source") or "json_file")
    if source not in ALLOWED_METRIC_SOURCES:
        raise MetricSourceError(f"unsupported metrics source: {source}")
    if source == "stdout_json_prefix":
        prefix = metrics.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            raise MetricSourceError("stdout_json_prefix requires metrics.prefix")
        return _load_stdout_record(run_path / "stdout.txt", prefix)
    path = _metric_path(run_path, metrics, cwd=cwd)
    if source in {"json_file", "adapter_result_file"}:
        return _metrics_from_json(path)
    return _metrics_from_jsonl(path)


def objective_value(metrics: dict[str, Any], metric: str) -> float | None:
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _metric_path(run_dir: Path, metrics: dict[str, Any], *, cwd: str | Path | None) -> Path:
    raw_path = metrics.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise MetricSourceError("file-based metrics source requires metrics.path")
    path = Path(raw_path.format(run_dir=str(run_dir)))
    if path.is_absolute():
        return path
    base = str(metrics.get("base") or "run_dir")
    if base == "cwd":
        return (Path(cwd) if cwd else Path.cwd()) / path
    if base != "run_dir":
        raise MetricSourceError("metrics.base must be 'run_dir' or 'cwd'")
    return run_dir / path


def _metrics_from_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MetricSourceError(f"metrics file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetricSourceError(f"metrics file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise MetricSourceError("metrics JSON must be an object")
    if isinstance(raw.get("metrics"), dict):
        return raw["metrics"]
    return raw


def _metrics_from_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MetricSourceError(f"metrics JSONL file not found: {path}")
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise MetricSourceError("metrics JSONL records must be objects")
        last = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
    if last is None:
        raise MetricSourceError("metrics JSONL file has no records")
    return last


def _load_stdout_record(path: Path, prefix: str) -> dict[str, Any]:
    if not path.exists():
        raise MetricSourceError(f"captured stdout not found: {path}")
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MetricSourceError("stdout metric record is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise MetricSourceError("stdout metric record must be an object")
        records.append(raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw)
    if len(records) != 1:
        raise MetricSourceError(f"expected exactly one stdout metric record, found {len(records)}")
    return records[0]

