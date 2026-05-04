from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotune.report.charts import metric_bar_chart


def generate_comparison_report(input_path: str | Path, output: str | Path | None = None) -> Path:
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    output_path = Path(output) if output is not None else source.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_comparison_report(data, source), encoding="utf-8")
    return output_path


def format_comparison_report(data: dict[str, Any], source: Path | None = None) -> str:
    baseline = _preferred_group(data, "baseline")
    tuned = _preferred_group(data, "tuned")
    deltas = _preferred_group(data, "deltas", aggregate=False)
    aggregate = data.get("aggregate", {})
    if aggregate:
        deltas = aggregate.get("deltas", deltas)
    lines = [
        "# AutoTuneAI Tuning Comparison",
        "",
        f"- Source: `{source}`" if source else "- Source: in-memory comparison",
        f"- Profile: `{data.get('tuned_profile')}`",
        f"- Repeat: {data.get('repeat', 1)}",
        f"- Baseline run: `{_run_ids(baseline)}`",
        f"- Tuned run: `{_run_ids(tuned)}`",
        "",
        "## Key Deltas",
        "",
        metric_bar_chart(
            "Performance Deltas",
            [
                ("benchmark duration %", _nested(deltas, "benchmark_duration_percent")),
                ("workload duration %", _nested(deltas, "workload_duration_percent")),
                ("samples/sec %", _nested(deltas, "workload", "samples_per_second", "percent")),
                ("peak memory %", _nested(deltas, "peak_memory_percent")),
                ("system tuning overhead sec", _nested(deltas, "system_tuning_overhead_seconds")),
            ],
        ),
        "",
        "## Baseline vs Tuned",
        "",
        metric_bar_chart(
            "Durations",
            [
                ("baseline benchmark sec", baseline.get("benchmark_duration_seconds")),
                ("tuned benchmark sec", tuned.get("benchmark_duration_seconds")),
                ("baseline lifecycle sec", baseline.get("lifecycle_duration_seconds")),
                ("tuned lifecycle sec", tuned.get("lifecycle_duration_seconds")),
            ],
            unit="s",
        ),
        "",
        metric_bar_chart(
            "Resource Peaks",
            [
                ("baseline peak memory MB", baseline.get("peak_memory_mb")),
                ("tuned peak memory MB", tuned.get("peak_memory_mb")),
                ("baseline system CPU %", baseline.get("peak_system_cpu_percent")),
                ("tuned system CPU %", tuned.get("peak_system_cpu_percent")),
            ],
        ),
        "",
        "## Diagnostics",
        "",
    ]
    diagnostics = [*baseline.get("diagnostics", []), *tuned.get("diagnostics", [])]
    if diagnostics:
        lines.extend(f"- {item}" for item in dict.fromkeys(diagnostics))
    else:
        lines.append("- No diagnostics emitted.")
    return "\n".join(lines) + "\n"


def _preferred_group(data: dict[str, Any], key: str, *, aggregate: bool = True) -> dict[str, Any]:
    if aggregate and isinstance(data.get("aggregate"), dict) and isinstance(data["aggregate"].get(key), dict):
        return data["aggregate"][key]
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _run_ids(group: dict[str, Any]) -> str:
    if "run_ids" in group:
        return ", ".join(str(item) for item in group.get("run_ids", []))
    return str(group.get("run_id"))
