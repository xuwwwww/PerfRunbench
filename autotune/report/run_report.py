from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR


def generate_run_report(run_id: str, output: str | Path | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    analysis = analyze_run(run_id, runs_dir)
    run_dir = runs_dir / run_id
    report_path = Path(output) if output is not None else run_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_run_report(analysis, run_dir), encoding="utf-8")
    return report_path


def format_run_report(analysis: dict[str, Any], run_dir: Path) -> str:
    lines = [
        f"# AutoTuneAI Run Report: {analysis['run_id']}",
        "",
        "## Summary",
        "",
        f"- Status: {analysis.get('status')}",
        f"- Return code: {analysis.get('return_code')}",
        f"- Command: `{_shell_join(analysis.get('command', []))}`",
        f"- Run directory: `{run_dir}`",
        "",
        "## Before / After",
        "",
        f"- Available memory at start MB: {analysis['memory'].get('available_memory_start_mb')}",
        f"- Available memory at end MB: {analysis['memory'].get('available_memory_end_mb')}",
        f"- Minimum available memory MB: {analysis['memory'].get('observed_min_available_memory_mb')}",
        f"- Workload peak memory MB: {analysis['memory'].get('peak_memory_mb')}",
        f"- Memory budget exceeded: {analysis['memory'].get('memory_budget_exceeded')}",
        f"- System tuning snapshots: {_system_tuning_snapshot_status(run_dir)}",
        f"- Source/config changes recorded: {_changed_file_count(run_dir)}",
        "",
        "## Executor",
        "",
        f"- Requested: {analysis['executor'].get('requested')}",
        f"- Selected: {analysis['executor'].get('selected')}",
        f"- sudo used: {analysis['executor'].get('sudo_used')}",
        "",
        "## CPU",
        "",
        f"- Logical CPU count: {analysis['cpu'].get('logical_cpu_count')}",
        f"- Reserved cores: {analysis['cpu'].get('requested_reserve_cores')}",
        f"- Allowed threads: {analysis['cpu'].get('allowed_threads')}",
        f"- Affinity applied: {analysis['cpu'].get('affinity_applied')}",
        f"- Affinity cores: {analysis['cpu'].get('affinity_cores')}",
        f"- Expected max total CPU percent: {analysis['cpu'].get('expected_max_total_cpu_percent')}",
        f"- Observed peak process CPU percent: {analysis['cpu'].get('observed_peak_process_cpu_percent')}",
        "",
        "## Memory",
        "",
        f"- Mode: {analysis['memory'].get('mode')}",
        f"- Requested memory GB: {analysis['memory'].get('requested_memory_gb')}",
        f"- Effective budget MB: {analysis['memory'].get('effective_budget_mb')}",
        f"- Peak memory MB: {analysis['memory'].get('peak_memory_mb')}",
        f"- Min available memory MB: {analysis['memory'].get('observed_min_available_memory_mb')}",
        f"- Memory budget exceeded: {analysis['memory'].get('memory_budget_exceeded')}",
        "",
        "## Cgroup",
        "",
        f"- Available: {analysis['cgroup'].get('available')}",
        f"- Path: {analysis['cgroup'].get('path')}",
        f"- Peak cgroup memory MB: {analysis['cgroup'].get('peak_memory_mb')}",
        f"- Peak cgroup CPU percent: {analysis['cgroup'].get('peak_cpu_percent')}",
        "",
        "## Diagnostics",
        "",
    ]
    diagnostics = analysis.get("diagnostics", [])
    if diagnostics:
        lines.extend(f"- {item}" for item in diagnostics)
    else:
        lines.append("- No diagnostics emitted.")
    system_diff = _load_json(run_dir / "system_tuning_diff.json", default=None)
    if system_diff is not None:
        lines.extend(["", "## System Tuning Diff", ""])
        for item in system_diff:
            source = item.get("source") or "sysctl"
            lines.append(
                f"- {item.get('key')} ({source}): before={item.get('before')} target={item.get('target')} "
                f"after={item.get('after')} changed={item.get('changed')} applied={item.get('applied')}"
            )
            if item.get("path"):
                lines.append(f"  - path: `{item.get('path')}`")
            if item.get("error"):
                lines.append(f"  - error: {item.get('error')}")
    return "\n".join(lines) + "\n"


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _system_tuning_snapshot_status(run_dir: Path) -> str:
    required = [
        "system_tuning_before.json",
        "system_tuning_after.json",
        "system_tuning_diff.json",
    ]
    present = [name for name in required if (run_dir / name).exists()]
    if not present:
        return "none"
    return ", ".join(present)


def _changed_file_count(run_dir: Path) -> int:
    manifest = _load_json(run_dir / "manifest.json", default={})
    return len(manifest.get("changed_files", []))


def _shell_join(command: list[str]) -> str:
    return " ".join(str(part) for part in command)
