from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from autotune.resource.run_state import RUNS_DIR


def analyze_run(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_dir}")
    manifest = _load_json(run_dir / "manifest.json")
    summary = _load_json(run_dir / "resource_summary.json", default={})
    timeline = _load_json(run_dir / "resource_timeline.json", default=[])
    budget = manifest.get("budget", {})
    notes = manifest.get("notes", [])
    affinity = _parse_affinity_context(notes)
    executor = _parse_executor(notes)
    memory = _analyze_memory(budget, summary, timeline)
    cpu = _analyze_cpu(budget, summary, timeline, affinity)
    cgroup = _analyze_cgroup(summary, timeline, notes)
    system_tuning = _analyze_system_tuning(run_dir, notes)
    workload = _load_json(run_dir / "training_metrics.json", default={})
    diagnostics = [*cpu["diagnostics"], *memory["diagnostics"], *cgroup["diagnostics"], *system_tuning["diagnostics"]]
    if workload:
        diagnostics.append("Workload wrote training metrics to training_metrics.json.")
    return {
        "run_id": run_id,
        "status": manifest.get("status"),
        "return_code": manifest.get("return_code"),
        "command": manifest.get("command", []),
        "executor": executor,
        "cpu": cpu,
        "memory": memory,
        "cgroup": cgroup,
        "system_tuning": system_tuning,
        "workload": workload,
        "diagnostics": diagnostics,
        "paths": {
            "run_dir": str(run_dir),
            "manifest": str(run_dir / "manifest.json"),
            "resource_summary": str(run_dir / "resource_summary.json"),
            "resource_timeline": str(run_dir / "resource_timeline.json"),
            "system_tuning_diff": str(run_dir / "system_tuning_diff.json"),
            "system_tuning_restore_after": str(run_dir / "system_tuning_restore_after.json"),
            "training_metrics": str(run_dir / "training_metrics.json"),
        },
    }


def format_analysis(analysis: dict[str, Any]) -> str:
    lines = [
        f"Run {analysis['run_id']}",
        f"status={analysis.get('status')} return_code={analysis.get('return_code')}",
        "",
        "Executor",
        f"  requested: {analysis['executor'].get('requested')}",
        f"  selected: {analysis['executor'].get('selected')}",
        f"  sudo_used: {analysis['executor'].get('sudo_used')}",
        "",
        "CPU",
        f"  logical_cpu_count: {analysis['cpu'].get('logical_cpu_count')}",
        f"  requested_reserve_cores: {analysis['cpu'].get('requested_reserve_cores')}",
        f"  allowed_threads: {analysis['cpu'].get('allowed_threads')}",
        f"  affinity_applied: {analysis['cpu'].get('affinity_applied')}",
        f"  affinity_cores: {analysis['cpu'].get('affinity_cores')}",
        f"  expected_max_total_cpu_percent: {analysis['cpu'].get('expected_max_total_cpu_percent')}",
        f"  observed_peak_process_cpu_percent: {analysis['cpu'].get('observed_peak_process_cpu_percent')}",
        f"  observed_peak_system_cpu_percent: {analysis['cpu'].get('observed_peak_system_cpu_percent')}",
        "",
        "Memory",
        f"  mode: {analysis['memory'].get('mode')}",
        f"  requested_memory_gb: {analysis['memory'].get('requested_memory_gb')}",
        f"  effective_budget_mb: {analysis['memory'].get('effective_budget_mb')}",
        f"  peak_memory_mb: {analysis['memory'].get('peak_memory_mb')}",
        f"  available_memory_start_mb: {analysis['memory'].get('available_memory_start_mb')}",
        f"  available_memory_end_mb: {analysis['memory'].get('available_memory_end_mb')}",
        f"  observed_min_available_memory_mb: {analysis['memory'].get('observed_min_available_memory_mb')}",
        f"  memory_budget_exceeded: {analysis['memory'].get('memory_budget_exceeded')}",
        "",
        "Cgroup",
        f"  available: {analysis['cgroup'].get('available')}",
        f"  path: {analysis['cgroup'].get('path')}",
        f"  peak_cgroup_memory_mb: {analysis['cgroup'].get('peak_memory_mb')}",
        f"  peak_cgroup_cpu_percent: {analysis['cgroup'].get('peak_cpu_percent')}",
        "",
        "System Tuning",
        f"  profile: {analysis.get('system_tuning', {}).get('profile')}",
        f"  changed_settings: {analysis.get('system_tuning', {}).get('changed_settings')}",
        f"  restored_settings: {analysis.get('system_tuning', {}).get('restored_settings')}",
        "",
        "Workload",
        "",
        "Diagnostics",
    ]
    workload = analysis.get("workload", {})
    if workload:
        for key in sorted(workload):
            lines.insert(-2, f"  {key}: {workload.get(key)}")
    else:
        lines.insert(-2, "  no training metrics")
    for item in analysis.get("diagnostics", []):
        lines.append(f"  - {item}")
    return "\n".join(lines)


def _analyze_cpu(
    budget: dict[str, Any],
    summary: dict[str, Any],
    timeline: list[dict[str, Any]],
    affinity: dict[str, Any],
) -> dict[str, Any]:
    logical = _to_int(affinity.get("logical_cpu_count"))
    allowed = _to_int(affinity.get("allowed_threads") or budget.get("allowed_threads"))
    expected = round((allowed / logical) * 100, 3) if logical and allowed else None
    peak_process = summary.get("peak_child_cpu_percent")
    system_values = [sample.get("system_cpu_percent") for sample in timeline if sample.get("system_cpu_percent") is not None]
    peak_system = round(max(system_values), 3) if system_values else None
    diagnostics = []
    if affinity.get("cpu_affinity_applied"):
        diagnostics.append(
            "CPU affinity was applied to the Linux process. Host UI may still show activity spread across cores under WSL."
        )
    if logical and allowed and allowed < logical:
        diagnostics.append(
            f"CPU reserve is consistent with allowed_threads={allowed}/{logical}, expected total usage around {expected}%."
        )
    if peak_process is not None and expected is not None and peak_process <= expected + 10:
        diagnostics.append("Observed process CPU peak is broadly consistent with the requested CPU limit.")
    return {
        "logical_cpu_count": logical,
        "requested_reserve_cores": budget.get("reserve_cores"),
        "requested_cpu_quota_percent": budget.get("cpu_quota_percent"),
        "allowed_threads": allowed,
        "affinity_applied": affinity.get("cpu_affinity_applied"),
        "affinity_cores": affinity.get("affinity_cores"),
        "expected_max_total_cpu_percent": expected,
        "observed_peak_process_cpu_percent": peak_process,
        "observed_peak_system_cpu_percent": peak_system,
        "diagnostics": diagnostics,
    }


def _analyze_memory(budget: dict[str, Any], summary: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    available = [sample.get("available_memory_mb") for sample in timeline if sample.get("available_memory_mb") is not None]
    start = round(available[0], 3) if available else None
    end = summary.get("available_memory_after_mb") if summary else None
    min_available = round(min(available), 3) if available else None
    effective = summary.get("effective_memory_budget_mb") or budget.get("effective_memory_budget_mb")
    peak = summary.get("peak_cgroup_memory_current_mb") or summary.get("peak_rss_mb")
    requested_gb = budget.get("memory_budget_gb")
    expected_free_gb = abs(requested_gb) if isinstance(requested_gb, (int, float)) and requested_gb < 0 else None
    observed_free_gb = round(min_available / 1024, 3) if min_available is not None else None
    diagnostics = []
    if budget.get("memory_budget_mode") == "reserve_to_full":
        diagnostics.append(
            "Negative memory budget is computed against Linux/WSL-visible total memory, not Windows Task Manager memory."
        )
        diagnostics.append(
            "Available memory includes page cache and WSL reclamation effects, so observed host headroom can differ from the configured reserve."
        )
    if summary.get("memory_budget_exceeded"):
        diagnostics.append("Memory budget was exceeded according to the monitored workload memory.")
    elif effective is not None and peak is not None:
        diagnostics.append("Monitored workload memory stayed within the effective memory budget.")
    return {
        "mode": budget.get("memory_budget_mode"),
        "requested_memory_gb": requested_gb,
        "requested_reserve_to_full_gb": expected_free_gb,
        "effective_budget_mb": effective,
        "peak_memory_mb": peak,
        "available_memory_start_mb": start,
        "available_memory_end_mb": end,
        "observed_min_available_memory_mb": min_available,
        "observed_min_available_memory_gb": observed_free_gb,
        "memory_budget_exceeded": summary.get("memory_budget_exceeded"),
        "diagnostics": diagnostics,
    }


def _analyze_cgroup(summary: dict[str, Any], timeline: list[dict[str, Any]], notes: list[str]) -> dict[str, Any]:
    cgroup_paths = [sample.get("cgroup_path") for sample in timeline if sample.get("cgroup_path")]
    control_group = _note_value(notes, "systemd_control_group=")
    path = summary.get("cgroup_path") or (cgroup_paths[-1] if cgroup_paths else None)
    if path is None and control_group:
        path = _control_group_path(control_group)
    diagnostics = []
    if path:
        if summary.get("peak_cgroup_memory_current_mb") is not None or summary.get("peak_cgroup_cpu_percent") is not None:
            diagnostics.append("Cgroup stats were captured; prefer cgroup memory for systemd runs.")
        else:
            diagnostics.append("Systemd control group was discovered, but cgroup stat files were not readable during sampling.")
    elif any("selected_executor=systemd" in note for note in notes):
        diagnostics.append("Systemd executor was selected, but cgroup stats were not captured for this run.")
    return {
        "available": bool(path),
        "path": path,
        "control_group": control_group,
        "peak_memory_mb": summary.get("peak_cgroup_memory_current_mb") or summary.get("peak_cgroup_memory_peak_mb"),
        "peak_cpu_percent": summary.get("peak_cgroup_cpu_percent"),
        "diagnostics": diagnostics,
    }


def _analyze_system_tuning(run_dir: Path, notes: list[str]) -> dict[str, Any]:
    diff = _load_json(run_dir / "system_tuning_diff.json", default=[])
    restored = _load_json(run_dir / "system_tuning_restore_after.json", default=[])
    profile = _note_value(notes, "system_tuning_profile=")
    changed = [item for item in diff if item.get("changed")]
    applied = [item for item in diff if item.get("applied")]
    failed = [item for item in diff if item.get("error")]
    diagnostics = []
    if profile:
        diagnostics.append(f"System tuning profile {profile} was selected for this run.")
    if applied:
        diagnostics.append(f"System tuning applied {len(applied)} setting(s), with {len(changed)} effective change(s).")
    if restored:
        failed_restores = [item for item in restored if item.get("return_code") != 0]
        diagnostics.append(f"System tuning restored {len(restored) - len(failed_restores)}/{len(restored)} setting(s).")
    if failed:
        diagnostics.append(f"System tuning had {len(failed)} setting error(s); inspect system_tuning_diff.json.")
    return {
        "profile": profile,
        "settings": diff,
        "restore": restored,
        "changed_settings": len(changed),
        "applied_settings": len(applied),
        "restored_settings": len(restored),
        "failed_settings": len(failed),
        "apply_seconds": _to_float(_note_value(notes, "system_tuning_apply_seconds=")),
        "restore_seconds": _to_float(_note_value(notes, "system_tuning_restore_seconds=")),
        "diagnostics": diagnostics,
    }


def _parse_executor(notes: list[str]) -> dict[str, Any]:
    result = {"requested": None, "selected": None, "sudo_used": None, "platform": None}
    for note in notes:
        for key, output_key in [
            ("requested_executor=", "requested"),
            ("selected_executor=", "selected"),
            ("sudo_used=", "sudo_used"),
            ("executor_platform=", "platform"),
        ]:
            if note.startswith(key):
                value = note[len(key) :]
                result[output_key] = _parse_value(value)
    return result


def _note_value(notes: list[str], prefix: str) -> str | None:
    for note in notes:
        if note.startswith(prefix):
            return note[len(prefix) :]
    return None


def _control_group_path(control_group: str) -> str:
    return f"/sys/fs/cgroup/{control_group.lstrip('/')}"


def _parse_affinity_context(notes: list[str]) -> dict[str, Any]:
    for note in notes:
        if not note.startswith("affinity_context="):
            continue
        payload = note[len("affinity_context=") :]
        return {
            "cpu_affinity_applied": "cpu_affinity_applied': True" in payload
            or '"cpu_affinity_applied": true' in payload,
            "logical_cpu_count": _regex_value(payload, r"'logical_cpu_count': ([^,}]+)"),
            "allowed_threads": _regex_value(payload, r"'allowed_threads': ([^,}]+)"),
            "affinity_cores": _regex_value(payload, r"'affinity_cores': '([^']+)'"),
        }
    return {}


def _regex_value(text: str, pattern: str) -> Any:
    match = re.search(pattern, text)
    if not match:
        return None
    return _parse_value(match.group(1))


def _parse_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
