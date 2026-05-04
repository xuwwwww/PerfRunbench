from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotune.report.charts import metric_bar_chart, sparkline_svg
from autotune.resource.run_analysis import analyze_run
from autotune.resource.run_state import RUNS_DIR


def generate_run_report(run_id: str, output: str | Path | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    analysis = analyze_run(run_id, runs_dir)
    run_dir = runs_dir / run_id
    report_path = Path(output) if output is not None else run_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.suffix.lower() == ".html":
        content = format_run_report_html(analysis, run_dir)
    else:
        content = format_run_report(analysis, run_dir)
    report_path.write_text(content, encoding="utf-8")
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
        f"- Training metrics captured: {bool(analysis.get('workload'))}",
        f"- System tuning snapshots: {_system_tuning_snapshot_status(run_dir)}",
        f"- Source/config changes recorded: {_changed_file_count(run_dir)}",
        "",
        "## Visual Summary",
        "",
        metric_bar_chart(
            "Run Summary",
            [
                ("peak memory MB", analysis["memory"].get("peak_memory_mb")),
                ("min available memory MB", analysis["memory"].get("observed_min_available_memory_mb")),
                ("peak process CPU %", analysis["cpu"].get("observed_peak_process_cpu_percent")),
                ("peak system CPU %", analysis["cpu"].get("observed_peak_system_cpu_percent")),
                ("workload samples/sec", analysis.get("workload", {}).get("samples_per_second")),
            ],
        ),
        "",
        sparkline_svg(
            "Available Memory Timeline",
            [item.get("available_memory_mb") for item in _load_json(run_dir / "resource_timeline.json", default=[])],
        ),
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
        "## Workload",
        "",
    ]
    workload = analysis.get("workload", {})
    if workload:
        for key in sorted(workload):
            lines.append(f"- {key}: {workload.get(key)}")
    else:
        lines.append("- No workload metrics captured.")
    lines.extend([
        "",
        "## Diagnostics",
        "",
    ])
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
    gpu_diff = _load_json(run_dir / "gpu_tuning_diff.json", default=None)
    if gpu_diff is not None:
        lines.extend(["", "## GPU Tuning Diff", ""])
        for item in gpu_diff:
            lines.append(
                f"- {item.get('key')}: before={item.get('before')} target={item.get('target')} "
                f"return_code={item.get('return_code')}"
            )
            if item.get("command"):
                lines.append(f"  - command: `{' '.join(str(part) for part in item.get('command'))}`")
            if item.get("error"):
                lines.append(f"  - error: {item.get('error')}")
    return "\n".join(lines) + "\n"


def format_run_report_html(analysis: dict[str, Any], run_dir: Path) -> str:
    workload = analysis.get("workload", {})
    diagnostics = analysis.get("diagnostics", [])
    workload_items = "".join(
        f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(workload.get(key))}</td></tr>"
        for key in sorted(workload)
    ) or "<tr><th>metrics</th><td>No workload metrics captured.</td></tr>"
    diagnostic_items = "".join(f"<li>{_html_escape(item)}</li>" for item in diagnostics) or "<li>No diagnostics emitted.</li>"
    summary_rows = [
        ("Status", analysis.get("status")),
        ("Return code", analysis.get("return_code")),
        ("Command", _shell_join(analysis.get("command", []))),
        ("Run directory", str(run_dir)),
    ]
    memory_rows = [
        ("Available memory at start MB", analysis["memory"].get("available_memory_start_mb")),
        ("Available memory at end MB", analysis["memory"].get("available_memory_end_mb")),
        ("Minimum available memory MB", analysis["memory"].get("observed_min_available_memory_mb")),
        ("Workload peak memory MB", analysis["memory"].get("peak_memory_mb")),
        ("Memory budget exceeded", analysis["memory"].get("memory_budget_exceeded")),
        ("Training metrics captured", bool(workload)),
        ("System tuning snapshots", _system_tuning_snapshot_status(run_dir)),
        ("Source/config changes recorded", _changed_file_count(run_dir)),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>AutoTuneAI Run Report {analysis['run_id']}</title>",
            "<style>",
            _run_report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main class=\"page\">",
            "<section class=\"hero\">",
            "<p class=\"eyebrow\">AutoTuneAI</p>",
            f"<h1>Run Report: {_html_escape(analysis['run_id'])}</h1>",
            "</section>",
            _html_table_card("Summary", summary_rows),
            _html_table_card("Before / After", memory_rows),
            "<section class=\"card\">",
            "<h2>Visual Summary</h2>",
            metric_bar_chart(
                "Run Summary",
                [
                    ("peak memory MB", analysis["memory"].get("peak_memory_mb")),
                    ("min available memory MB", analysis["memory"].get("observed_min_available_memory_mb")),
                    ("peak process CPU %", analysis["cpu"].get("observed_peak_process_cpu_percent")),
                    ("peak system CPU %", analysis["cpu"].get("observed_peak_system_cpu_percent")),
                    ("workload samples/sec", workload.get("samples_per_second")),
                ],
            ),
            sparkline_svg(
                "Available Memory Timeline",
                [item.get("available_memory_mb") for item in _load_json(run_dir / "resource_timeline.json", default=[])],
            ),
            "</section>",
            _html_table_card(
                "Executor",
                [
                    ("Requested", analysis["executor"].get("requested")),
                    ("Selected", analysis["executor"].get("selected")),
                    ("sudo used", analysis["executor"].get("sudo_used")),
                ],
            ),
            _html_table_card(
                "CPU",
                [
                    ("Logical CPU count", analysis["cpu"].get("logical_cpu_count")),
                    ("Reserved cores", analysis["cpu"].get("requested_reserve_cores")),
                    ("Allowed threads", analysis["cpu"].get("allowed_threads")),
                    ("Affinity applied", analysis["cpu"].get("affinity_applied")),
                    ("Affinity cores", analysis["cpu"].get("affinity_cores")),
                    ("Expected max total CPU percent", analysis["cpu"].get("expected_max_total_cpu_percent")),
                    ("Observed peak process CPU percent", analysis["cpu"].get("observed_peak_process_cpu_percent")),
                ],
            ),
            _html_table_card(
                "Memory",
                [
                    ("Mode", analysis["memory"].get("mode")),
                    ("Requested memory GB", analysis["memory"].get("requested_memory_gb")),
                    ("Effective budget MB", analysis["memory"].get("effective_budget_mb")),
                    ("Peak memory MB", analysis["memory"].get("peak_memory_mb")),
                    ("Min available memory MB", analysis["memory"].get("observed_min_available_memory_mb")),
                    ("Memory budget exceeded", analysis["memory"].get("memory_budget_exceeded")),
                ],
            ),
            _html_table_card(
                "Cgroup",
                [
                    ("Available", analysis["cgroup"].get("available")),
                    ("Path", analysis["cgroup"].get("path")),
                    ("Peak cgroup memory MB", analysis["cgroup"].get("peak_memory_mb")),
                    ("Peak cgroup CPU percent", analysis["cgroup"].get("peak_cpu_percent")),
                ],
            ),
            "<section class=\"card\">",
            "<h2>Workload</h2>",
            f"<table><tbody>{workload_items}</tbody></table>",
            "</section>",
            "<section class=\"card\">",
            "<h2>Diagnostics</h2>",
            f"<ul class=\"diagnostics\">{diagnostic_items}</ul>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


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


def _html_table_card(title: str, rows: list[tuple[str, object]]) -> str:
    body = "".join(
        f"<tr><th>{_html_escape(label)}</th><td>{_html_escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"<section class=\"card\"><h2>{_html_escape(title)}</h2><table><tbody>{body}</tbody></table></section>"


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _run_report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f7fb;
  --card: #ffffff;
  --ink: #102033;
  --muted: #5b6b7d;
  --border: #d8e2ec;
  --accent: #0f766e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top right, #d1fae5 0, transparent 24rem),
    linear-gradient(180deg, #fbfffd 0%, var(--bg) 100%);
}
.page {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
.hero { padding: 8px 4px 20px; }
.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 12px;
}
h1, h2 { margin: 0 0 12px; }
h1 { font-size: 38px; line-height: 1.05; }
h2 { font-size: 22px; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
  padding: 20px;
  margin-bottom: 18px;
}
.card svg {
  width: 100%;
  height: auto;
  display: block;
  margin-top: 10px;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  width: 280px;
  color: var(--muted);
  font-weight: 600;
}
.diagnostics {
  margin: 0;
  padding-left: 20px;
  color: var(--muted);
}
@media (max-width: 720px) {
  .page { padding: 20px 12px 40px; }
  h1 { font-size: 30px; }
  .card { padding: 14px; }
  th { width: 42%; }
}
""".strip()
