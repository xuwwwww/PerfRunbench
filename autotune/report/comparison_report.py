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
    if output_path.suffix.lower() == ".html":
        content = format_comparison_report_html(data, source)
    else:
        content = format_comparison_report(data, source)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def format_comparison_report(data: dict[str, Any], source: Path | None = None) -> str:
    if data.get("kind") == "profile_comparison_summary":
        return _format_profile_summary_report(data, source)
    title = _report_title(data)
    baseline_label = _baseline_label(data)
    tuned_label = _tuned_label(data)
    baseline = _preferred_group(data, "baseline")
    tuned = _preferred_group(data, "tuned")
    deltas = _preferred_group(data, "deltas", aggregate=False)
    aggregate = data.get("aggregate", {})
    if aggregate:
        deltas = aggregate.get("deltas", deltas)
    lines = [
        f"# AutoTuneAI {title}",
        "",
        f"- Source: `{source}`" if source else "- Source: in-memory comparison",
        f"- Profile: `{data.get('tuned_profile')}`",
        f"- Repeat: {data.get('repeat', 1)}",
        f"- {baseline_label.title()} run: `{_run_ids(baseline)}`",
        f"- {tuned_label.title()} run: `{_run_ids(tuned)}`",
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
                (f"{baseline_label} benchmark sec", baseline.get("benchmark_duration_seconds")),
                (f"{tuned_label} benchmark sec", tuned.get("benchmark_duration_seconds")),
                (f"{baseline_label} lifecycle sec", baseline.get("lifecycle_duration_seconds")),
                (f"{tuned_label} lifecycle sec", tuned.get("lifecycle_duration_seconds")),
            ],
            unit="s",
        ),
        "",
        metric_bar_chart(
            "Resource Peaks",
            [
                (f"{baseline_label} peak memory MB", baseline.get("peak_memory_mb")),
                (f"{tuned_label} peak memory MB", tuned.get("peak_memory_mb")),
                (f"{baseline_label} system CPU %", baseline.get("peak_system_cpu_percent")),
                (f"{tuned_label} system CPU %", tuned.get("peak_system_cpu_percent")),
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


def format_comparison_report_html(data: dict[str, Any], source: Path | None = None) -> str:
    if data.get("kind") == "profile_comparison_summary":
        return _format_profile_summary_report_html(data, source)
    title = _report_title(data)
    baseline_label = _baseline_label(data)
    tuned_label = _tuned_label(data)
    baseline = _preferred_group(data, "baseline")
    tuned = _preferred_group(data, "tuned")
    deltas = _preferred_group(data, "deltas", aggregate=False)
    aggregate = data.get("aggregate", {})
    if aggregate:
        deltas = aggregate.get("deltas", deltas)
    diagnostics = [*baseline.get("diagnostics", []), *tuned.get("diagnostics", [])]
    diagnostic_items = "".join(
        f"<li>{_html_escape(item)}</li>" for item in dict.fromkeys(diagnostics)
    ) or "<li>No diagnostics emitted.</li>"
    summary_rows = [
        ("Source", str(source) if source else "in-memory comparison"),
        ("Profile", str(data.get("tuned_profile"))),
        ("Repeat", str(data.get("repeat", 1))),
        (f"{baseline_label.title()} run", _run_ids(baseline)),
        (f"{tuned_label.title()} run", _run_ids(tuned)),
    ]
    summary_html = "".join(
        f"<div class=\"stat\"><span class=\"label\">{_html_escape(label)}</span><span class=\"value\">{_html_escape(value)}</span></div>"
        for label, value in summary_rows
    )
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>AutoTuneAI {title}</title>",
            "<style>",
            _comparison_report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main class=\"page\">",
            "<section class=\"hero\">",
            "<p class=\"eyebrow\">AutoTuneAI</p>",
            f"<h1>{_html_escape(title)}</h1>",
            "<p class=\"lede\">Performance-first comparison with inline charts and repeat-aware aggregates.</p>",
            "</section>",
            f"<section class=\"summary-grid\">{summary_html}</section>",
            "<section class=\"card\">",
            "<h2>Key Deltas</h2>",
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
            "</section>",
            "<section class=\"card\">",
            "<h2>Baseline vs Tuned</h2>",
            metric_bar_chart(
                "Durations",
                [
                    (f"{baseline_label} benchmark sec", baseline.get("benchmark_duration_seconds")),
                    (f"{tuned_label} benchmark sec", tuned.get("benchmark_duration_seconds")),
                    (f"{baseline_label} lifecycle sec", baseline.get("lifecycle_duration_seconds")),
                    (f"{tuned_label} lifecycle sec", tuned.get("lifecycle_duration_seconds")),
                ],
                unit="s",
            ),
            metric_bar_chart(
                "Resource Peaks",
                [
                    (f"{baseline_label} peak memory MB", baseline.get("peak_memory_mb")),
                    (f"{tuned_label} peak memory MB", tuned.get("peak_memory_mb")),
                    (f"{baseline_label} system CPU %", baseline.get("peak_system_cpu_percent")),
                    (f"{tuned_label} system CPU %", tuned.get("peak_system_cpu_percent")),
                ],
            ),
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


def _format_profile_summary_report(data: dict[str, Any], source: Path | None = None) -> str:
    comparisons = data.get("comparisons", [])
    lines = [
        "# AutoTuneAI Profile Comparison Summary",
        "",
        f"- Source: `{source}`" if source else "- Source: in-memory comparison",
        f"- Repeat: {data.get('repeat', 1)}",
        f"- Best profile: `{data.get('best_profile')}`",
        f"- Best profile beats baseline: `{data.get('best_profile_beats_baseline')}`",
        "",
        "## Throughput Ranking",
        "",
        metric_bar_chart(
            "samples/sec delta %",
            [(str(item.get("profile")), item.get("samples_per_second_percent")) for item in comparisons],
        ),
        "",
        "## Benchmark Duration Ranking",
        "",
        metric_bar_chart(
            "benchmark duration delta %",
            [(str(item.get("profile")), item.get("benchmark_duration_percent")) for item in comparisons],
        ),
        "",
        "## Profiles",
        "",
    ]
    if comparisons:
        for item in comparisons:
            lines.append(
                "- "
                f"{item.get('profile')}: samples/sec {item.get('samples_per_second_percent')}%, "
                f"benchmark duration {item.get('benchmark_duration_percent')}%, "
                f"peak memory {item.get('peak_memory_percent')}%, "
                f"output `{item.get('output')}`"
            )
    else:
        lines.append("- No profile comparisons recorded.")
    return "\n".join(lines) + "\n"


def _format_profile_summary_report_html(data: dict[str, Any], source: Path | None = None) -> str:
    comparisons = data.get("comparisons", [])
    summary_rows = [
        ("Source", str(source) if source else "in-memory comparison"),
        ("Repeat", str(data.get("repeat", 1))),
        ("Best profile", str(data.get("best_profile"))),
        ("Beats baseline", str(data.get("best_profile_beats_baseline"))),
    ]
    summary_html = "".join(
        f"<div class=\"stat\"><span class=\"label\">{_html_escape(label)}</span><span class=\"value\">{_html_escape(value)}</span></div>"
        for label, value in summary_rows
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(item.get('profile'))}</td>"
        f"<td>{_html_escape(item.get('samples_per_second_percent'))}</td>"
        f"<td>{_html_escape(item.get('benchmark_duration_percent'))}</td>"
        f"<td>{_html_escape(item.get('peak_memory_percent'))}</td>"
        f"<td>{_html_escape(item.get('output'))}</td>"
        "</tr>"
        for item in comparisons
    ) or "<tr><td colspan=\"5\">No profile comparisons recorded.</td></tr>"
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>AutoTuneAI Profile Comparison Summary</title>",
            "<style>",
            _comparison_report_css(),
            "table{width:100%;border-collapse:collapse}th,td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left}th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}",
            "</style>",
            "</head>",
            "<body>",
            "<main class=\"page\">",
            "<section class=\"hero\">",
            "<p class=\"eyebrow\">AutoTuneAI</p>",
            "<h1>Profile Comparison Summary</h1>",
            "<p class=\"lede\">Ranked profile sweep with throughput-first comparison outputs.</p>",
            "</section>",
            f"<section class=\"summary-grid\">{summary_html}</section>",
            "<section class=\"card\">",
            "<h2>Throughput Ranking</h2>",
            metric_bar_chart(
                "samples/sec delta %",
                [(str(item.get("profile")), item.get("samples_per_second_percent")) for item in comparisons],
            ),
            "</section>",
            "<section class=\"card\">",
            "<h2>Benchmark Duration Ranking</h2>",
            metric_bar_chart(
                "benchmark duration delta %",
                [(str(item.get("profile")), item.get("benchmark_duration_percent")) for item in comparisons],
            ),
            "</section>",
            "<section class=\"card\">",
            "<h2>Profiles</h2>",
            "<table><thead><tr><th>Profile</th><th>Samples/sec %</th><th>Benchmark duration %</th><th>Peak memory %</th><th>JSON</th></tr></thead>",
            f"<tbody>{table_rows}</tbody></table>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


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


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _report_title(data: dict[str, Any]) -> str:
    kind = data.get("kind")
    if kind == "budget_mode_comparison":
        return "Budget Mode Comparison"
    return "Tuning Comparison"


def _baseline_label(data: dict[str, Any]) -> str:
    return str(data.get("baseline_label") or "baseline")


def _tuned_label(data: dict[str, Any]) -> str:
    return str(data.get("tuned_label") or "tuned")


def _comparison_report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f3f7fb;
  --card: #ffffff;
  --ink: #102033;
  --muted: #5b6b7d;
  --border: #d8e2ec;
  --accent: #1d4ed8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, #dbeafe 0, transparent 26rem),
    linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
}
.page {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
.hero {
  padding: 8px 4px 20px;
}
.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 12px;
}
h1, h2 {
  margin: 0 0 12px;
}
h1 {
  font-size: 40px;
  line-height: 1.05;
}
h2 {
  font-size: 22px;
}
.lede {
  margin: 0;
  color: var(--muted);
  max-width: 760px;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin: 8px 0 18px;
}
.stat, .card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
.stat {
  padding: 16px 18px;
}
.label {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.value {
  display: block;
  font-size: 16px;
  font-weight: 600;
  word-break: break-word;
}
.card {
  padding: 20px;
  margin-bottom: 18px;
}
.card svg {
  width: 100%;
  height: auto;
  display: block;
  margin-top: 10px;
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
}
""".strip()
