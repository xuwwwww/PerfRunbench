from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import atomic_write_json


REPORT_SCHEMA_VERSION = "1.0"
_INK = "#0f172a"
_MUTED = "#64748b"
_GRID = "#dbe3ee"
_BASELINE = "#2563eb"
_SELECTED = "#dc2626"
_ACCENT = "#059669"


def generate_arm_inference_report(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("kind") != "c7g4xlarge_realistic_arm_inference_validation":
        raise ValueError("input must be a c7g4xlarge_realistic_arm_inference_validation")
    root = Path(output_dir) if output_dir is not None else source.with_name(f"{source.stem}_visuals")
    charts = root / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    summary = build_arm_inference_visual_summary(data)
    thread_rows = summary["thread_scaling"]
    comparison_rows = summary["serving_comparisons"]
    chart_payloads = {
        "thread_scaling.svg": _thread_scaling_svg(thread_rows),
        "serving_before_after.svg": _serving_before_after_svg(comparison_rows),
        "paired_confirmation.svg": _paired_confirmation_svg(comparison_rows),
        "ttft_p95.svg": _ttft_svg(comparison_rows),
        "guard_summary.svg": _guard_summary_svg(summary),
    }
    for name, content in chart_payloads.items():
        (charts / name).write_text(content, encoding="utf-8", newline="\n")

    summary_json = root / "arm_inference_summary.json"
    thread_csv = root / "thread_scaling.csv"
    comparison_csv = root / "serving_comparisons.csv"
    html_path = root / "arm_inference_report.html"
    atomic_write_json(summary_json, summary)
    _write_thread_csv(thread_csv, thread_rows)
    _write_comparison_csv(comparison_csv, comparison_rows)
    html_path.write_text(
        format_arm_inference_report_html(
            summary,
            chart_names=list(chart_payloads),
            source_name=source.name,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "report": str(html_path),
        "summary": str(summary_json),
        "thread_scaling_csv": str(thread_csv),
        "serving_comparisons_csv": str(comparison_csv),
        "charts": str(charts),
    }


def build_arm_inference_visual_summary(data: dict[str, Any]) -> dict[str, Any]:
    thread_rows = [
        {
            "threads": int(item["threads"]),
            "test": str(item["test"]),
            "avg_tokens_per_second": _number(item.get("avg_tokens_per_second")),
            "stddev_tokens_per_second": _number(item.get("stddev_tokens_per_second")),
        }
        for item in data.get("llama_bench_scaling", {}).get("rows", [])
        if isinstance(item, dict)
        and _number(item.get("avg_tokens_per_second")) is not None
    ]
    comparisons: list[dict[str, Any]] = []
    comparisons.extend(
        _comparison_rows(
            "7-CPU capacity match / c4",
            data.get("capacity_matched_7cpu", {}),
        )
    )
    for concurrency, analysis in sorted(data.get("full_capacity", {}).items()):
        if isinstance(analysis, dict):
            comparisons.extend(
                _comparison_rows(
                    f"15-CPU full capacity / c{concurrency}",
                    analysis,
                )
            )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "arm_inference_visual_summary",
        "result_origin": data.get("result_origin"),
        "status": data.get("status"),
        "target": data.get("target", {}),
        "source_commit": data.get("source_commit"),
        "llama_cpp_commit": data.get("llama_cpp_commit"),
        "model_sha256": data.get("model_sha256"),
        "candidate_count_including_baseline": data.get("candidate_count_including_baseline"),
        "server_startups": data.get("server_startups"),
        "measured_http_requests": data.get("measured_http_requests"),
        "thread_scaling": thread_rows,
        "serving_comparisons": comparisons,
        "recommendation_count": len(data.get("recommendations", [])),
        "performance_passport_status": data.get("performance_passport_status"),
        "cleanup_passed": data.get("cleanup_passed"),
        "remote_artifacts_checked": data.get("remote_artifacts_checked"),
        "remote_artifacts_failed": data.get("remote_artifacts_failed"),
        "claim_boundary": data.get("claim_boundary"),
    }


def format_arm_inference_report_html(
    summary: dict[str, Any],
    *,
    chart_names: list[str],
    source_name: str,
) -> str:
    comparisons = summary["serving_comparisons"]
    eligible = [item for item in comparisons if item["throughput_eligible"] or item["latency_eligible"]]
    best = max(
        (
            item["ratio_of_medians_improvement_percent"]
            for item in eligible
            if item["ratio_of_medians_improvement_percent"] is not None
        ),
        default=None,
    )
    chart_sections = "\n".join(
        f'<section class="panel"><h2>{escape(_chart_title(name))}</h2>'
        f'<img src="charts/{escape(name)}" alt="{escape(_chart_title(name))}"></section>'
        for name in chart_names
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(item['stage'])}</td>"
        f"<td>{escape(item['workload_id'])}</td>"
        f"<td>{_fmt(item['baseline_output_tokens_per_second_median'])}</td>"
        f"<td>{_fmt(item['candidate_output_tokens_per_second_median'])}</td>"
        f"<td>{_fmt(item['ratio_of_medians_improvement_percent'])}%</td>"
        f"<td>{'eligible' if item['throughput_eligible'] else 'not eligible'}</td>"
        "</tr>"
        for item in comparisons
    ) or '<tr><td colspan="6">No measured serving comparisons.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PerfRunbench Arm Inference Evidence</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#dbe3ee;--blue:#2563eb;--red:#dc2626;--green:#059669}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--ink);font-family:Segoe UI,Arial,sans-serif}}
.page{{max-width:1180px;margin:0 auto;padding:32px 28px 48px}}.hero{{border-bottom:3px solid var(--ink);padding-bottom:22px;margin-bottom:24px}}
.eyebrow{{font-size:13px;font-weight:700;text-transform:uppercase;color:var(--blue);margin:0 0 8px}}
h1{{font-size:42px;line-height:1.08;margin:0 0 12px}}h2{{font-size:21px;margin:0 0 14px}}
.lede,.note{{color:var(--muted);max-width:940px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:22px}}
.stat,.panel{{border:1px solid var(--line);border-radius:8px;background:#fff}}.stat{{padding:14px}}
.label{{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;margin-bottom:6px}}.value{{font-size:21px;font-weight:700;word-break:break-word}}
.panel{{padding:18px;margin-bottom:16px}}.panel img{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}
th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted);text-transform:uppercase}}
code{{font-family:Cascadia Mono,Consolas,monospace;font-size:.92em}}@media(max-width:760px){{.page{{padding:20px 12px}}.stats{{grid-template-columns:1fr 1fr}}h1{{font-size:32px}}}}
</style>
</head>
<body>
<main class="page">
<header class="hero">
<p class="eyebrow">PerfRunbench measured Arm64 evidence</p>
<h1>Realistic Arm inference tuning</h1>
<p class="lede">Pinned Qwen2.5-7B Q4_K_M and llama.cpp evidence on one C7g.4xlarge. Thread scaling, capacity matching, and runtime tuning remain separate protocols.</p>
</header>
<section class="stats">
<div class="stat"><span class="label">Best eligible delta</span><span class="value">{_fmt(best)}%</span></div>
<div class="stat"><span class="label">Measured requests</span><span class="value">{summary.get("measured_http_requests")}</span></div>
<div class="stat"><span class="label">Candidates</span><span class="value">{summary.get("candidate_count_including_baseline")}</span></div>
<div class="stat"><span class="label">Passport</span><span class="value">{escape(str(summary.get("performance_passport_status")))}</span></div>
</section>
<section class="panel">
<h2>Evidence boundary</h2>
<table><tbody>
<tr><th>Source</th><td><code>{escape(source_name)}</code></td></tr>
<tr><th>Origin</th><td><code>{escape(str(summary.get("result_origin")))}</code></td></tr>
<tr><th>Target</th><td><code>{escape(str(summary.get("target", {}).get("instance_class")))}</code> / {escape(str(summary.get("target", {}).get("cpu")))}</td></tr>
<tr><th>Topology</th><td><code>{escape(str(summary.get("target", {}).get("topology")))}</code></td></tr>
<tr><th>Cleanup</th><td><code>{escape(str(summary.get("cleanup_passed")))}</code></td></tr>
<tr><th>Claim boundary</th><td>{escape(str(summary.get("claim_boundary")))}</td></tr>
</tbody></table>
<p class="note">Hardware scaling percentages and runtime-tuning percentages are never added together. Loopback evidence is not remote-network evidence.</p>
</section>
{chart_sections}
<section class="panel">
<h2>Measured serving comparisons</h2>
<table><thead><tr><th>Protocol</th><th>Workload</th><th>Baseline tok/s</th><th>Selected tok/s</th><th>Delta</th><th>Guard</th></tr></thead>
<tbody>{comparison_rows}</tbody></table>
</section>
</main>
</body>
</html>
"""


def _comparison_rows(stage: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.get("comparisons", []):
        if not isinstance(item, dict):
            continue
        rows.append({
            "stage": stage,
            "workload_id": str(item.get("workload_id")),
            "pair_count": int(item.get("pair_count") or 0),
            "baseline_output_tokens_per_second_median": _number(
                item.get("baseline_output_tokens_per_second_median")
            ),
            "candidate_output_tokens_per_second_median": _number(
                item.get("candidate_output_tokens_per_second_median")
            ),
            "ratio_of_medians_improvement_percent": _number(
                item.get("ratio_of_medians_improvement_percent")
            ),
            "paired_improvement_percent": item.get("paired_improvement_percent", {}),
            "baseline_ttft_p95_ms": _number(item.get("baseline_ttft_p95_ms_block_median")),
            "candidate_ttft_p95_ms": _number(item.get("candidate_ttft_p95_ms_block_median")),
            "ttft_p95_reduction_percent": _number(item.get("ttft_p95_reduction_percent")),
            "baseline_by_pair": item.get("baseline_output_tokens_per_second_by_pair", {}),
            "candidate_by_pair": item.get("candidate_output_tokens_per_second_by_pair", {}),
            "bootstrap": item.get("bootstrap", {}),
            "guards_passed": item.get("guards_passed") is True,
            "cleanup_passed": item.get("cleanup_passed") is True,
            "throughput_eligible": item.get("throughput_eligible") is True,
            "latency_eligible": item.get("latency_eligible") is True,
        })
    return rows


def _thread_scaling_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1100, 600
    left, right, top, bottom = 110, 1030, 110, 500
    by_test: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_test.setdefault(row["test"], []).append(
            (row["threads"], row["avg_tokens_per_second"])
        )
    values = [value for points in by_test.values() for _, value in points]
    high = max(values, default=1.0) * 1.12
    thread_values = sorted({thread for points in by_test.values() for thread, _ in points})

    def x(thread: int) -> float:
        if len(thread_values) <= 1:
            return (left + right) / 2
        return left + thread_values.index(thread) / (len(thread_values) - 1) * (right - left)

    def y(value: float) -> float:
        return bottom - value / high * (bottom - top)

    parts = _svg_start(width, height, "llama-bench thread scaling")
    parts.extend(_heading("llama-bench thread scaling", "Separate prefill and decode protocol; higher is better."))
    for tick in range(6):
        value = high * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{_fmt(value)}</text>')
    colors = [_BASELINE, _SELECTED, _ACCENT]
    for index, (test, points) in enumerate(sorted(by_test.items())):
        points = sorted(points)
        color = colors[index % len(colors)]
        path = " ".join(
            f"{'M' if point_index == 0 else 'L'} {x(thread):.2f} {y(value):.2f}"
            for point_index, (thread, value) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"/>')
        for thread, value in points:
            parts.append(f'<circle cx="{x(thread):.2f}" cy="{y(value):.2f}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{right - index * 150}" y="82" text-anchor="end" fill="{color}" class="legend">{escape(test)}</text>')
    for thread in thread_values:
        parts.append(f'<text x="{x(thread):.2f}" y="{bottom + 34}" text-anchor="middle" class="axis">{thread}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="570" text-anchor="middle" class="note">server threads</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _serving_before_after_svg(rows: list[dict[str, Any]]) -> str:
    plot_rows = [
        row
        for row in rows
        if row["baseline_output_tokens_per_second_median"] is not None
        and row["candidate_output_tokens_per_second_median"] is not None
    ]
    width = 1100
    height = max(460, 150 + len(plot_rows) * 72)
    left, right = 360, 1030
    high = max(
        (
            max(
                row["baseline_output_tokens_per_second_median"],
                row["candidate_output_tokens_per_second_median"],
            )
            for row in plot_rows
        ),
        default=1.0,
    ) * 1.12
    parts = _svg_start(width, height, "Serving before and after")
    parts.extend(_heading("Serving before and after", "Each row compares one paired protocol; runtime and capacity stages stay separate."))
    for index, row in enumerate(plot_rows):
        y = 135 + index * 72
        baseline_x = left + row["baseline_output_tokens_per_second_median"] / high * (right - left)
        selected_x = left + row["candidate_output_tokens_per_second_median"] / high * (right - left)
        label = f"{row['stage']} / {row['workload_id']}"
        parts.append(f'<text x="24" y="{y + 5}" class="row-label">{escape(label)}</text>')
        parts.append(f'<line x1="{baseline_x:.2f}" y1="{y}" x2="{selected_x:.2f}" y2="{y}" stroke="#94a3b8" stroke-width="4"/>')
        parts.append(f'<circle cx="{baseline_x:.2f}" cy="{y}" r="8" fill="{_BASELINE}"/>')
        parts.append(f'<circle cx="{selected_x:.2f}" cy="{y}" r="8" fill="{_SELECTED}"/>')
        parts.append(f'<text x="{baseline_x:.2f}" y="{y - 15}" text-anchor="middle" class="value">{_fmt(row["baseline_output_tokens_per_second_median"])}</text>')
        parts.append(f'<text x="{selected_x:.2f}" y="{y + 28}" text-anchor="middle" class="value">{_fmt(row["candidate_output_tokens_per_second_median"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _paired_confirmation_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1100, 620
    values = []
    for row in rows:
        paired = row.get("paired_improvement_percent", {})
        if _number(paired.get("median")) is not None:
            values.append((f"{row['stage']} / {row['workload_id']}", float(paired["median"])))
    parts = _svg_start(width, height, "Paired confirmation deltas")
    parts.extend(_heading("Paired confirmation deltas", "Median paired throughput delta; the dashed line is the 5% noise guard."))
    top, bottom, left, right = 120, 520, 250, 1020
    low = min([0.0, *(value for _, value in values)], default=0.0) - 5
    high = max([10.0, *(value for _, value in values)], default=10.0) + 5

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    guard_y = y(5.0)
    parts.append(f'<line x1="{left}" y1="{guard_y:.2f}" x2="{right}" y2="{guard_y:.2f}" stroke="{_ACCENT}" stroke-dasharray="8 8" stroke-width="3"/>')
    count = max(len(values), 1)
    for index, (label, value) in enumerate(values):
        x = left + (index + 0.5) / count * (right - left)
        zero_y = y(0.0)
        value_y = y(value)
        parts.append(f'<line x1="{x:.2f}" y1="{zero_y:.2f}" x2="{x:.2f}" y2="{value_y:.2f}" stroke="{_SELECTED}" stroke-width="28"/>')
        parts.append(f'<text x="{x:.2f}" y="{value_y - 12:.2f}" text-anchor="middle" class="value">{_fmt(value)}%</text>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 30}" text-anchor="middle" class="axis">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _ttft_svg(rows: list[dict[str, Any]]) -> str:
    values = [
        row
        for row in rows
        if row["baseline_ttft_p95_ms"] is not None
        and row["candidate_ttft_p95_ms"] is not None
    ]
    width = 1100
    height = max(460, 150 + len(values) * 72)
    high = max(
        (
            max(row["baseline_ttft_p95_ms"], row["candidate_ttft_p95_ms"])
            for row in values
        ),
        default=1.0,
    ) * 1.1
    left, right = 360, 1030
    parts = _svg_start(width, height, "p95 TTFT before and after")
    parts.extend(_heading("p95 TTFT before and after", "Block-level p95 medians; lower is better."))
    for index, row in enumerate(values):
        y = 135 + index * 72
        label = f"{row['stage']} / {row['workload_id']}"
        base_width = row["baseline_ttft_p95_ms"] / high * (right - left)
        selected_width = row["candidate_ttft_p95_ms"] / high * (right - left)
        parts.append(f'<text x="24" y="{y + 6}" class="row-label">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y - 22}" width="{base_width:.2f}" height="18" fill="{_BASELINE}" opacity=".85"/>')
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{selected_width:.2f}" height="18" fill="{_SELECTED}" opacity=".9"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _guard_summary_svg(summary: dict[str, Any]) -> str:
    checks = [
        ("Measured origin", summary.get("result_origin") == "measured"),
        ("Run status", summary.get("status") == "passed"),
        ("Cleanup", summary.get("cleanup_passed") is True),
        ("Artifact validation", summary.get("remote_artifacts_failed") == 0),
        ("Recommendation", summary.get("recommendation_count", 0) > 0),
    ]
    parts = _svg_start(1100, 430, "Evidence guards")
    parts.extend(_heading("Evidence guards", "A performance claim remains ineligible when a required guard fails."))
    for index, (label, passed) in enumerate(checks):
        x = 60 + index * 205
        color = _ACCENT if passed else _MUTED
        status = "PASSED" if passed else "NOT PASSED"
        parts.append(f'<rect x="{x}" y="150" width="175" height="120" rx="8" fill="#fff" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 87.5}" y="198" text-anchor="middle" class="row-label">{escape(label)}</text>')
        parts.append(f'<text x="{x + 87.5}" y="242" text-anchor="middle" fill="{color}" class="status">{status}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _write_thread_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "threads",
                "test",
                "avg_tokens_per_second",
                "stddev_tokens_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage",
        "workload_id",
        "pair_count",
        "baseline_output_tokens_per_second_median",
        "candidate_output_tokens_per_second_median",
        "ratio_of_medians_improvement_percent",
        "baseline_ttft_p95_ms",
        "candidate_ttft_p95_ms",
        "ttft_p95_reduction_percent",
        "guards_passed",
        "cleanup_passed",
        "throughput_eligible",
        "latency_eligible",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>"
        "text{font-family:Segoe UI,Arial,sans-serif;fill:#0f172a;letter-spacing:0}"
        ".title{font-size:28px;font-weight:700}.subtitle{font-size:15px;fill:#64748b}"
        ".axis{font-size:12px;fill:#64748b}.note{font-size:14px;fill:#64748b}"
        ".legend{font-size:14px;font-weight:700}.row-label{font-size:14px;font-weight:600}"
        ".value{font-size:12px;font-weight:700}.status{font-size:15px;font-weight:700}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _heading(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="24" y="42" class="title">{escape(title)}</text>',
        f'<text x="24" y="70" class="subtitle">{escape(subtitle)}</text>',
    ]


def _chart_title(name: str) -> str:
    return {
        "thread_scaling.svg": "llama-bench thread scaling",
        "serving_before_after.svg": "Serving throughput before and after",
        "paired_confirmation.svg": "Paired confirmation deltas",
        "ttft_p95.svg": "p95 TTFT before and after",
        "guard_summary.svg": "Evidence guards",
    }.get(name, name)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:,.3f}"
