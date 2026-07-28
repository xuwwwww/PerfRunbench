from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from statistics import median
from typing import Any

from autotune.recommendation.schemas import atomic_write_json


REPORT_SCHEMA_VERSION = "1.0"
_BASELINE = "#2563eb"
_SELECTED = "#dc2626"
_MUTED = "#64748b"
_GRID = "#e2e8f0"
_INK = "#0f172a"
_SURFACE = "#f8fafc"


def generate_training_tuning_report(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    cloudwatch_path: str | Path | None = None,
) -> dict[str, str]:
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("kind") != "pytorch_training_tuning_result":
        raise ValueError("input must be a pytorch_training_tuning_result")
    root = Path(output_dir) if output_dir is not None else source.with_name(f"{source.stem}_visuals")
    charts = root / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    trials = _valid_trials(data.get("trials"))
    pairs = _confirmation_pairs(trials, str(data.get("selected_candidate_id") or "selected"))
    summary = build_training_tuning_summary(data, trials=trials, pairs=pairs)
    cloudwatch = _load_cloudwatch(cloudwatch_path)

    chart_payloads = {
        "candidate_ranking.svg": _candidate_ranking_svg(data, trials),
        "paired_before_after.svg": _paired_before_after_svg(pairs, summary),
        "throughput_distribution.svg": _distribution_svg(pairs, summary),
        "step_time_distribution.svg": _step_time_svg(pairs),
        "guard_summary.svg": _guard_summary_svg(data),
    }
    if cloudwatch:
        chart_payloads["cpu_credit_timeline.svg"] = _credit_timeline_svg(cloudwatch)
    for name, content in chart_payloads.items():
        (charts / name).write_text(content, encoding="utf-8", newline="\n")

    trials_csv = root / "training_trials.csv"
    pairs_csv = root / "paired_confirmation.csv"
    summary_json = root / "training_summary.json"
    html_path = root / "training_tuning_report.html"
    _write_trials_csv(trials_csv, trials)
    _write_pairs_csv(pairs_csv, pairs)
    atomic_write_json(summary_json, summary)
    html_path.write_text(
        format_training_tuning_report_html(
            data,
            summary=summary,
            chart_names=list(chart_payloads),
            source=source,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "report": str(html_path),
        "summary": str(summary_json),
        "trials_csv": str(trials_csv),
        "pairs_csv": str(pairs_csv),
        "charts": str(charts),
    }


def build_training_tuning_summary(
    data: dict[str, Any],
    *,
    trials: list[dict[str, Any]] | None = None,
    pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    valid_trials = trials if trials is not None else _valid_trials(data.get("trials"))
    paired = pairs if pairs is not None else _confirmation_pairs(valid_trials, str(data.get("selected_candidate_id") or "selected"))
    baseline = [item["baseline_samples_per_second"] for item in paired]
    selected = [item["selected_samples_per_second"] for item in paired]
    deltas = [item["improvement_percent"] for item in paired]
    baseline_steps = [item["baseline_step_time_ms"] for item in paired if item["baseline_step_time_ms"] is not None]
    selected_steps = [item["selected_step_time_ms"] for item in paired if item["selected_step_time_ms"] is not None]
    evidence = data.get("statistical_evidence", {})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "pytorch_training_visual_summary",
        "source_run_id": data.get("run_id"),
        "result_origin": data.get("result_origin"),
        "status": data.get("status"),
        "classification": evidence.get("classification") if isinstance(evidence, dict) else None,
        "selected_candidate_id": data.get("selected_candidate_id"),
        "candidate_count": len(data.get("candidates", [])),
        "valid_trial_count": len(valid_trials),
        "failed_trial_count": len([item for item in data.get("trials", []) if isinstance(item, dict) and item.get("failed") is True]),
        "paired_confirmation_count": len(paired),
        "baseline_samples_per_second": _distribution_summary(baseline),
        "selected_samples_per_second": _distribution_summary(selected),
        "paired_improvement_percent": _distribution_summary(deltas),
        "baseline_step_time_ms": _distribution_summary(baseline_steps),
        "selected_step_time_ms": _distribution_summary(selected_steps),
        "noise_band_percent": evidence.get("noise_band_percent") if isinstance(evidence, dict) else None,
        "selected_variability_percent": evidence.get("selected_variability_percent") if isinstance(evidence, dict) else None,
        "search_budget": data.get("search_budget", {}),
        "claim_boundary": (
            "Best observed eligible configuration under the tested training workload, "
            "slice protocol, and measurement protocol."
        ),
    }


def format_training_tuning_report_html(
    data: dict[str, Any],
    *,
    summary: dict[str, Any],
    chart_names: list[str],
    source: Path | None = None,
) -> str:
    baseline = summary["baseline_samples_per_second"]
    selected = summary["selected_samples_per_second"]
    improvement = summary["paired_improvement_percent"]
    chart_sections = "\n".join(
        f'<section class="panel"><h2>{escape(_chart_title(name))}</h2>'
        f'<img src="charts/{escape(name)}" alt="{escape(_chart_title(name))}"></section>'
        for name in chart_names
    )
    guards = _guard_results(data)
    guard_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('stage') or 'unknown'))}</td>"
        f"<td>{escape(str(item.get('code') or 'unknown'))}</td>"
        f"<td>{escape(str(item.get('status') or 'unknown'))}</td>"
        f"<td>{escape(str(item.get('message') or ''))}</td>"
        "</tr>"
        for item in guards
    ) or '<tr><td colspan="4">No guard results recorded.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PerfRunbench Training Tuning Evidence</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#dbe3ee;--surface:#f8fafc;--blue:#2563eb;--red:#dc2626}}
*{{box-sizing:border-box}} body{{margin:0;background:#fff;color:var(--ink);font-family:Segoe UI,Arial,sans-serif}}
.page{{max-width:1180px;margin:0 auto;padding:32px 28px 48px}}
.hero{{border-bottom:3px solid var(--ink);padding-bottom:22px;margin-bottom:24px}}
.eyebrow{{font-size:13px;font-weight:700;text-transform:uppercase;color:var(--blue);margin:0 0 8px}}
h1{{font-size:42px;line-height:1.08;margin:0 0 12px}} h2{{font-size:21px;margin:0 0 14px}}
.lede,.note{{color:var(--muted);max-width:920px}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:22px}}
.stat,.panel{{border:1px solid var(--line);border-radius:8px;background:#fff}}
.stat{{padding:14px}} .label{{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;margin-bottom:6px}}
.value{{font-size:21px;font-weight:700;word-break:break-word}} .panel{{padding:18px;margin-bottom:16px}}
.panel img{{display:block;width:100%;height:auto}} table{{width:100%;border-collapse:collapse}}
th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase}}
code{{font-family:Cascadia Mono,Consolas,monospace;font-size:.92em}}
@media(max-width:760px){{.page{{padding:20px 12px}}.stats{{grid-template-columns:1fr 1fr}}h1{{font-size:32px}}}}
</style>
</head>
<body>
<main class="page">
<header class="hero">
<p class="eyebrow">PerfRunbench measured evidence</p>
<h1>Training tuning before and after</h1>
<p class="lede">Repeated paired confirmation from one bounded PyTorch training protocol. Search trials, warm-up trials, failures, guards, and cleanup evidence remain separate.</p>
</header>
<section class="stats">
<div class="stat"><span class="label">Baseline median</span><span class="value">{_fmt(baseline.get("median"))} samples/s</span></div>
<div class="stat"><span class="label">Selected median</span><span class="value">{_fmt(selected.get("median"))} samples/s</span></div>
<div class="stat"><span class="label">Paired median delta</span><span class="value">{_fmt(improvement.get("median"))}%</span></div>
<div class="stat"><span class="label">Paired confirmations</span><span class="value">{summary.get("paired_confirmation_count")}</span></div>
</section>
<section class="panel">
<h2>Evidence boundary</h2>
<table><tbody>
<tr><th>Source</th><td><code>{escape(source.name if source else "in-memory")}</code></td></tr>
<tr><th>Run ID</th><td><code>{escape(str(summary.get("source_run_id")))}</code></td></tr>
<tr><th>Result origin</th><td><code>{escape(str(summary.get("result_origin")))}</code></td></tr>
<tr><th>Status</th><td><code>{escape(str(summary.get("status")))}</code></td></tr>
<tr><th>Selected candidate</th><td><code>{escape(str(summary.get("selected_candidate_id")))}</code></td></tr>
<tr><th>Noise band</th><td>{_fmt(summary.get("noise_band_percent"))}%</td></tr>
<tr><th>Claim boundary</th><td>{escape(str(summary.get("claim_boundary")))}</td></tr>
</tbody></table>
<p class="note">A result from one T4g.small and a synthetic CPU PyTorch workload is not a general PyTorch or Arm speedup claim. C7g and llama.cpp evidence use different protocols and must not be combined with this result.</p>
</section>
{chart_sections}
<section class="panel">
<h2>Statistical guards</h2>
<table><thead><tr><th>Stage</th><th>Code</th><th>Status</th><th>Message</th></tr></thead><tbody>{guard_rows}</tbody></table>
</section>
</main>
</body>
</html>
"""


def _valid_trials(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, dict)
        and item.get("trial_status") == "valid"
        and item.get("failed") is not True
        and _finite(item.get("objective_metric_value"))
    ]


def _confirmation_pairs(trials: list[dict[str, Any]], selected_candidate_id: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for trial in trials:
        pair_id = trial.get("pair_id")
        if trial.get("phase") != "confirmation" or not isinstance(pair_id, str):
            continue
        role = "baseline" if trial.get("candidate_id") == "baseline" else "selected"
        if role == "selected" and trial.get("candidate_id") != selected_candidate_id:
            continue
        grouped.setdefault(pair_id, {})[role] = trial
    pairs = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        baseline = pair.get("baseline")
        selected = pair.get("selected")
        if not baseline or not selected:
            continue
        baseline_value = float(baseline["objective_metric_value"])
        selected_value = float(selected["objective_metric_value"])
        pairs.append(
            {
                "pair_id": pair_id,
                "baseline_samples_per_second": baseline_value,
                "selected_samples_per_second": selected_value,
                "improvement_percent": ((selected_value - baseline_value) / abs(baseline_value)) * 100.0,
                "baseline_step_time_ms": _metric(baseline, "step_time_ms"),
                "selected_step_time_ms": _metric(selected, "step_time_ms"),
                "baseline_final_loss": _metric(baseline, "final_loss"),
                "selected_final_loss": _metric(selected, "final_loss"),
            }
        )
    return pairs


def _candidate_ranking_svg(data: dict[str, Any], trials: list[dict[str, Any]]) -> str:
    candidates = [item for item in data.get("candidates", []) if isinstance(item, dict)]
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "unknown")
        values = [
            float(item["objective_metric_value"])
            for item in trials
            if item.get("phase") == "search" and item.get("candidate_id") == candidate_id
        ]
        if values:
            rows.append((_candidate_chart_label(candidate, candidate_id), values))
    rows.sort(key=lambda item: median(item[1]), reverse=True)
    return _horizontal_range_chart("Candidate search throughput", rows, "samples/s")


def _paired_before_after_svg(pairs: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    width, height = 1100, 620
    top, bottom = 120, 500
    baseline = [item["baseline_samples_per_second"] for item in pairs]
    selected = [item["selected_samples_per_second"] for item in pairs]
    values = [*baseline, *selected]
    low, high = _bounds(values)

    def y(value: float) -> float:
        return bottom - ((value - low) / (high - low)) * (bottom - top)

    parts = _svg_start(width, height, "Paired before and after")
    parts.extend(_chart_heading("Paired before and after", "Each line is one fresh baseline/selected confirmation pair."))
    parts.append(f'<line x1="290" y1="{top}" x2="290" y2="{bottom}" stroke="{_GRID}" stroke-width="2"/>')
    parts.append(f'<line x1="810" y1="{top}" x2="810" y2="{bottom}" stroke="{_GRID}" stroke-width="2"/>')
    for item in pairs:
        by = y(item["baseline_samples_per_second"])
        sy = y(item["selected_samples_per_second"])
        parts.append(f'<line x1="290" y1="{by:.2f}" x2="810" y2="{sy:.2f}" stroke="#94a3b8" stroke-width="2" opacity=".7"/>')
        parts.append(f'<circle cx="290" cy="{by:.2f}" r="6" fill="{_BASELINE}"/>')
        parts.append(f'<circle cx="810" cy="{sy:.2f}" r="6" fill="{_SELECTED}"/>')
    parts.append(f'<text x="290" y="540" text-anchor="middle" class="axis">Baseline</text>')
    parts.append(f'<text x="810" y="540" text-anchor="middle" class="axis">Selected</text>')
    parts.append(
        f'<text x="550" y="580" text-anchor="middle" class="note">'
        f'{len(pairs)} pairs | median delta {_fmt(summary["paired_improvement_percent"].get("median"))}%</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _distribution_svg(pairs: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    baseline = [item["baseline_samples_per_second"] for item in pairs]
    selected = [item["selected_samples_per_second"] for item in pairs]
    return _box_plot_svg(
        "Confirmation throughput distribution",
        [("Baseline", baseline, _BASELINE), ("Selected", selected, _SELECTED)],
        "samples/s",
        f'noise band {summary.get("noise_band_percent")}% | all valid confirmation samples shown',
    )


def _step_time_svg(pairs: list[dict[str, Any]]) -> str:
    baseline = [item["baseline_step_time_ms"] for item in pairs if item["baseline_step_time_ms"] is not None]
    selected = [item["selected_step_time_ms"] for item in pairs if item["selected_step_time_ms"] is not None]
    return _box_plot_svg(
        "Confirmation step-time distribution",
        [("Baseline", baseline, _BASELINE), ("Selected", selected, _SELECTED)],
        "ms/step",
        "Lower is better; values come from the same paired runs as throughput.",
    )


def _guard_summary_svg(data: dict[str, Any]) -> str:
    evidence = data.get("statistical_evidence", {})
    guards = _guard_results(data)
    rows = [
        ("Measured provenance", "pass" if data.get("result_origin") == "measured" else "fail"),
        ("Fresh paired evidence", "pass" if evidence.get("paired_confirmation_count", 0) else "fail"),
        ("Minimum samples", _guard_status(guards, "min_valid_trials")),
        ("Outside noise band", _guard_status(guards, "improvement_outside_noise_band")),
        ("Consistent direction", _guard_status(guards, "consistent_confirmation_direction")),
        ("Variability threshold", _guard_status(guards, "variability_threshold")),
        ("Failed trials", "pass" if not evidence.get("failed_trials") else "warn"),
        ("Search budget", "pass" if data.get("search_budget", {}).get("status") == "completed" else "warn"),
    ]
    width, height = 1100, 160 + 58 * len(rows)
    parts = _svg_start(width, height, "Statistical and execution guards")
    parts.extend(_chart_heading("Statistical and execution guards", "Eligibility is earned from evidence; unsupported or inconclusive states remain visible."))
    colors = {"pass": "#15803d", "warn": "#a16207", "fail": "#b91c1c", "not_available": _MUTED}
    for index, (label, status) in enumerate(rows):
        y = 140 + index * 58
        parts.append(f'<text x="80" y="{y + 28}" class="label">{escape(label)}</text>')
        parts.append(f'<rect x="760" y="{y}" width="240" height="38" rx="6" fill="{colors.get(status, _MUTED)}"/>')
        parts.append(f'<text x="880" y="{y + 26}" text-anchor="middle" class="badge">{escape(status)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _credit_timeline_svg(cloudwatch: dict[str, list[dict[str, Any]]]) -> str:
    balance = [
        (str(item.get("timestamp")), float(item["value"]))
        for item in cloudwatch.get("CPUCreditBalance", [])
        if _finite(item.get("value"))
    ]
    surplus = [
        (str(item.get("timestamp")), float(item["value"]))
        for item in cloudwatch.get("CPUSurplusCreditBalance", [])
        if _finite(item.get("value"))
    ]
    charged = [
        float(item["value"])
        for item in cloudwatch.get("CPUSurplusCreditsCharged", [])
        if _finite(item.get("value"))
    ]
    plotted = [
        ("CPUCreditBalance", balance, _BASELINE),
        ("CPUSurplusCreditBalance", surplus, _SELECTED),
    ]
    values = [value for _label, series, _color in plotted for _timestamp, value in series]
    width, height = 1100, 520
    parts = _svg_start(width, height, "T4g CPU credit timeline")
    parts.extend(
        _chart_heading(
            "T4g CPU credit timeline",
            "CloudWatch balance and surplus balance; five-minute granularity on basic monitoring.",
        )
    )
    if not values:
        parts.append('<text x="80" y="230" class="note">No CPU credit datapoints were supplied.</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    low, high = _bounds(values, include_zero=True)
    left, right, top, bottom = 90, 1020, 130, 410
    for label, series, color in plotted:
        if not series:
            continue
        points = []
        for index, (_timestamp, value) in enumerate(series):
            x = left + (right - left) * index / max(1, len(series) - 1)
            y = bottom - ((value - low) / (high - low)) * (bottom - top)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4"/>')
        legend_x = 600 if label == "CPUCreditBalance" else 815
        parts.append(f'<line x1="{legend_x}" y1="64" x2="{legend_x + 32}" y2="64" stroke="{color}" stroke-width="5"/>')
        parts.append(f'<text x="{legend_x + 40}" y="70" class="note">{escape(label)}</text>')
    balance_first = balance[0][1] if balance else None
    balance_last = balance[-1][1] if balance else None
    surplus_first = surplus[0][1] if surplus else None
    surplus_last = surplus[-1][1] if surplus else None
    parts.append(
        f'<text x="{left}" y="455" class="note">'
        f'balance {_fmt(balance_first)} to {_fmt(balance_last)} | '
        f'surplus {_fmt(surplus_first)} to {_fmt(surplus_last)} | '
        f'charged samples total {_fmt(sum(charged))}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _horizontal_range_chart(title: str, rows: list[tuple[str, list[float]]], unit: str) -> str:
    width = 1100
    height = 160 + max(1, len(rows)) * 74
    values = [value for _label, samples in rows for value in samples]
    low, high = _bounds(values, include_zero=True)
    left, plot_right, value_x = 350, 820, 1010
    parts = _svg_start(width, height, title)
    parts.extend(_chart_heading(title, "Median bar with min/max whisker across valid search trials."))
    if not rows:
        parts.append('<text x="80" y="200" class="note">No valid search trials.</text>')
    for index, (label, samples) in enumerate(rows):
        y = 140 + index * 74
        scale = lambda value: left + ((value - low) / (high - low)) * (plot_right - left)
        minimum, maximum, med = min(samples), max(samples), median(samples)
        parts.append(f'<text x="70" y="{y + 30}" class="label">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{max(1, scale(med) - left):.2f}" height="34" rx="6" fill="{_BASELINE}"/>')
        parts.append(f'<line x1="{scale(minimum):.2f}" y1="{y + 23}" x2="{scale(maximum):.2f}" y2="{y + 23}" stroke="{_INK}" stroke-width="3"/>')
        parts.append(f'<text x="{value_x}" y="{y + 30}" text-anchor="end" class="value">{_fmt(med)} {unit}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _box_plot_svg(title: str, groups: list[tuple[str, list[float], str]], unit: str, note: str) -> str:
    width, height = 1100, 560
    values = [value for _label, samples, _color in groups for value in samples]
    low, high = _bounds(values)
    top, bottom = 130, 420
    parts = _svg_start(width, height, title)
    parts.extend(_chart_heading(title, note))
    for index, (label, samples, color) in enumerate(groups):
        if not samples:
            continue
        x = 360 + index * 390
        stats = _distribution_summary(samples)
        scale = lambda value: bottom - ((float(value) - low) / (high - low)) * (bottom - top)
        y_min, y_q1, y_med, y_q3, y_max = [scale(stats[key]) for key in ("min", "q1", "median", "q3", "max")]
        parts.append(f'<line x1="{x}" y1="{y_min:.2f}" x2="{x}" y2="{y_max:.2f}" stroke="{_INK}" stroke-width="3"/>')
        parts.append(f'<rect x="{x - 75}" y="{y_q3:.2f}" width="150" height="{max(2, y_q1 - y_q3):.2f}" fill="{color}" opacity=".22" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<line x1="{x - 75}" y1="{y_med:.2f}" x2="{x + 75}" y2="{y_med:.2f}" stroke="{color}" stroke-width="5"/>')
        for value in samples:
            parts.append(f'<circle cx="{x}" cy="{scale(value):.2f}" r="5" fill="{color}" opacity=".65"/>')
        parts.append(f'<text x="{x}" y="475" text-anchor="middle" class="axis">{escape(label)}</text>')
        parts.append(f'<text x="{x}" y="510" text-anchor="middle" class="note">median {_fmt(stats["median"])} {escape(unit)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _write_trials_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "phase",
        "role",
        "run_index",
        "pair_id",
        "samples_per_second",
        "step_time_ms",
        "final_loss",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "candidate_id": trial.get("candidate_id"),
                    "phase": trial.get("phase"),
                    "role": trial.get("role"),
                    "run_index": trial.get("run_index"),
                    "pair_id": trial.get("pair_id"),
                    "samples_per_second": trial.get("objective_metric_value"),
                    "step_time_ms": _metric(trial, "step_time_ms"),
                    "final_loss": _metric(trial, "final_loss"),
                    "run_dir": trial.get("run_dir"),
                }
            )


def _write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = list(pairs[0]) if pairs else [
        "pair_id",
        "baseline_samples_per_second",
        "selected_samples_per_second",
        "improvement_percent",
        "baseline_step_time_ms",
        "selected_step_time_ms",
        "baseline_final_loss",
        "selected_final_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)


def _load_cloudwatch(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    return payload if isinstance(payload, dict) else {}


def _guard_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = data.get("statistical_evidence", {})
    raw = evidence.get("guard_results", []) if isinstance(evidence, dict) else []
    return [item for item in raw if isinstance(item, dict)]


def _guard_status(guards: list[dict[str, Any]], code: str) -> str:
    match = next((item for item in guards if item.get("code") == code), None)
    return str(match.get("status")) if match else "not_available"


def _metric(trial: dict[str, Any], name: str) -> float | None:
    metrics = trial.get("metrics", {})
    value = metrics.get(name) if isinstance(metrics, dict) else None
    return float(value) if _finite(value) else None


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    numeric = sorted(float(value) for value in values if _finite(value))
    if not numeric:
        return {"count": 0, "min": None, "q1": None, "median": None, "q3": None, "p95": None, "max": None}
    return {
        "count": len(numeric),
        "min": numeric[0],
        "q1": _percentile(numeric, 25),
        "median": median(numeric),
        "q3": _percentile(numeric, 75),
        "p95": _percentile(numeric, 95),
        "max": numeric[-1],
    }


def _percentile(values: list[float], percent: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (percent / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _bounds(values: list[float], *, include_zero: bool = False) -> tuple[float, float]:
    numeric = [float(value) for value in values if _finite(value)]
    if not numeric:
        return 0.0, 1.0
    low = min(numeric)
    high = max(numeric)
    if include_zero and low > 0:
        low = 0.0
    padding = (high - low) * 0.08
    if padding == 0:
        padding = abs(high) * 0.08 or 1.0
    return low - padding, high + padding


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<rect width="100%" height="100%" fill="{_SURFACE}"/>',
        "<style>.title{font:700 30px Segoe UI,Arial,sans-serif;fill:#0f172a}.subtitle{font:16px Segoe UI,Arial,sans-serif;fill:#64748b}.label{font:600 16px Segoe UI,Arial,sans-serif;fill:#0f172a}.value{font:700 15px Segoe UI,Arial,sans-serif;fill:#0f172a}.axis{font:700 19px Segoe UI,Arial,sans-serif;fill:#0f172a}.note{font:15px Segoe UI,Arial,sans-serif;fill:#64748b}.badge{font:700 16px Segoe UI,Arial,sans-serif;fill:#fff}</style>",
    ]


def _chart_heading(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="70" y="58" class="title">{escape(title)}</text>',
        f'<text x="70" y="91" class="subtitle">{escape(subtitle)}</text>',
    ]


def _chart_title(name: str) -> str:
    return Path(name).stem.replace("_", " ").title()


def _candidate_chart_label(candidate: dict[str, Any], candidate_id: str) -> str:
    knobs = candidate.get("knob_values", {})
    if not isinstance(knobs, dict):
        return candidate_id
    batch = knobs.get("micro_batch_size")
    accumulation = knobs.get("gradient_accumulation")
    workers = knobs.get("dataloader_workers")
    threads = knobs.get("cpu_threads")
    if candidate_id == "baseline" and batch is not None and accumulation is not None:
        return f"baseline: batch {batch} / accumulation {accumulation}"
    if candidate.get("stage") == "batch_config" and batch is not None and accumulation is not None:
        return f"batch {batch} / accumulation {accumulation}"
    if candidate.get("stage") == "input_pipeline" and workers is not None:
        return f"workers {workers}"
    if candidate.get("stage") == "cpu_runtime" and threads is not None:
        return f"OMP threads {threads}"
    return candidate_id


def _fmt(value: Any) -> str:
    return "n/a" if not _finite(value) else f"{float(value):,.3f}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
