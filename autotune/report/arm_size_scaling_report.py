from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

from autotune.recommendation.schemas import atomic_write_json


REPORT_SCHEMA_VERSION = "1.0"
_INSTANCE_SPECS = {
    "c7g.xlarge": {"vcpu": 4, "memory_gib": 8},
    "c7g.2xlarge": {"vcpu": 8, "memory_gib": 16},
    "c7g.4xlarge": {"vcpu": 16, "memory_gib": 32},
}
_INSTANCE_ORDER = {name: index for index, name in enumerate(_INSTANCE_SPECS)}
_BLUE = "#2563eb"
_RED = "#dc2626"
_GREEN = "#059669"
_AMBER = "#d97706"
_INK = "#0f172a"
_MUTED = "#64748b"
_GRID = "#dbe3ee"


def generate_arm_size_scaling_report(
    input_paths: Sequence[str | Path],
    revalidation_paths: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, str]:
    if len(input_paths) != len(revalidation_paths):
        raise ValueError("each input requires one controller revalidation")
    if len(input_paths) < 2:
        raise ValueError("at least two measured instance results are required")

    results = [_load_json(path) for path in input_paths]
    revalidations = [_load_json(path) for path in revalidation_paths]
    summary = build_arm_size_scaling_summary(results, revalidations)

    root = Path(output_dir)
    charts = root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    chart_payloads = {
        "thread_scaling_across_sizes.svg": _thread_scaling_svg(summary),
        "serving_before_after_by_size.svg": _serving_before_after_svg(summary),
        "absolute_throughput_by_size.svg": _absolute_throughput_svg(summary),
        "ttft_before_after_by_size.svg": _ttft_svg(summary),
        "workload_guard_matrix.svg": _guard_matrix_svg(summary),
        "parallel_efficiency.svg": _parallel_efficiency_svg(summary),
    }
    for name, content in chart_payloads.items():
        (charts / name).write_text(content, encoding="utf-8", newline="\n")

    summary_path = root / "c7g_size_scaling_validation.json"
    thread_csv = root / "thread_scaling.csv"
    serving_csv = root / "serving_comparisons.csv"
    report_path = root / "c7g_size_scaling_report.html"
    atomic_write_json(summary_path, summary)
    _write_csv(thread_csv, summary["thread_scaling"])
    _write_csv(serving_csv, summary["serving_comparisons"])
    report_path.write_text(
        format_arm_size_scaling_report_html(summary, chart_payloads),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "thread_scaling_csv": str(thread_csv),
        "serving_comparisons_csv": str(serving_csv),
        "charts": str(charts),
    }


def build_arm_size_scaling_summary(
    results: Sequence[dict[str, Any]],
    revalidations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(results) != len(revalidations):
        raise ValueError("each result requires one controller revalidation")

    identities: dict[str, set[str]] = {
        key: set()
        for key in (
            "source_commit",
            "llama_cpp_commit",
            "model_sha256",
            "runtime_bundle_sha256",
            "bench_binary_sha256",
            "server_binary_sha256",
        )
    }
    targets: list[dict[str, Any]] = []
    thread_rows: list[dict[str, Any]] = []
    serving_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []

    for result, revalidation in zip(results, revalidations):
        _validate_result(result)
        _validate_revalidation(revalidation)
        instance_class = str(result["target"]["instance_class"])
        if any(item["instance_class"] == instance_class for item in targets):
            raise ValueError(f"duplicate instance result: {instance_class}")
        spec = _INSTANCE_SPECS[instance_class]
        logical_cpus = int(result["target"]["logical_cpus"])
        if logical_cpus != spec["vcpu"]:
            raise ValueError(f"{instance_class} logical CPU count does not match its public specification")

        for key in identities:
            identities[key].add(str(result[key]))

        target = {
            "instance_class": instance_class,
            "cpu": str(result["target"]["cpu"]),
            "logical_cpus": logical_cpus,
            "memory_gib": spec["memory_gib"],
            "client_cpus": 1,
            "server_threads": logical_cpus - 1,
            "topology": str(result["target"]["topology"]),
            "candidate_count_including_baseline": int(
                result["candidate_count_including_baseline"]
            ),
            "measured_http_requests": int(result["measured_http_requests"]),
            "server_startups": int(result["server_startups"]),
            "performance_passport_status": str(result["performance_passport_status"]),
            "cleanup_passed": result["cleanup_passed"] is True,
            "host_restored": result["host_restored"] is True,
            "remote_artifacts_checked": int(result["remote_artifacts_checked"]),
            "remote_artifacts_failed": int(result["remote_artifacts_failed"]),
            "controller_artifacts_checked": int(revalidation["checked"]),
            "controller_artifacts_failed": int(revalidation["failed"]),
            "archive_sha256": str(revalidation["archive_sha256"]),
        }
        targets.append(target)

        for row in result["llama_bench_scaling"]["rows"]:
            thread_rows.append(
                {
                    "instance_class": instance_class,
                    "threads": int(row["threads"]),
                    "test": str(row["test"]),
                    "avg_tokens_per_second": _required_number(
                        row.get("avg_tokens_per_second"),
                        "avg_tokens_per_second",
                    ),
                    "stddev_tokens_per_second": _required_number(
                        row.get("stddev_tokens_per_second"),
                        "stddev_tokens_per_second",
                    ),
                    "measured_repetitions": int(
                        result["llama_bench_scaling"]["measured_repetitions"]
                    ),
                }
            )

        decision_by_workload = {
            str(item["workload_id"]): item
            for item in result["screening_decisions"]
        }
        for workload_id, decision in decision_by_workload.items():
            candidates = decision.get("candidates", [])
            for candidate in candidates:
                guard_rows.append(
                    {
                        "instance_class": instance_class,
                        "workload_id": workload_id,
                        "profile_id": str(candidate["profile_id"]),
                        "throughput_improvement_percent": _required_number(
                            candidate.get("throughput_improvement_percent"),
                            "throughput_improvement_percent",
                        ),
                        "screening_eligible": candidate.get("screening_eligible") is True,
                        "guard_reasons": [
                            str(reason) for reason in candidate.get("guard_reasons", [])
                        ],
                    }
                )

        groups = result["paired_confirmation"].get("groups", [])
        for group in groups:
            profile_id = str(group["profile_id"])
            for comparison in group["analysis"].get("comparisons", []):
                serving_rows.append(
                    _serving_row(
                        instance_class,
                        profile_id,
                        comparison,
                    )
                )

    for key, values in identities.items():
        if len(values) != 1:
            raise ValueError(f"cross-instance identity mismatch: {key}")

    targets.sort(key=lambda item: _INSTANCE_ORDER[item["instance_class"]])
    thread_rows.sort(
        key=lambda item: (
            _INSTANCE_ORDER[item["instance_class"]],
            item["threads"],
            item["test"],
        )
    )
    serving_rows.sort(
        key=lambda item: (
            _INSTANCE_ORDER[item["instance_class"]],
            item["workload_id"],
        )
    )
    guard_rows.sort(
        key=lambda item: (
            _INSTANCE_ORDER[item["instance_class"]],
            item["workload_id"],
            item["profile_id"],
        )
    )

    total_requests = sum(item["measured_http_requests"] for item in targets)
    total_starts = sum(item["server_startups"] for item in targets)
    total_controller_artifacts = sum(
        item["controller_artifacts_checked"] for item in targets
    )
    identity = {key: next(iter(values)) for key, values in identities.items()}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "c7g_size_scaling_validation",
        "result_origin": "measured",
        "status": "passed",
        "identity": identity,
        "protocol": {
            "platform": "Linux aarch64 / AWS Graviton3 / Neoverse-V1",
            "model": "Qwen2.5-7B Instruct Q4_K_M",
            "llama_bench": {
                "prompt_tokens": 512,
                "generation_tokens": 128,
                "warmup_processes_per_thread_point": 1,
                "measured_repetitions_per_thread_point": 5,
            },
            "llama_server": {
                "topology": "single-host loopback",
                "client_cpu_count": 1,
                "concurrency": 4,
                "screening_measured_requests_per_slice": 12,
                "paired_confirmation_pairs": 3,
                "paired_measured_requests_per_workload_block": 20,
                "server_restart_between_blocks": True,
                "retries": 0,
                "noise_band_percent": 5,
                "bootstrap_samples": 10000,
            },
        },
        "targets": targets,
        "thread_scaling": thread_rows,
        "serving_comparisons": serving_rows,
        "screening_guards": guard_rows,
        "evidence_totals": {
            "measured_http_requests": total_requests,
            "server_startups": total_starts,
            "controller_artifacts_revalidated": total_controller_artifacts,
            "controller_artifact_failures": 0,
            "failed_http_requests": 0,
            "quality_failures": 0,
            "cleanup_passed_on_all_targets": True,
            "host_restored_on_all_targets": True,
        },
        "recommendation": {
            "profile_family": "four parallel slots with continuous batching",
            "eligible_workloads": ["decode-heavy", "interactive-short"],
            "rejected_workloads": ["prefill-heavy"],
            "wording": (
                "Best observed eligible configuration under each tested Arm target, "
                "model, inference workload, and measurement protocol."
            ),
        },
        "glossary": {
            "prefill": "Processing input prompt tokens before generation begins.",
            "decode": "Generating output tokens after prompt processing.",
            "ttft": "Time to first token, measured from request start to the first streamed token.",
            "p95": "The value at or below which 95% of observations fall.",
            "continuous_batching": "Scheduling active requests together as token generation progresses.",
            "noise_band": "A minimum change threshold used to avoid recommending measurement noise.",
            "bootstrap_interval": "A resampled uncertainty interval computed from paired measurements.",
            "parallel_efficiency": "Observed throughput divided by ideal linear throughput at the same thread count.",
            "performance_passport": "A versioned evidence record binding results to target, model, runtime, protocol, and guards.",
        },
        "claim_boundary": (
            "Measured results apply only to the tested C7g sizes, pinned model and "
            "runtime, reserved loopback client CPU, concurrency-4 workloads, and "
            "declared protocols. Hardware-size scaling and runtime-tuning percentages "
            "are reported separately and are never added. Prefill-heavy serving "
            "regressions were rejected by the guard."
        ),
    }


def format_arm_size_scaling_report_html(
    summary: dict[str, Any],
    chart_payloads: dict[str, str],
) -> str:
    targets = summary["targets"]
    comparisons = summary["serving_comparisons"]
    target_rows = "".join(
        "<tr>"
        f"<td><strong>{escape(item['instance_class'])}</strong></td>"
        f"<td>{item['logical_cpus']}</td>"
        f"<td>{item['memory_gib']} GiB</td>"
        f"<td>{item['server_threads']}</td>"
        f"<td>{item['measured_http_requests']}</td>"
        f"<td>{item['controller_artifacts_checked']} / {item['controller_artifacts_failed']}</td>"
        "</tr>"
        for item in targets
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(item['instance_class'])}</td>"
        f"<td>{escape(item['workload_id'])}</td>"
        f"<td>{_fmt(item['baseline_output_tokens_per_second_median'])}</td>"
        f"<td>{_fmt(item['candidate_output_tokens_per_second_median'])}</td>"
        f"<td>{_fmt(item['ratio_of_medians_improvement_percent'])}%</td>"
        f"<td>{_fmt(item['baseline_ttft_p95_ms'])}</td>"
        f"<td>{_fmt(item['candidate_ttft_p95_ms'])}</td>"
        f"<td>{'passed' if item['throughput_eligible'] else 'rejected'}</td>"
        "</tr>"
        for item in comparisons
    )
    glossary_rows = "".join(
        f"<dt>{escape(term.replace('_', ' '))}</dt><dd>{escape(description)}</dd>"
        for term, description in summary["glossary"].items()
    )
    chart_sections = "\n".join(
        f'<section class="band"><h2>{escape(_chart_title(name))}</h2>'
        f'<img src="charts/{escape(name)}" alt="{escape(_chart_title(name))}"></section>'
        for name in chart_payloads
    )
    total = summary["evidence_totals"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PerfRunbench C7g Size-Scaling Evidence</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#dbe3ee;--blue:#2563eb;--red:#dc2626;--green:#059669;--paper:#ffffff;--wash:#f8fafc}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}}
.page{{max-width:1220px;margin:0 auto;padding:34px 30px 54px}}.hero{{border-bottom:4px solid var(--ink);padding-bottom:24px;margin-bottom:24px}}
.eyebrow{{margin:0 0 8px;color:var(--blue);font-size:13px;font-weight:700;text-transform:uppercase}}
h1{{margin:0 0 12px;font-size:44px;line-height:1.08}}h2{{font-size:23px;margin:0 0 14px}}h3{{font-size:17px;margin:0 0 8px}}
.lede,.note{{color:var(--muted);max-width:980px;line-height:1.55}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}}
.metric{{border-top:4px solid var(--blue);padding:13px 2px 8px}}.metric strong{{display:block;font-size:27px}}.metric span{{font-size:12px;color:var(--muted);text-transform:uppercase}}
.band{{border-top:1px solid var(--line);padding:24px 0}}.band img{{display:block;width:100%;height:auto}}.guard{{border-left:5px solid var(--green);padding:14px 18px;background:var(--wash)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase}}code{{font-family:Cascadia Mono,Consolas,monospace;font-size:.92em}}
dl{{display:grid;grid-template-columns:190px 1fr;gap:10px 18px}}dt{{font-weight:700;text-transform:uppercase;font-size:12px}}dd{{margin:0;color:var(--muted);line-height:1.45}}
@media(max-width:800px){{.page{{padding:22px 14px}}.metrics{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}table{{font-size:13px}}dl{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main class="page">
<header class="hero">
<p class="eyebrow">PerfRunbench measured Arm64 evidence</p>
<h1>One workload, three C7g sizes, guarded tuning</h1>
<p class="lede">The same pinned Qwen2.5-7B Q4_K_M model and llama.cpp runtime were measured on C7g.xlarge, C7g.2xlarge, and C7g.4xlarge. A fixed protocol separates hardware scaling from runtime tuning.</p>
</header>
<section class="metrics">
<div class="metric"><strong>3</strong><span>Graviton3 sizes</span></div>
<div class="metric"><strong>{total['measured_http_requests']}</strong><span>Measured HTTP requests</span></div>
<div class="metric"><strong>{total['server_startups']}</strong><span>Fresh server starts</span></div>
<div class="metric"><strong>{total['controller_artifacts_revalidated']}</strong><span>Artifacts revalidated</span></div>
</section>
<section class="band">
<h2>Evidence boundary</h2>
<p class="note">{escape(summary['claim_boundary'])}</p>
<table><thead><tr><th>Instance</th><th>vCPU</th><th>Memory</th><th>Server threads</th><th>HTTP requests</th><th>Controller digest checks</th></tr></thead>
<tbody>{target_rows}</tbody></table>
</section>
{chart_sections}
<section class="band">
<h2>Formal paired confirmation</h2>
<table><thead><tr><th>Instance</th><th>Workload</th><th>Baseline tok/s</th><th>Tuned tok/s</th><th>Delta</th><th>Baseline p95 TTFT ms</th><th>Tuned p95 TTFT ms</th><th>Guard</th></tr></thead>
<tbody>{comparison_rows}</tbody></table>
</section>
<section class="band">
<h2>Why the guard matters</h2>
<div class="guard"><strong>Prefill-heavy was not recommended.</strong> Four-slot serving increased interactive and decode throughput, but prefill-heavy screening regressed on all three sizes and triggered the TTFT guard.</div>
</section>
<section class="band">
<h2>Glossary</h2>
<dl>{glossary_rows}</dl>
</section>
</main>
</body>
</html>
"""


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("input schema_version is unsupported")
    if result.get("kind") != "c7g_size_scaling_comparative_result":
        raise ValueError("input must be a c7g_size_scaling_comparative_result")
    if result.get("result_origin") != "measured" or result.get("status") != "passed":
        raise ValueError("all inputs must be passed measured results")
    target = result.get("target", {})
    instance_class = str(target.get("instance_class"))
    if instance_class not in _INSTANCE_SPECS:
        raise ValueError(f"unsupported C7g size: {instance_class}")
    if target.get("cpu") != "Neoverse-V1":
        raise ValueError("target CPU identity must be Neoverse-V1")
    if target.get("topology") != "single_host_loopback":
        raise ValueError("target topology must be single_host_loopback")
    if result.get("cleanup_passed") is not True or result.get("host_restored") is not True:
        raise ValueError(f"{instance_class} cleanup or restoration did not pass")
    if int(result.get("remote_artifacts_failed") or 0) != 0:
        raise ValueError(f"{instance_class} has remote artifact failures")
    if result.get("failures"):
        raise ValueError(f"{instance_class} has recorded failures")
    if result.get("performance_passport_status") != "eligible_preliminary":
        raise ValueError(f"{instance_class} passport is not eligible_preliminary")
    if int(result.get("candidate_count_including_baseline") or 0) < 2:
        raise ValueError(f"{instance_class} has no measured candidate")
    if int(result.get("measured_http_requests") or 0) <= 0:
        raise ValueError(f"{instance_class} has no measured HTTP requests")
    if int(result.get("server_startups") or 0) <= 0:
        raise ValueError(f"{instance_class} has no fresh server starts")
    if int(result.get("remote_artifacts_checked") or 0) <= 0:
        raise ValueError(f"{instance_class} has no remote artifact validation")

    logical_cpus = int(target.get("logical_cpus") or 0)
    expected_profile = f"slots4-cont-on-threads{logical_cpus - 1}"
    recommendations = result.get("recommendations", [])
    if len(recommendations) != 1:
        raise ValueError(f"{instance_class} must have exactly one recommendation")
    recommendation = recommendations[0]
    if recommendation.get("profile_id") != expected_profile:
        raise ValueError(f"{instance_class} recommendation profile does not match CPU allocation")
    if set(recommendation.get("eligible_workloads", [])) != {
        "decode-heavy",
        "interactive-short",
    }:
        raise ValueError(f"{instance_class} recommendation workload scope is invalid")

    scaling = result.get("llama_bench_scaling", {})
    rows = scaling.get("rows", [])
    if not rows:
        raise ValueError(f"{instance_class} has no llama-bench scaling rows")
    if int(scaling.get("measured_repetitions") or 0) != 5:
        raise ValueError(f"{instance_class} llama-bench repetition count is invalid")
    row_keys = {(int(row["threads"]), str(row["test"])) for row in rows}
    required_row_keys = {
        (1, "pp512"),
        (1, "tg128"),
        (logical_cpus - 1, "pp512"),
        (logical_cpus - 1, "tg128"),
    }
    if not required_row_keys.issubset(row_keys):
        raise ValueError(f"{instance_class} llama-bench scaling endpoints are incomplete")
    if any(
        int(row["threads"]) < 1
        or int(row["threads"]) > logical_cpus - 1
        or str(row["test"]) not in {"pp512", "tg128"}
        for row in rows
    ):
        raise ValueError(f"{instance_class} llama-bench scaling row is invalid")

    decisions = result.get("screening_decisions", [])
    if {str(item.get("workload_id")) for item in decisions} != {
        "decode-heavy",
        "interactive-short",
        "prefill-heavy",
    }:
        raise ValueError(f"{instance_class} screening workload set is invalid")
    if any(int(item.get("concurrency") or 0) != 4 for item in decisions):
        raise ValueError(f"{instance_class} screening concurrency is invalid")
    prefill = next(item for item in decisions if item["workload_id"] == "prefill-heavy")
    prefill_candidate = next(
        (
            candidate
            for candidate in prefill.get("candidates", [])
            if candidate.get("profile_id") == expected_profile
        ),
        None,
    )
    if (
        prefill_candidate is None
        or prefill_candidate.get("screening_eligible") is not False
        or _required_number(
            prefill_candidate.get("throughput_improvement_percent"),
            "prefill-heavy.throughput_improvement_percent",
        )
        >= -5.0
        or "ttft_regression" not in prefill_candidate.get("guard_reasons", [])
    ):
        raise ValueError(f"{instance_class} prefill-heavy rejection evidence is invalid")

    groups = result.get("paired_confirmation", {}).get("groups", [])
    if len(groups) != 1:
        raise ValueError(f"{instance_class} must have exactly one paired confirmation group")
    group = groups[0]
    if group.get("profile_id") != expected_profile:
        raise ValueError(f"{instance_class} paired profile does not match recommendation")
    comparisons = group.get("analysis", {}).get("comparisons", [])
    if {str(item.get("workload_id")) for item in comparisons} != {
        "decode-heavy",
        "interactive-short",
    }:
        raise ValueError(f"{instance_class} formal comparison workload set is invalid")
    if not comparisons:
        raise ValueError(f"{instance_class} has no paired confirmation")
    for comparison in comparisons:
        bootstrap = comparison.get("bootstrap", {})
        improvement = _required_number(
            comparison.get("ratio_of_medians_improvement_percent"),
            "ratio_of_medians_improvement_percent",
        )
        if (
            int(comparison.get("pair_count") or 0) < 3
            or improvement < 5.0
            or comparison.get("guards_passed") is not True
            or comparison.get("cleanup_passed") is not True
            or comparison.get("throughput_eligible") is not True
            or comparison.get("latency_eligible") is not True
            or bootstrap.get("excludes_zero") is not True
            or _required_number(
                bootstrap.get("p05_percent"),
                "bootstrap.p05_percent",
            )
            <= 0
        ):
            raise ValueError(f"{instance_class} formal paired comparison is not eligible")


def _validate_revalidation(revalidation: dict[str, Any]) -> None:
    if revalidation.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("controller revalidation schema_version is unsupported")
    if int(revalidation.get("checked") or 0) <= 0:
        raise ValueError("controller revalidation did not check any artifacts")
    if int(revalidation.get("failed") or 0) != 0 or revalidation.get("failures"):
        raise ValueError("controller artifact revalidation failed")
    digest = str(revalidation.get("archive_sha256") or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("controller revalidation archive digest is invalid")


def _serving_row(
    instance_class: str,
    profile_id: str,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = comparison.get("bootstrap", {})
    return {
        "instance_class": instance_class,
        "profile_id": profile_id,
        "workload_id": str(comparison["workload_id"]),
        "pair_count": int(comparison["pair_count"]),
        "baseline_output_tokens_per_second_median": _required_number(
            comparison.get("baseline_output_tokens_per_second_median"),
            "baseline_output_tokens_per_second_median",
        ),
        "candidate_output_tokens_per_second_median": _required_number(
            comparison.get("candidate_output_tokens_per_second_median"),
            "candidate_output_tokens_per_second_median",
        ),
        "ratio_of_medians_improvement_percent": _required_number(
            comparison.get("ratio_of_medians_improvement_percent"),
            "ratio_of_medians_improvement_percent",
        ),
        "paired_improvement_median_percent": _required_number(
            comparison.get("paired_improvement_percent", {}).get("median"),
            "paired_improvement_percent.median",
        ),
        "bootstrap_p05_percent": _required_number(
            bootstrap.get("p05_percent"),
            "bootstrap.p05_percent",
        ),
        "bootstrap_p95_percent": _required_number(
            bootstrap.get("p95_percent"),
            "bootstrap.p95_percent",
        ),
        "bootstrap_excludes_zero": bootstrap.get("excludes_zero") is True,
        "baseline_ttft_p95_ms": _required_number(
            comparison.get("baseline_ttft_p95_ms_block_median"),
            "baseline_ttft_p95_ms_block_median",
        ),
        "candidate_ttft_p95_ms": _required_number(
            comparison.get("candidate_ttft_p95_ms_block_median"),
            "candidate_ttft_p95_ms_block_median",
        ),
        "ttft_p95_reduction_percent": _required_number(
            comparison.get("ttft_p95_reduction_percent"),
            "ttft_p95_reduction_percent",
        ),
        "guards_passed": comparison.get("guards_passed") is True,
        "cleanup_passed": comparison.get("cleanup_passed") is True,
        "throughput_eligible": comparison.get("throughput_eligible") is True,
        "latency_eligible": comparison.get("latency_eligible") is True,
    }


def _thread_scaling_svg(summary: dict[str, Any]) -> str:
    rows = summary["thread_scaling"]
    instances = [item["instance_class"] for item in summary["targets"]]
    width, height = 1200, 650
    left, right, top, bottom = 105, 1150, 115, 545
    high = max(row["avg_tokens_per_second"] for row in rows) * 1.12
    max_threads = max(row["threads"] for row in rows)
    colors = {"c7g.xlarge": _BLUE, "c7g.2xlarge": _GREEN, "c7g.4xlarge": _RED}
    dashes = {"pp512": "", "tg128": "10 7"}
    parts = _svg_start(width, height, "llama-bench thread scaling across C7g sizes")
    parts.extend(
        _heading(
            "llama-bench thread scaling across C7g sizes",
            "Solid lines: prefill pp512. Dashed lines: decode tg128. Five measured repetitions per point.",
        )
    )

    def x(thread: int) -> float:
        return left + (thread - 1) / max(max_threads - 1, 1) * (right - left)

    def y(value: float) -> float:
        return bottom - value / high * (bottom - top)

    for tick in range(6):
        value = high * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{_fmt(value)}</text>')
    for instance in instances:
        for test in ("pp512", "tg128"):
            points = [
                row
                for row in rows
                if row["instance_class"] == instance and row["test"] == test
            ]
            if not points:
                continue
            path = " ".join(
                f"{'M' if index == 0 else 'L'} {x(row['threads']):.2f} {y(row['avg_tokens_per_second']):.2f}"
                for index, row in enumerate(points)
            )
            dash = f' stroke-dasharray="{dashes[test]}"' if dashes[test] else ""
            parts.append(
                f'<path d="{path}" fill="none" stroke="{colors[instance]}" stroke-width="4"{dash}/>'
            )
            for row in points:
                parts.append(
                    f'<circle cx="{x(row["threads"]):.2f}" cy="{y(row["avg_tokens_per_second"]):.2f}" r="5" fill="{colors[instance]}"/>'
                )
    for thread in sorted({row["threads"] for row in rows}):
        parts.append(f'<text x="{x(thread):.2f}" y="{bottom + 32}" text-anchor="middle" class="axis">{thread}</text>')
    legend_x = 670
    for index, instance in enumerate(instances):
        lx = legend_x + index * 165
        parts.append(f'<line x1="{lx}" y1="82" x2="{lx + 28}" y2="82" stroke="{colors[instance]}" stroke-width="5"/>')
        parts.append(f'<text x="{lx + 36}" y="87" class="legend">{escape(instance)}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="615" text-anchor="middle" class="note">server threads</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _serving_before_after_svg(summary: dict[str, Any]) -> str:
    rows = summary["serving_comparisons"]
    width, height = 1200, 590
    left, right = 310, 1140
    top = 125
    high = max(row["candidate_output_tokens_per_second_median"] for row in rows) * 1.12
    parts = _svg_start(width, height, "Serving before and after by C7g size")
    parts.extend(
        _heading(
            "Serving before and after by C7g size",
            "Formal paired confirmation, concurrency 4. Blue is one-slot baseline; red is four-slot continuous batching.",
        )
    )
    for index, row in enumerate(rows):
        y = top + index * 68
        baseline_x = left + row["baseline_output_tokens_per_second_median"] / high * (right - left)
        candidate_x = left + row["candidate_output_tokens_per_second_median"] / high * (right - left)
        label = f"{row['instance_class']} / {row['workload_id']}"
        parts.append(f'<text x="22" y="{y + 5}" class="row-label">{escape(label)}</text>')
        parts.append(f'<line x1="{baseline_x:.2f}" y1="{y}" x2="{candidate_x:.2f}" y2="{y}" stroke="#94a3b8" stroke-width="4"/>')
        parts.append(f'<circle cx="{baseline_x:.2f}" cy="{y}" r="8" fill="{_BLUE}"/>')
        parts.append(f'<circle cx="{candidate_x:.2f}" cy="{y}" r="8" fill="{_RED}"/>')
        parts.append(f'<text x="{baseline_x:.2f}" y="{y - 14}" text-anchor="middle" class="value">{_fmt(row["baseline_output_tokens_per_second_median"])}</text>')
        parts.append(f'<text x="{candidate_x:.2f}" y="{y + 27}" text-anchor="middle" class="value">{_fmt(row["candidate_output_tokens_per_second_median"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _absolute_throughput_svg(summary: dict[str, Any]) -> str:
    rows = [
        row for row in summary["serving_comparisons"]
        if row["workload_id"] == "decode-heavy"
    ]
    width, height = 1200, 600
    left, right, top, bottom = 100, 1140, 125, 510
    high = max(row["candidate_output_tokens_per_second_median"] for row in rows) * 1.15
    parts = _svg_start(width, height, "Absolute decode serving throughput")
    parts.extend(
        _heading(
            "Absolute decode serving throughput",
            "Same pinned model and runtime; baseline and tuned values are shown separately for each instance size.",
        )
    )
    group_width = (right - left) / len(rows)
    for tick in range(6):
        value = high * tick / 5
        y = bottom - value / high * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{_fmt(value)}</text>')
    for index, row in enumerate(rows):
        center = left + (index + 0.5) * group_width
        for offset, key, color, label in (
            (-45, "baseline_output_tokens_per_second_median", _BLUE, "baseline"),
            (45, "candidate_output_tokens_per_second_median", _RED, "tuned"),
        ):
            value = row[key]
            y = bottom - value / high * (bottom - top)
            parts.append(f'<rect x="{center + offset - 32:.2f}" y="{y:.2f}" width="64" height="{bottom - y:.2f}" fill="{color}"/>')
            parts.append(f'<text x="{center + offset:.2f}" y="{y - 10:.2f}" text-anchor="middle" class="value">{_fmt(value)}</text>')
            if index == 0:
                parts.append(f'<text x="{center + offset:.2f}" y="{bottom + 56}" text-anchor="middle" class="axis">{label}</text>')
        parts.append(f'<text x="{center:.2f}" y="{bottom + 30}" text-anchor="middle" class="legend">{escape(row["instance_class"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _ttft_svg(summary: dict[str, Any]) -> str:
    rows = [
        row for row in summary["serving_comparisons"]
        if row["workload_id"] == "interactive-short"
    ]
    width, height = 1200, 560
    left, right, top, bottom = 325, 1140, 120, 480
    high = max(row["baseline_ttft_p95_ms"] for row in rows) * 1.08
    parts = _svg_start(width, height, "Interactive p95 time to first token")
    parts.extend(
        _heading(
            "Interactive p95 time to first token",
            "Lower is better. Each point is the median of block-level p95 TTFT across three paired confirmations.",
        )
    )
    for index, row in enumerate(rows):
        y = top + index * 105
        baseline_x = left + row["baseline_ttft_p95_ms"] / high * (right - left)
        candidate_x = left + row["candidate_ttft_p95_ms"] / high * (right - left)
        parts.append(f'<text x="22" y="{y + 5}" class="row-label">{escape(row["instance_class"])}</text>')
        parts.append(f'<line x1="{candidate_x:.2f}" y1="{y}" x2="{baseline_x:.2f}" y2="{y}" stroke="#94a3b8" stroke-width="4"/>')
        parts.append(f'<circle cx="{baseline_x:.2f}" cy="{y}" r="9" fill="{_BLUE}"/>')
        parts.append(f'<circle cx="{candidate_x:.2f}" cy="{y}" r="9" fill="{_GREEN}"/>')
        parts.append(f'<text x="{baseline_x:.2f}" y="{y - 17}" text-anchor="middle" class="value">{_fmt(row["baseline_ttft_p95_ms"])} ms</text>')
        parts.append(f'<text x="{candidate_x:.2f}" y="{y + 29}" text-anchor="middle" class="value">{_fmt(row["candidate_ttft_p95_ms"])} ms</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _guard_matrix_svg(summary: dict[str, Any]) -> str:
    rows = summary["screening_guards"]
    workloads = ["interactive-short", "decode-heavy", "prefill-heavy"]
    instances = [item["instance_class"] for item in summary["targets"]]
    profiles = ["slots2", "slots4"]
    width, height = 1200, 560
    left, top = 305, 145
    cell_w = min(180, (width - left - 20) / max(len(instances) * 2, 1))
    cell_h = 95
    lookup = {
        (
            row["instance_class"],
            row["workload_id"],
            "slots4" if row["profile_id"].startswith("slots4") else "slots2",
        ): row
        for row in rows
    }
    parts = _svg_start(width, height, "Workload-aware screening guard")
    parts.extend(
        _heading(
            "Workload-aware screening guard",
            "Green cells cleared the 5% screening threshold. Red cells were rejected; prefill-heavy also triggered TTFT regression.",
        )
    )
    for column, instance in enumerate(instances):
        x = left + column * cell_w * 2
        parts.append(f'<text x="{x + cell_w}" y="{top - 25}" text-anchor="middle" class="legend">{escape(instance)}</text>')
        for profile_index, profile in enumerate(profiles):
            parts.append(f'<text x="{x + profile_index * cell_w + cell_w / 2}" y="{top - 5}" text-anchor="middle" class="axis">{profile}</text>')
    for workload_index, workload in enumerate(workloads):
        y = top + workload_index * cell_h
        parts.append(f'<text x="22" y="{y + 50}" class="row-label">{escape(workload)}</text>')
        for instance_index, instance in enumerate(instances):
            for profile_index, profile in enumerate(profiles):
                item = lookup[(instance, workload, profile)]
                x = left + (instance_index * 2 + profile_index) * cell_w
                fill = "#dcfce7" if item["screening_eligible"] else "#fee2e2"
                ink = _GREEN if item["screening_eligible"] else _RED
                parts.append(f'<rect x="{x + 5}" y="{y + 5}" width="{cell_w - 10}" height="{cell_h - 10}" rx="6" fill="{fill}" stroke="{ink}"/>')
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + 43}" text-anchor="middle" fill="{ink}" class="value">{item["throughput_improvement_percent"]:+.1f}%</text>')
                status = "eligible" if item["screening_eligible"] else "rejected"
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + 67}" text-anchor="middle" fill="{ink}" class="axis">{status}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _parallel_efficiency_svg(summary: dict[str, Any]) -> str:
    rows = summary["thread_scaling"]
    values: list[dict[str, Any]] = []
    instances = [item["instance_class"] for item in summary["targets"]]
    for instance in instances:
        instance_rows = [row for row in rows if row["instance_class"] == instance]
        for test in ("pp512", "tg128"):
            test_rows = [row for row in instance_rows if row["test"] == test]
            one = next(row for row in test_rows if row["threads"] == 1)
            selected = max(test_rows, key=lambda row: row["threads"])
            values.append(
                {
                    "instance_class": instance,
                    "test": test,
                    "threads": selected["threads"],
                    "efficiency_percent": (
                        selected["avg_tokens_per_second"]
                        / (one["avg_tokens_per_second"] * selected["threads"])
                        * 100
                    ),
                }
            )
    width, height = 1200, 590
    left, right, top, bottom = 110, 1140, 125, 500
    group_width = (right - left) / len(instances)
    parts = _svg_start(width, height, "Parallel efficiency at the largest measured server thread count")
    parts.extend(
        _heading(
            "Parallel efficiency at the largest measured server thread count",
            "Efficiency compares observed throughput with ideal linear scaling from the one-thread result.",
        )
    )
    for tick in range(6):
        value = tick * 20
        y = bottom - value / 100 * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{value}%</text>')
    for instance_index, instance in enumerate(instances):
        center = left + (instance_index + 0.5) * group_width
        for offset, test, color in ((-48, "pp512", _BLUE), (48, "tg128", _AMBER)):
            item = next(row for row in values if row["instance_class"] == instance and row["test"] == test)
            value = item["efficiency_percent"]
            y = bottom - value / 100 * (bottom - top)
            parts.append(f'<rect x="{center + offset - 34:.2f}" y="{y:.2f}" width="68" height="{bottom - y:.2f}" fill="{color}"/>')
            parts.append(f'<text x="{center + offset:.2f}" y="{y - 10:.2f}" text-anchor="middle" class="value">{value:.1f}%</text>')
        parts.append(f'<text x="{center:.2f}" y="{bottom + 32}" text-anchor="middle" class="legend">{escape(instance)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        '<style>.title{font:700 29px Segoe UI,Arial,sans-serif;fill:#0f172a}.subtitle{font:16px Segoe UI,Arial,sans-serif;fill:#64748b}.axis{font:13px Segoe UI,Arial,sans-serif;fill:#64748b}.legend{font:600 14px Segoe UI,Arial,sans-serif;fill:#334155}.row-label{font:600 14px Segoe UI,Arial,sans-serif;fill:#0f172a}.value{font:700 13px Segoe UI,Arial,sans-serif;fill:#0f172a}.note{font:14px Segoe UI,Arial,sans-serif;fill:#64748b}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _heading(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="22" y="42" class="title">{escape(title)}</text>',
        f'<text x="22" y="72" class="subtitle">{escape(subtitle)}</text>',
    ]


def _chart_title(name: str) -> str:
    return {
        "thread_scaling_across_sizes.svg": "Thread scaling across three C7g sizes",
        "serving_before_after_by_size.svg": "Serving before and after by size",
        "absolute_throughput_by_size.svg": "Absolute decode throughput by size",
        "ttft_before_after_by_size.svg": "Interactive p95 TTFT before and after",
        "workload_guard_matrix.svg": "Workload-aware recommendation guard",
        "parallel_efficiency.svg": "Parallel efficiency",
    }[name]


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    items = list(rows)
    if not items:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def _required_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.3f}"
    return "n/a"
