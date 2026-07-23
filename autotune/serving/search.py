from __future__ import annotations

from dataclasses import replace
from typing import Any

from autotune.serving.capabilities import discover_server_capabilities
from autotune.serving.metrics import summarize_values
from autotune.serving.schemas import (
    NOT_AVAILABLE,
    ServerCapabilityReport,
    ServerMemoryProfile,
    ServerRuntimeProfile,
    ServingExperimentPlan,
    ServingExperimentResult,
    ServingMetricSummary,
    ServingParetoPoint,
    ServingParetoResult,
    ServingRecommendation,
    ServingTrafficProfile,
    ServingWorkloadProfile,
    stable_digest,
)
from autotune.serving.workloads import get_standard_workload


DEFAULT_HELP_TEXT = """
llama.cpp server build: b9999
--model --host --port --threads --ctx-size --batch-size --ubatch-size
--parallel --cont-batching --no-cont-batching --mmap --no-mmap --mlock
--prompt-cache --cache-type-k --cache-type-v
/health /metrics /slots /v1/chat/completions stream seed temperature max_tokens
"""


def create_serving_plan(
    *,
    workload_id: str,
    traffic_mode: str = "closed-loop",
    concurrency: int = 1,
    candidate_cap: int = 12,
    capability_report: ServerCapabilityReport | None = None,
    logical_cpus: int = 8,
    available_memory_bytes: int = 12 * 1024 * 1024 * 1024,
) -> ServingExperimentPlan:
    workload = get_standard_workload(workload_id)
    traffic = ServingTrafficProfile(
        traffic_mode=traffic_mode,
        target_concurrency=concurrency,
        request_count=max(concurrency, 1),
        target_request_rate=float(concurrency) if traffic_mode == "open-loop" else None,
        timeout_seconds=workload.timeout_seconds,
    )
    capabilities = capability_report or discover_server_capabilities(
        help_text=DEFAULT_HELP_TEXT,
        source_commit="b9999",
        architecture="planned",
        backend="unknown",
        kernel_identity="planned",
    )
    candidates, warnings = generate_runtime_candidates(
        workload=workload,
        capabilities=capabilities,
        candidate_cap=candidate_cap,
        logical_cpus=logical_cpus,
        available_memory_bytes=available_memory_bytes,
    )
    experiment_id = stable_digest(
        {
            "workload": workload.to_dict(),
            "traffic": traffic.to_dict(),
            "capability_digest": capabilities.capability_digest,
            "candidate_ids": [candidate.profile_id for candidate in candidates],
        }
    )[:24]
    return ServingExperimentPlan(
        experiment_id=experiment_id,
        workload=workload,
        traffic=traffic,
        candidates=candidates,
        capability_report=capabilities,
        candidate_cap=candidate_cap,
        warnings=warnings,
    )


def generate_runtime_candidates(
    *,
    workload: ServingWorkloadProfile,
    capabilities: ServerCapabilityReport,
    candidate_cap: int,
    logical_cpus: int,
    available_memory_bytes: int,
) -> tuple[list[ServerRuntimeProfile], list[str]]:
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be >= 1")
    baseline = ServerRuntimeProfile(
        profile_id="baseline",
        parallel_slots=1,
        continuous_batching=None,
        context_size=max(2048, workload.max_context_tokens_required),
        effective_context_per_slot=max(2048, workload.max_context_tokens_required),
        generation_threads=min(4, logical_cpus),
        batch_threads=None,
        batch_size=512,
        micro_batch_size=128,
        mmap=True,
        mlock=False,
        prompt_cache_policy="disabled",
        kv_cache_type=NOT_AVAILABLE,
    )
    candidates = [baseline]
    warnings: list[str] = []

    def add(candidate: ServerRuntimeProfile) -> None:
        if len(candidates) >= candidate_cap:
            return
        if any(existing.profile_id == candidate.profile_id for existing in candidates):
            return
        candidates.append(candidate)

    if _available(capabilities, "parallel_slots"):
        for slots in [2, 4, 8]:
            if slots > logical_cpus:
                continue
            context = max(workload.max_context_tokens_required * slots, baseline.context_size or 2048)
            if not _memory_safe(slots, context, available_memory_bytes):
                warnings.append(f"parallel_slots_{slots}_skipped_memory_guard")
                continue
            add(
                replace(
                    baseline,
                    profile_id=f"slots-{slots}",
                    parallel_slots=slots,
                    continuous_batching=True if _available(capabilities, "continuous_batching") else None,
                    context_size=context,
                    effective_context_per_slot=context // slots,
                )
            )
    if _available(capabilities, "generation_threads"):
        for threads in [4, 6, 8]:
            if threads > logical_cpus:
                continue
            add(replace(baseline, profile_id=f"threads-{threads}", generation_threads=threads))
    if _available(capabilities, "batch_size") or _available(capabilities, "micro_batch_size"):
        for batch, ubatch in [(256, 128), (512, 256), (1024, 256)]:
            add(
                replace(
                    baseline,
                    profile_id=f"batch-{batch}-ubatch-{ubatch}",
                    batch_size=batch if _available(capabilities, "batch_size") else baseline.batch_size,
                    micro_batch_size=ubatch if _available(capabilities, "micro_batch_size") else baseline.micro_batch_size,
                )
            )
    if _available(capabilities, "context_size"):
        required = workload.max_context_tokens_required
        for context in sorted({required, max(required, 4096)}):
            add(replace(baseline, profile_id=f"context-{context}", context_size=context, effective_context_per_slot=context))
    if _available(capabilities, "mmap"):
        add(replace(baseline, profile_id="no-mmap", mmap=False, mlock=False))
    if _available(capabilities, "mlock"):
        add(replace(baseline, profile_id="mmap-mlock", mmap=True, mlock=True))
    if workload.workload_id == "shared-prefix" and _available(capabilities, "prompt_cache_behavior"):
        add(replace(baseline, profile_id="shared-prefix-cache", prompt_cache_policy="enabled"))
    if _available(capabilities, "kv_cache_configuration"):
        add(replace(baseline, profile_id="kv-cache-q8", kv_cache_type="q8_0"))
    return candidates, warnings


def run_fixture_serving_search(plan: ServingExperimentPlan) -> tuple[list[ServingExperimentResult], ServingParetoResult]:
    results = [_fixture_result(plan, candidate) for candidate in plan.candidates]
    pareto = compute_pareto_result(plan, results)
    return results, pareto


def compute_pareto_result(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingParetoResult:
    points = [_point(result) for result in results]
    frontier = [point for point in points if _not_dominated(point, points)]
    recommendations = {
        "interactive_latency": _recommend_latency(plan, results),
        "balanced_multi_user": _recommend_balanced(plan, results),
        "maximum_throughput": _recommend_throughput(plan, results),
        "shared_prefix": _recommend_shared_prefix(plan, results),
        "memory_constrained": _recommend_memory(plan, results),
    }
    eligible_ids = {rec.candidate_id for rec in recommendations.values() if rec.status == "recommended" and rec.candidate_id}
    return ServingParetoResult(
        experiment_id=plan.experiment_id,
        frontier_points=frontier,
        recommendations=recommendations,
        baseline_candidate_id="baseline",
        candidate_count=len(results),
        eligible_candidate_count=len(eligible_ids),
        result_origin=_result_origin(results),
    )


def serving_passport_extension(
    *,
    plan: ServingExperimentPlan,
    pareto: ServingParetoResult,
    result: ServingExperimentResult,
    server_binary_digest: str = "unknown",
    model_digest: str = "unknown",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "serving_workload_id": plan.workload.workload_id,
        "traffic_mode": plan.traffic.traffic_mode,
        "concurrency": plan.traffic.target_concurrency,
        "request_rate": plan.traffic.target_request_rate,
        "prompt_digest": plan.workload.prompt_digest,
        "input_output_token_distribution": {
            "input_tokens": result.metrics["input_prefill_tokens_per_second"].to_dict(),
            "output_tokens": result.metrics["output_tokens"].to_dict(),
        },
        "runtime_profile": result.runtime_profile.to_dict(),
        "network_profile": result.network_profile.to_dict(),
        "latency_percentiles": {
            "ttft_ms": result.metrics["ttft_ms"].to_dict(),
            "e2e_latency_ms": result.metrics["e2e_latency_ms"].to_dict(),
        },
        "throughput": {
            "aggregate_output_tokens_per_second": result.metrics["aggregate_output_tokens_per_second"].to_dict(),
            "aggregate_total_tokens_per_second": result.metrics["aggregate_total_tokens_per_second"].to_dict(),
        },
        "failure_rate": _failure_rate(result),
        "fairness": _fairness(result),
        "peak_rss_bytes": result.memory_profile.peak_rss_bytes,
        "quality_evidence": {"quality_guard_status": result.quality_guard_status},
        "server_binary_digest": server_binary_digest,
        "model_digest": model_digest,
        "server_capability_digest": plan.capability_report.capability_digest,
        "hard_drift_keys": [
            "binary_or_model_digest",
            "unsupported_server_option",
            "workload_shape",
            "concurrency_class",
            "network_topology",
            "server_capability_digest",
            "memory_headroom",
            "kernel_architecture",
        ],
        "pareto_result_digest": stable_digest(pareto.to_dict()),
    }


def _fixture_result(plan: ServingExperimentPlan, candidate: ServerRuntimeProfile) -> ServingExperimentResult:
    baseline_throughput = 100.0
    throughput = baseline_throughput
    ttft = 100.0
    e2e = 500.0
    peak = 2_000_000_000
    quality = "passed"
    failed = 0
    if candidate.parallel_slots and candidate.parallel_slots > 1:
        throughput *= 1.0 + min(candidate.parallel_slots, 4) * 0.12
        ttft *= 1.05 + min(candidate.parallel_slots, 4) * 0.02
        peak += candidate.parallel_slots * 120_000_000
    if candidate.generation_threads and candidate.generation_threads >= 6:
        throughput *= 1.08
        ttft *= 0.98
    if candidate.profile_id.startswith("batch-"):
        throughput *= 1.07
        ttft *= 1.03
    if candidate.profile_id == "shared-prefix-cache":
        ttft *= 0.70
        e2e *= 0.82
        throughput *= 1.18
    if candidate.profile_id == "no-mmap":
        peak += 400_000_000
    if candidate.profile_id == "kv-cache-q8":
        peak -= 500_000_000
        quality = "failed"
    metrics = {
        "completed_requests_per_second": _summary("completed_requests_per_second", [plan.traffic.request_count / 1.0], failed),
        "aggregate_output_tokens_per_second": _summary("aggregate_output_tokens_per_second", [throughput], failed),
        "aggregate_total_tokens_per_second": _summary("aggregate_total_tokens_per_second", [throughput * 1.2], failed),
        "input_prefill_tokens_per_second": _summary("input_prefill_tokens_per_second", [throughput * 0.2], failed),
        "ttft_ms": _summary("ttft_ms", [ttft, ttft * 1.01, ttft * 0.99, ttft * 1.02, ttft * 0.98], failed),
        "e2e_latency_ms": _summary("e2e_latency_ms", [e2e, e2e * 1.02, e2e * 0.99, e2e * 1.01, e2e * 0.98], failed),
        "output_tokens": _summary("output_tokens", [float(plan.workload.requested_output_tokens)] * 5, failed),
    }
    return ServingExperimentResult(
        experiment_id=plan.experiment_id,
        candidate_id=candidate.profile_id,
        workload_id=plan.workload.workload_id,
        traffic_profile=plan.traffic,
        runtime_profile=candidate,
        result_origin="test_fixture",
        request_count=5,
        failed_request_count=failed,
        metrics=metrics,
        memory_profile=ServerMemoryProfile(peak_rss_bytes=peak, available_memory_headroom_bytes=6_000_000_000 - peak),
        quality_guard_status=quality,
        cleanup_status="passed",
        client_validity={"schema_version": "1.0", "client_overload_detected": False},
    )


def _summary(name: str, values: list[float], failed: int) -> ServingMetricSummary:
    return summarize_values(name, values, failed_count=failed)


def _point(result: ServingExperimentResult) -> ServingParetoPoint:
    return ServingParetoPoint(
        candidate_id=result.candidate_id,
        aggregate_output_tokens_per_second=_p50(result, "aggregate_output_tokens_per_second"),
        ttft_p95_ms=_p95(result, "ttft_ms"),
        e2e_p95_ms=_p95(result, "e2e_latency_ms"),
        peak_rss_bytes=result.memory_profile.peak_rss_bytes,
        failure_rate=_failure_rate(result),
        quality_guard_status=result.quality_guard_status,
    )


def _not_dominated(point: ServingParetoPoint, points: list[ServingParetoPoint]) -> bool:
    for other in points:
        if other.candidate_id == point.candidate_id:
            continue
        if _dominates(other, point):
            return False
    return True


def _dominates(a: ServingParetoPoint, b: ServingParetoPoint) -> bool:
    if None in {a.aggregate_output_tokens_per_second, b.aggregate_output_tokens_per_second, a.ttft_p95_ms, b.ttft_p95_ms}:
        return False
    return (
        float(a.aggregate_output_tokens_per_second) >= float(b.aggregate_output_tokens_per_second)
        and float(a.ttft_p95_ms) <= float(b.ttft_p95_ms)
        and a.failure_rate <= b.failure_rate
        and (
            float(a.aggregate_output_tokens_per_second) > float(b.aggregate_output_tokens_per_second)
            or float(a.ttft_p95_ms) < float(b.ttft_p95_ms)
            or a.failure_rate < b.failure_rate
        )
    )


def _recommend_throughput(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingRecommendation:
    baseline = _result(results, "baseline")
    best = max(_eligible(results), key=lambda item: _p50(item, "aggregate_output_tokens_per_second") or -1, default=None)
    if baseline is None or best is None:
        return _none("maximum_throughput", "No eligible measured candidates.")
    improvement = _percent_delta(_p50(baseline, "aggregate_output_tokens_per_second"), _p50(best, "aggregate_output_tokens_per_second"))
    guards = _guards(best, improvement=improvement, minimum=5.0, mode="throughput")
    if best.candidate_id == "baseline" or improvement is None or improvement < 5.0 or any(item["status"] == "failed" for item in guards):
        return ServingRecommendation(
            recommendation_class="maximum_throughput",
            status="inconclusive",
            candidate_id=best.candidate_id,
            message="No throughput candidate exceeded the 5% guard with zero failures and quality pass.",
            guard_results=guards,
        )
    return ServingRecommendation(
        recommendation_class="maximum_throughput",
        status="recommended",
        candidate_id=best.candidate_id,
        message="Best observed eligible throughput configuration for this serving workload and protocol.",
        guard_results=guards,
        selected_metrics={"improvement_percent": round(improvement, 6)},
    )


def _recommend_latency(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingRecommendation:
    baseline = _result(results, "baseline")
    best = min(_eligible(results), key=lambda item: _p95(item, "ttft_ms") or float("inf"), default=None)
    if baseline is None or best is None:
        return _none("interactive_latency", "No eligible measured candidates.")
    improvement = _latency_improvement(_p95(baseline, "ttft_ms"), _p95(best, "ttft_ms"))
    guards = _guards(best, improvement=improvement, minimum=10.0, mode="latency")
    throughput_regression = _percent_delta(_p50(baseline, "aggregate_output_tokens_per_second"), _p50(best, "aggregate_output_tokens_per_second"))
    if throughput_regression is not None and throughput_regression < -5.0:
        guards.append({"schema_version": "1.0", "name": "throughput_regression_limit", "status": "failed"})
    if best.candidate_id == "baseline" or improvement is None or improvement < 10.0 or any(item["status"] == "failed" for item in guards):
        return ServingRecommendation(
            recommendation_class="interactive_latency",
            status="inconclusive",
            candidate_id=best.candidate_id,
            message="No latency candidate exceeded the 10% guard without violating throughput, failure, and quality constraints.",
            guard_results=guards,
        )
    return ServingRecommendation(
        recommendation_class="interactive_latency",
        status="recommended",
        candidate_id=best.candidate_id,
        message="Best observed eligible latency configuration for this serving workload and protocol.",
        guard_results=guards,
        selected_metrics={"ttft_p95_improvement_percent": round(improvement, 6)},
    )


def _recommend_balanced(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingRecommendation:
    candidates = _eligible(results)
    if not candidates:
        return _none("balanced_multi_user", "No eligible candidates.")
    best = sorted(
        candidates,
        key=lambda item: (
            -float(_p50(item, "aggregate_output_tokens_per_second") or 0),
            float(_p95(item, "ttft_ms") or float("inf")),
        ),
    )[0]
    return ServingRecommendation(
        recommendation_class="balanced_multi_user",
        status="recommended" if best.candidate_id != "baseline" else "inconclusive",
        candidate_id=best.candidate_id,
        message="Balanced candidate is selected from the Pareto frontier; this is workload-specific.",
    )


def _recommend_shared_prefix(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingRecommendation:
    if plan.workload.workload_id != "shared-prefix":
        return _none("shared_prefix", "Shared-prefix recommendation applies only to shared-prefix workload identity.")
    candidate = _result(results, "shared-prefix-cache")
    if candidate is None or candidate.quality_guard_status != "passed":
        return _none("shared_prefix", "No eligible shared-prefix cache candidate.")
    return ServingRecommendation(
        recommendation_class="shared_prefix",
        status="recommended",
        candidate_id=candidate.candidate_id,
        message="Shared-prefix recommendation is scoped only to repeated-prefix workloads.",
    )


def _recommend_memory(plan: ServingExperimentPlan, results: list[ServingExperimentResult]) -> ServingRecommendation:
    candidates = [item for item in _eligible(results) if item.memory_profile.peak_rss_bytes is not None]
    if not candidates:
        return _none("memory_constrained", "No eligible memory evidence.")
    best = min(candidates, key=lambda item: int(item.memory_profile.peak_rss_bytes or 0))
    return ServingRecommendation(
        recommendation_class="memory_constrained",
        status="recommended" if best.candidate_id != "baseline" else "inconclusive",
        candidate_id=best.candidate_id,
        message="Memory recommendation is a capacity/headroom result, not a raw speedup claim.",
    )


def _none(recommendation_class: str, message: str) -> ServingRecommendation:
    return ServingRecommendation(recommendation_class=recommendation_class, status="no_eligible_recommendation", candidate_id=None, message=message)


def _guards(result: ServingExperimentResult, *, improvement: float | None, minimum: float, mode: str) -> list[dict[str, Any]]:
    guards = [
        {"schema_version": "1.0", "name": "zero_failed_requests", "status": "passed" if result.failed_request_count == 0 else "failed"},
        {"schema_version": "1.0", "name": "quality_guard_pass", "status": "passed" if result.quality_guard_status == "passed" else "failed"},
        {"schema_version": "1.0", "name": "cleanup_pass", "status": "passed" if result.cleanup_status == "passed" else "failed"},
        {
            "schema_version": "1.0",
            "name": f"{mode}_minimum_delta",
            "status": "passed" if improvement is not None and improvement >= minimum else "failed",
            "observed_percent": improvement,
            "required_percent": minimum,
        },
    ]
    return guards


def _eligible(results: list[ServingExperimentResult]) -> list[ServingExperimentResult]:
    return [
        item
        for item in results
        if item.failed_request_count == 0
        and item.quality_guard_status == "passed"
        and item.cleanup_status == "passed"
        and not item.client_validity.get("client_overload_detected", False)
    ]


def _available(capabilities: ServerCapabilityReport, concept: str) -> bool:
    option = capabilities.options.get(concept)
    return bool(option and option.availability == "available")


def _memory_safe(slots: int, context_size: int, available_memory_bytes: int) -> bool:
    estimated = slots * context_size * 4096
    return estimated < max(0, available_memory_bytes - 1024 * 1024 * 1024)


def _result(results: list[ServingExperimentResult], candidate_id: str) -> ServingExperimentResult | None:
    for result in results:
        if result.candidate_id == candidate_id:
            return result
    return None


def _p50(result: ServingExperimentResult, metric: str) -> float | None:
    summary = result.metrics.get(metric)
    return summary.p50 if summary else None


def _p95(result: ServingExperimentResult, metric: str) -> float | None:
    summary = result.metrics.get(metric)
    return summary.p95 if summary else None


def _failure_rate(result: ServingExperimentResult) -> float:
    if result.request_count <= 0:
        return 1.0
    return round(result.failed_request_count / result.request_count, 6)


def _fairness(result: ServingExperimentResult) -> dict[str, Any]:
    per_client: dict[str, int] = {}
    for request in result.request_evidence:
        per_client[request.client_id] = per_client.get(request.client_id, 0) + 1
    return {"schema_version": "1.0", "per_client_completed_requests": per_client, "starvation_detected": False}


def _percent_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def _latency_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return ((baseline - candidate) / baseline) * 100.0


def _result_origin(results: list[ServingExperimentResult]) -> str:
    origins = {result.result_origin for result in results}
    if origins == {"test_fixture"}:
        return "test_fixture"
    if origins == {"measured"}:
        return "measured"
    return "unverified"
