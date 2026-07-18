from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.aggregation import aggregate_quality_results
from autotune.arm_tune.quality.schemas import (
    CandidateQualityResult,
    QualityAggregateResult,
    QualityCase,
    QualityCaseResult,
    QualitySuiteManifest,
)
from autotune.arm_tune.quality.suite import load_quality_suite, prompt_suite_from_quality_cases, suite_snapshot_payload
from autotune.arm_tune.quality.validators import validate_quality_case
from autotune.arm_tune.schemas import (
    ArtifactValidationResult,
    BenchmarkArtifactManifest,
    CpuTopology,
    GenerationRequestResult,
    LlamaServerBenchmarkResult,
    LlamaServerJobSpec,
    LlamaServerSettings,
    to_plain,
    utc_now,
)
from autotune.arm_tune.search.candidates import generate_thread_counts, visible_cpu_indexes
from autotune.arm_tune.search.runner import SearchConfig
from autotune.arm_tune.search.statistics import MetricDistribution, NoiseBandDecision, summarize_metric
from autotune.arm_tune.targets.base import ArmTuneTarget
from autotune.arm_tune.targets.protocol import atomic_write_json


PRIMARY_METRIC = "ttft_ms"
SECONDARY_METRIC = "end_to_end_latency_ms"
SERVER_OPTIMIZE_OPERATION = "optimize.llama_server"
RECOMMENDATION_STATES = {
    "recommended",
    "inconclusive",
    "rejected_quality",
    "rejected_execution",
    "insufficient_evidence",
    "no_eligible_candidate",
}


@dataclass(frozen=True)
class ServerSearchCandidate:
    candidate_id: str
    stage: str
    threads: int
    context_size: int
    batch_size: int | None = None
    ubatch_size: int | None = None
    affinity_mode: str = "scheduler-default"
    cpu_affinity: list[int] | None = None
    allowed_cpu_mask: str = "scheduler-default"
    is_baseline: bool = False
    description: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    def settings_for(self, base: LlamaServerSettings) -> LlamaServerSettings:
        return replace(
            base,
            threads=self.threads,
            context_size=self.context_size,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            affinity_mode=self.affinity_mode,
            cpu_affinity=list(self.cpu_affinity) if self.cpu_affinity is not None else None,
        )


@dataclass(frozen=True)
class ServerSearchStageResult:
    stage: str
    candidate_ids: list[str]
    selected_candidate_id: str | None
    decision: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerBenchmarkRunRecord:
    phase: str
    run_id: str
    artifact_directory: str
    result_origin: str
    validation_status: str
    startup_status: str
    readiness_status: str
    request_status: str
    stream_parse_status: str
    cleanup_status: str
    benchmark_status: str
    measured_request_count: int
    failed_request_count: int
    median_ttft_ms: float | None
    median_end_to_end_latency_ms: float | None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerSearchCandidateResult:
    candidate: ServerSearchCandidate
    benchmark_runs: list[ServerBenchmarkRunRecord]
    quality_result: CandidateQualityResult
    distributions: dict[str, MetricDistribution]
    eligibility_status: str
    eligible: bool
    result_origin: str
    failure_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerSearchRecommendation:
    status: str
    candidate_id: str | None
    message: str
    noise_band_decision: NoiseBandDecision | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArmTuneServerSearchResult:
    run_id: str
    status: str
    operation: str
    runtime_adapter: str
    result_origin: str
    target_type: str
    target_architecture: str
    primary_metric: str
    secondary_metric: str
    search_config: SearchConfig
    quality_gate_status: str
    quality_suite_manifest: QualitySuiteManifest
    candidate_count: int
    eligible_candidate_count: int
    failed_candidate_count: int
    baseline_candidate_id: str
    selected_candidate_id: str | None
    recommendation: ServerSearchRecommendation
    candidates: list[ServerSearchCandidateResult]
    stages: list[ServerSearchStageResult]
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: BenchmarkArtifactManifest = field(default_factory=BenchmarkArtifactManifest)
    artifact_validation: ArtifactValidationResult = field(default_factory=ArtifactValidationResult)
    artifact_directory: str = ""
    created_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def run_server_search(
    *,
    target: ArmTuneTarget,
    base_job: LlamaServerJobSpec,
    config: SearchConfig,
    run_id: str,
    optimize_dir: str | Path,
) -> ArmTuneServerSearchResult:
    root = Path(optimize_dir)
    root.mkdir(parents=True, exist_ok=True)
    suite_id, quality_cases, quality_manifest, suite_payload = _load_quality_context(config)
    atomic_write_json(root / "optimize_job.json", _optimize_job_payload(target, base_job, config, run_id))
    atomic_write_json(root / "effective_config.json", _effective_config_payload(target, base_job, config))
    atomic_write_json(root / "quality_suite_manifest.json", quality_manifest.to_dict())
    atomic_write_json(root / "quality_suite_snapshot.json", suite_payload)
    prompt_suite_path = _write_prompt_suite_snapshot(root, target, quality_cases)
    if target.spec.type == "local" and prompt_suite_path is not None:
        base_job = replace(base_job, workload=replace(base_job.workload, prompt_suite_path=str(prompt_suite_path)))

    inspection = target.inspect()
    topology = inspection.host.topology
    baseline = baseline_candidate(base_job)
    candidates_by_id: dict[str, ServerSearchCandidate] = {baseline.candidate_id: baseline}
    benchmark_results: dict[str, list[LlamaServerBenchmarkResult]] = {baseline.candidate_id: []}
    stages: list[ServerSearchStageResult] = []
    warnings: list[str] = []

    _run_candidate(
        target=target,
        base_job=base_job,
        config=config,
        run_id=run_id,
        root=root,
        candidate=baseline,
        phase="search",
        results=benchmark_results,
    )
    supported_flags = _supported_server_flags(benchmark_results[baseline.candidate_id])

    for stage_name, raw_candidates in [
        ("batch_parameters", generate_server_batch_candidates(baseline, base_job, supported_flags=supported_flags)),
        ("thread_count", generate_server_thread_candidates(base_job, topology)),
        (
            "affinity",
            generate_server_affinity_candidates(
                baseline,
                topology,
                allowed_modes=config.affinity_modes,
            ),
        ),
    ]:
        stage_candidates = _apply_candidate_cap(raw_candidates, candidates_by_id, config)
        for candidate in stage_candidates:
            candidates_by_id[candidate.candidate_id] = candidate
            benchmark_results.setdefault(candidate.candidate_id, [])
            _run_candidate(
                target=target,
                base_job=base_job,
                config=config,
                run_id=run_id,
                root=root,
                candidate=candidate,
                phase="search",
                results=benchmark_results,
            )
        candidate_summaries = _candidate_summaries(
            candidates_by_id=candidates_by_id,
            benchmark_results=benchmark_results,
            quality_cases=quality_cases,
            quality_manifest=quality_manifest,
            quality_enabled=config.quality_gate_enabled,
        )
        selected = _select_best_eligible(candidate_summaries)
        stages.append(
            _stage_result(
                stage_name=stage_name,
                raw_candidates=raw_candidates,
                stage_candidates=stage_candidates,
                selected_candidate_id=selected.candidate.candidate_id if selected else None,
                candidates_by_id=candidates_by_id,
                config=config,
            )
        )

    initial_summaries = _candidate_summaries(
        candidates_by_id=candidates_by_id,
        benchmark_results=benchmark_results,
        quality_cases=quality_cases,
        quality_manifest=quality_manifest,
        quality_enabled=config.quality_gate_enabled,
    )
    selected_before_confirmation = _select_best_eligible(initial_summaries)
    if selected_before_confirmation is not None:
        _run_confirmation(
            target=target,
            base_job=base_job,
            config=config,
            run_id=run_id,
            root=root,
            baseline=baseline,
            selected=selected_before_confirmation.candidate,
            results=benchmark_results,
        )

    candidate_summaries = _candidate_summaries(
        candidates_by_id=candidates_by_id,
        benchmark_results=benchmark_results,
        quality_cases=quality_cases,
        quality_manifest=quality_manifest,
        quality_enabled=config.quality_gate_enabled,
    )
    selected = _select_best_eligible(candidate_summaries)
    baseline_summary = next(item for item in candidate_summaries if item.candidate.candidate_id == baseline.candidate_id)
    recommendation = _recommendation(
        baseline=baseline_summary,
        selected=selected,
        candidate_count=len(candidate_summaries),
        noise_band_percent=config.noise_band_percent,
    )
    result_origin = _result_origin(candidate_summaries)
    result = ArmTuneServerSearchResult(
        run_id=run_id,
        status="completed" if candidate_summaries else "failed",
        operation=SERVER_OPTIMIZE_OPERATION,
        runtime_adapter="llama_server",
        result_origin=result_origin,
        target_type=target.spec.type,
        target_architecture=inspection.host.architecture.normalized_machine,
        primary_metric=PRIMARY_METRIC,
        secondary_metric=SECONDARY_METRIC,
        search_config=config,
        quality_gate_status=_overall_quality_status(candidate_summaries),
        quality_suite_manifest=quality_manifest,
        candidate_count=len(candidate_summaries),
        eligible_candidate_count=sum(1 for item in candidate_summaries if item.eligible),
        failed_candidate_count=sum(1 for item in candidate_summaries if not item.eligible),
        baseline_candidate_id=baseline.candidate_id,
        selected_candidate_id=selected.candidate.candidate_id if selected else None,
        recommendation=recommendation,
        candidates=candidate_summaries,
        stages=stages,
        rejected_candidates=_rejected_candidates(candidate_summaries),
        warnings=warnings + _distribution_warnings(candidate_summaries),
        artifact_directory=str(root),
    )
    return _finalize_result(root, result)


def baseline_candidate(job: LlamaServerJobSpec) -> ServerSearchCandidate:
    server = job.server
    return ServerSearchCandidate(
        candidate_id="baseline",
        stage="baseline",
        threads=server.threads,
        context_size=server.context_size,
        batch_size=server.batch_size,
        ubatch_size=server.ubatch_size,
        affinity_mode=server.affinity_mode,
        cpu_affinity=list(server.cpu_affinity) if server.cpu_affinity is not None else None,
        allowed_cpu_mask=_mask(server.cpu_affinity),
        is_baseline=True,
        description="configured llama-server baseline",
    )


def generate_server_thread_candidates(job: LlamaServerJobSpec, topology: CpuTopology) -> list[ServerSearchCandidate]:
    base = baseline_candidate(job)
    candidates: list[ServerSearchCandidate] = []
    for threads in generate_thread_counts(topology, baseline_threads=job.server.threads):
        if threads == base.threads:
            continue
        candidates.append(
            replace(
                base,
                candidate_id=f"threads-{threads}",
                stage="thread_count",
                threads=threads,
                is_baseline=False,
                cpu_affinity=None,
                affinity_mode="scheduler-default",
                allowed_cpu_mask="scheduler-default",
                description=f"{threads} llama-server threads",
            )
        )
    return candidates


def generate_server_batch_candidates(
    base: ServerSearchCandidate,
    job: LlamaServerJobSpec,
    *,
    supported_flags: list[str],
) -> list[ServerSearchCandidate]:
    supported = set(supported_flags)
    batch_supported = "--batch-size" in supported
    ubatch_supported = "--ubatch-size" in supported
    if not batch_supported and not ubatch_supported:
        return []
    baseline_batch = base.batch_size or 128
    batch_size = max(1, baseline_batch)
    ubatch_size = base.ubatch_size
    label_parts: list[str] = []
    if batch_supported:
        batch_size = max(1, baseline_batch * 2)
        label_parts.append(f"batch{batch_size}")
    if ubatch_supported:
        ubatch_size = min(batch_size, base.ubatch_size or batch_size)
        label_parts.append(f"ubatch{ubatch_size}")
    candidate = replace(
        base,
        candidate_id="server-" + "-".join(label_parts),
        stage="batch_parameters",
        batch_size=batch_size if batch_supported else base.batch_size,
        ubatch_size=ubatch_size,
        is_baseline=False,
        description="supported llama-server batch parameter candidate",
    )
    if candidate.batch_size == job.server.batch_size and candidate.ubatch_size == job.server.ubatch_size:
        return []
    return [candidate]


def generate_server_affinity_candidates(
    base: ServerSearchCandidate,
    topology: CpuTopology,
    *,
    allowed_modes: list[str] | None,
) -> list[ServerSearchCandidate]:
    cpus = visible_cpu_indexes(topology)
    threads = min(base.threads, len(cpus))
    if threads <= 0:
        return []
    allowed = set(allowed_modes) if allowed_modes is not None else {"compact", "spread", "reserve-one-core", "numa-local"}
    candidates: list[ServerSearchCandidate] = []
    seen: set[str] = set()
    if "compact" in allowed:
        _add_affinity_candidate(candidates, seen, base, "compact", cpus[:threads])
    if "spread" in allowed:
        _add_affinity_candidate(candidates, seen, base, "spread", _spread(cpus, threads))
    if "reserve-one-core" in allowed and len(cpus) > 1:
        reserve = cpus[:-1]
        _add_affinity_candidate(candidates, seen, base, "reserve-one-core", reserve[: min(threads, len(reserve))])
    if "numa-local" in allowed and topology.numa_nodes and topology.numa_nodes > 1:
        _add_affinity_candidate(candidates, seen, base, "numa-local", cpus[:threads])
    return candidates


def _run_candidate(
    *,
    target: ArmTuneTarget,
    base_job: LlamaServerJobSpec,
    config: SearchConfig,
    run_id: str,
    root: Path,
    candidate: ServerSearchCandidate,
    phase: str,
    results: dict[str, list[LlamaServerBenchmarkResult]],
) -> None:
    phase_results = results.setdefault(candidate.candidate_id, [])
    run_index = len(phase_results) + 1
    candidate_dir = root / "candidates" / candidate.candidate_id / phase / f"{run_index:04d}"
    workload = replace(
        base_job.workload,
        repetitions=config.confirmation_runs if phase == "confirmation" else config.search_runs,
        warmup_requests=0 if phase == "confirmation" else config.warmup_runs,
    )
    job = replace(
        base_job,
        run_id=f"{run_id}_{candidate.candidate_id}_{phase}_{run_index:04d}",
        target_run_directory=str(candidate_dir),
        server=candidate.settings_for(base_job.server),
        workload=workload,
    )
    result = target.run_server_benchmark(job, candidate_dir)
    phase_results.append(result)
    if config.cooldown_seconds > 0:
        time.sleep(config.cooldown_seconds)


def _run_confirmation(
    *,
    target: ArmTuneTarget,
    base_job: LlamaServerJobSpec,
    config: SearchConfig,
    run_id: str,
    root: Path,
    baseline: ServerSearchCandidate,
    selected: ServerSearchCandidate,
    results: dict[str, list[LlamaServerBenchmarkResult]],
) -> None:
    confirmation = [baseline]
    if selected.candidate_id != baseline.candidate_id:
        confirmation.append(selected)
    for candidate in confirmation:
        _run_candidate(
            target=target,
            base_job=base_job,
            config=config,
            run_id=run_id,
            root=root,
            candidate=candidate,
            phase="confirmation",
            results=results,
        )


def _candidate_summaries(
    *,
    candidates_by_id: dict[str, ServerSearchCandidate],
    benchmark_results: dict[str, list[LlamaServerBenchmarkResult]],
    quality_cases: list[QualityCase],
    quality_manifest: QualitySuiteManifest,
    quality_enabled: bool,
) -> list[ServerSearchCandidateResult]:
    summaries: list[ServerSearchCandidateResult] = []
    for candidate_id, candidate in candidates_by_id.items():
        results = benchmark_results.get(candidate_id, [])
        quality = _quality_result(candidate_id, quality_cases, quality_manifest, results, quality_enabled=quality_enabled)
        distributions = _performance_distributions(results)
        eligible, status, reasons = _eligibility(results, quality, quality_enabled=quality_enabled)
        summaries.append(
            ServerSearchCandidateResult(
                candidate=candidate,
                benchmark_runs=[_benchmark_run_record(result, phase=_phase_from_run_id(result.run_id)) for result in results],
                quality_result=quality,
                distributions=distributions,
                eligibility_status=status,
                eligible=eligible,
                result_origin=_candidate_origin(results),
                failure_reasons=reasons,
            )
        )
    return summaries


def _quality_result(
    candidate_id: str,
    cases: list[QualityCase],
    manifest: QualitySuiteManifest,
    benchmark_results: list[LlamaServerBenchmarkResult],
    *,
    quality_enabled: bool,
) -> CandidateQualityResult:
    if not quality_enabled:
        aggregate = QualityAggregateResult(
            total_cases=0,
            required_cases=0,
            passed_cases=0,
            failed_cases=0,
            critical_validator_failures=0,
            json_validity_rate=1.0,
            required_case_pass_rate=1.0,
            evidence_citation_correctness=1.0,
            quality_gate_status="not_evaluated",
        )
        return CandidateQualityResult(candidate_id=candidate_id, suite_manifest=manifest, aggregate=aggregate, cases=[])
    case_by_id = {case.case_id: case for case in cases}
    case_results: list[QualityCaseResult] = []
    observed_cases: set[str] = set()
    for benchmark in benchmark_results:
        for request in benchmark.requests:
            if request.phase != "measured":
                continue
            case = case_by_id.get(request.prompt_id)
            if case is None:
                continue
            observed_cases.add(case.case_id)
            case_results.append(
                validate_quality_case(
                    case,
                    request.generated_text,
                    request_status=request.request_status,
                    stream_parse_status=request.stream_parse_status,
                    output_artifact=request.request_artifact_dir,
                    repetition_index=request.repetition_index,
                )
            )
    for case in cases:
        if case.required and case.case_id not in observed_cases:
            case_results.append(
                validate_quality_case(
                    case,
                    "",
                    request_status="not_run",
                    stream_parse_status="not_parsed",
                )
            )
    return CandidateQualityResult(
        candidate_id=candidate_id,
        suite_manifest=manifest,
        aggregate=aggregate_quality_results(case_results),
        cases=case_results,
    )


def _eligibility(
    results: list[LlamaServerBenchmarkResult],
    quality: CandidateQualityResult,
    *,
    quality_enabled: bool,
) -> tuple[bool, str, list[str]]:
    reasons: list[str] = []
    if not results:
        return False, "rejected_execution", ["candidate was not executed"]
    for result in results:
        if result.result_origin == "unverified":
            reasons.append(f"{result.run_id}: result_origin is unverified")
        if result.validation_status != "passed":
            reasons.append(f"{result.run_id}: validation_status={result.validation_status}")
        if result.startup_status != "ready" or result.readiness_status != "ready":
            reasons.append(f"{result.run_id}: server was not ready")
        if result.request_status != "completed" or result.stream_parse_status != "parsed":
            reasons.append(f"{result.run_id}: request or stream parsing failed")
        if result.cleanup_status not in {"passed", "forced"}:
            reasons.append(f"{result.run_id}: cleanup_status={result.cleanup_status}")
        if result.benchmark_status != "passed":
            reasons.append(f"{result.run_id}: benchmark_status={result.benchmark_status}")
        if result.artifact_validation.validation_status != "passed":
            reasons.append(f"{result.run_id}: artifact validation failed")
    if reasons:
        return False, "rejected_execution", sorted(set(reasons))
    if not quality_enabled or quality.aggregate.quality_gate_status == "not_evaluated":
        return False, "insufficient_evidence", ["quality gate was not evaluated"]
    if quality.aggregate.quality_gate_status != "passed":
        return False, "rejected_quality", quality.aggregate.failure_reasons or ["quality gate failed"]
    if _performance_distributions(results)[PRIMARY_METRIC].median is None:
        return False, "insufficient_evidence", ["median TTFT is not available"]
    return True, "eligible", []


def _performance_distributions(results: list[LlamaServerBenchmarkResult]) -> dict[str, MetricDistribution]:
    failed = 0
    ttft: list[float] = []
    e2e: list[float] = []
    for result in results:
        measured = [request for request in result.requests if request.phase == "measured"]
        failed += sum(1 for request in measured if request.request_status != "completed" or request.stream_parse_status != "parsed")
        ttft.extend(request.ttft_ms for request in measured if request.ttft_ms is not None)
        e2e.extend(request.end_to_end_latency_ms for request in measured if request.end_to_end_latency_ms is not None)
    return {
        PRIMARY_METRIC: summarize_metric(PRIMARY_METRIC, ttft, failed_count=failed),
        SECONDARY_METRIC: summarize_metric(SECONDARY_METRIC, e2e, failed_count=failed),
    }


def _select_best_eligible(summaries: list[ServerSearchCandidateResult]) -> ServerSearchCandidateResult | None:
    best: ServerSearchCandidateResult | None = None
    best_median: float | None = None
    for summary in summaries:
        median = summary.distributions[PRIMARY_METRIC].median
        if not summary.eligible or median is None:
            continue
        if best_median is None or median < best_median:
            best = summary
            best_median = median
    return best


def _recommendation(
    *,
    baseline: ServerSearchCandidateResult,
    selected: ServerSearchCandidateResult | None,
    candidate_count: int,
    noise_band_percent: float,
) -> ServerSearchRecommendation:
    if not baseline.eligible and selected is None:
        return ServerSearchRecommendation(
            status="no_eligible_candidate",
            candidate_id=None,
            message="No candidate is eligible because execution evidence or the quality gate failed.",
        )
    if selected is None:
        return ServerSearchRecommendation(
            status="insufficient_evidence",
            candidate_id=None,
            message="No candidate has sufficient TTFT evidence for recommendation.",
        )
    decision = classify_latency_noise_band(
        baseline=baseline.distributions[PRIMARY_METRIC],
        candidate=selected.distributions[PRIMARY_METRIC],
        noise_band_percent=noise_band_percent,
    )
    if selected.candidate.candidate_id == baseline.candidate.candidate_id:
        return ServerSearchRecommendation(
            status="inconclusive",
            candidate_id=selected.candidate.candidate_id,
            message=f"No eligible non-baseline candidate among {candidate_count} tested configurations exceeded the configured noise band.",
            noise_band_decision=decision,
        )
    if decision.classification == "improved":
        return ServerSearchRecommendation(
            status="recommended",
            candidate_id=selected.candidate.candidate_id,
            message=f"Best observed quality-passing performance candidate among {candidate_count} tested configurations.",
            noise_band_decision=decision,
        )
    return ServerSearchRecommendation(
        status="inconclusive",
        candidate_id=selected.candidate.candidate_id,
        message=(
            f"Best observed quality-passing candidate among {candidate_count} tested configurations is not recommended "
            "as an improvement because median TTFT is not outside the configured noise band."
        ),
        noise_band_decision=decision,
    )


def classify_latency_noise_band(
    *,
    baseline: MetricDistribution,
    candidate: MetricDistribution,
    noise_band_percent: float,
) -> NoiseBandDecision:
    if baseline.median is None or candidate.median is None or baseline.median <= 0:
        return NoiseBandDecision(
            classification="not_comparable",
            baseline_median=baseline.median,
            candidate_median=candidate.median,
            delta_percent=None,
            noise_band_percent=noise_band_percent,
            message="missing comparable median TTFT metrics",
        )
    improvement = ((baseline.median - candidate.median) / baseline.median) * 100.0
    if improvement > noise_band_percent:
        classification = "improved"
        message = "candidate median TTFT is lower than baseline beyond the configured noise band"
    elif improvement < -noise_band_percent:
        classification = "regressed"
        message = "candidate median TTFT is higher than baseline beyond the configured noise band"
    else:
        classification = "inconclusive"
        message = "candidate median TTFT is within the configured noise band"
    return NoiseBandDecision(
        classification=classification,
        baseline_median=baseline.median,
        candidate_median=candidate.median,
        delta_percent=round(improvement, 6),
        noise_band_percent=noise_band_percent,
        message=message,
    )


def _benchmark_run_record(result: LlamaServerBenchmarkResult, *, phase: str) -> ServerBenchmarkRunRecord:
    measured = [request for request in result.requests if request.phase == "measured"]
    failed = [
        request
        for request in measured
        if request.request_status != "completed" or request.stream_parse_status != "parsed"
    ]
    ttft = result.distributions.get(PRIMARY_METRIC)
    e2e = result.distributions.get(SECONDARY_METRIC)
    return ServerBenchmarkRunRecord(
        phase=phase,
        run_id=result.run_id,
        artifact_directory=result.artifact_directory,
        result_origin=result.result_origin,
        validation_status=result.validation_status,
        startup_status=result.startup_status,
        readiness_status=result.readiness_status,
        request_status=result.request_status,
        stream_parse_status=result.stream_parse_status,
        cleanup_status=result.cleanup_status,
        benchmark_status=result.benchmark_status,
        measured_request_count=len(measured),
        failed_request_count=len(failed),
        median_ttft_ms=ttft.median if ttft else None,
        median_end_to_end_latency_ms=e2e.median if e2e else None,
    )


def _stage_result(
    *,
    stage_name: str,
    raw_candidates: list[ServerSearchCandidate],
    stage_candidates: list[ServerSearchCandidate],
    selected_candidate_id: str | None,
    candidates_by_id: dict[str, ServerSearchCandidate],
    config: SearchConfig,
) -> ServerSearchStageResult:
    if stage_candidates:
        decision = {"status": "evaluated", "reason": "stage candidate(s) executed"}
    elif raw_candidates and _candidate_limit_reached(candidates_by_id, config):
        decision = {
            "status": "skipped_due_to_candidate_limit",
            "reason": "candidate limit was reached before this stage could be fully evaluated",
        }
    else:
        decision = {"status": "skipped", "reason": "no meaningful candidates for this stage"}
    return ServerSearchStageResult(
        stage=stage_name,
        candidate_ids=[candidate.candidate_id for candidate in stage_candidates],
        selected_candidate_id=selected_candidate_id,
        decision=decision,
    )


def _apply_candidate_cap(
    candidates: list[ServerSearchCandidate],
    candidates_by_id: dict[str, ServerSearchCandidate],
    config: SearchConfig,
) -> list[ServerSearchCandidate]:
    if config.max_candidates is None:
        return candidates
    remaining = max(0, config.max_candidates - len(candidates_by_id))
    selected: list[ServerSearchCandidate] = []
    for candidate in candidates:
        if candidate.candidate_id in candidates_by_id:
            selected.append(candidate)
            continue
        if remaining <= 0:
            continue
        selected.append(candidate)
        remaining -= 1
    return selected


def _candidate_limit_reached(candidates_by_id: dict[str, ServerSearchCandidate], config: SearchConfig) -> bool:
    return config.max_candidates is not None and len(candidates_by_id) >= config.max_candidates


def _supported_server_flags(results: list[LlamaServerBenchmarkResult]) -> list[str]:
    for result in results:
        if result.inspection is not None and result.inspection.supported_flags:
            return list(result.inspection.supported_flags)
    return []


def _load_quality_context(config: SearchConfig) -> tuple[str, list[QualityCase], QualitySuiteManifest, dict[str, Any]]:
    if not config.quality_gate_enabled:
        manifest = QualitySuiteManifest(
            suite_id="not_evaluated",
            suite_digest="",
            case_count=0,
            required_case_count=0,
        )
        return "not_evaluated", [], manifest, {"schema_version": SCHEMA_VERSION, "suite_id": "not_evaluated", "cases": []}
    if not config.quality_suite_path:
        raise ArmTuneConfigError("quality_suite_path is required when quality_gate_enabled is true")
    return load_quality_suite(config.quality_suite_path)


def _write_prompt_suite_snapshot(root: Path, target: ArmTuneTarget, cases: list[QualityCase]) -> Path | None:
    if not cases:
        return None
    prompt_suite = prompt_suite_from_quality_cases(cases)
    path = root / "prompt_suite_snapshot.json"
    atomic_write_json(path, prompt_suite)
    return path.resolve() if target.spec.type == "local" else path


def _optimize_job_payload(
    target: ArmTuneTarget,
    base_job: LlamaServerJobSpec,
    config: SearchConfig,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": SERVER_OPTIMIZE_OPERATION,
        "run_id": run_id,
        "runtime_adapter": "llama_server",
        "target": target.spec.sanitized_dict(),
        "base_job": base_job.to_dict(),
        "search_config": config.to_dict(),
    }


def _effective_config_payload(target: ArmTuneTarget, base_job: LlamaServerJobSpec, config: SearchConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target.spec.sanitized_dict(),
        "runtime_adapter": "llama_server",
        "server": base_job.server.to_dict(),
        "workload": base_job.workload.to_dict(),
        "search": config.to_dict(),
        "quality_gate": {
            "schema_version": SCHEMA_VERSION,
            "enabled": config.quality_gate_enabled,
            "suite_path": config.quality_suite_path,
        },
    }


def _phase_from_run_id(run_id: str) -> str:
    if "_confirmation_" in run_id:
        return "confirmation"
    return "search"


def _candidate_origin(results: list[LlamaServerBenchmarkResult]) -> str:
    origins = {result.result_origin for result in results}
    if "unverified" in origins or not origins:
        return "unverified"
    if origins == {"measured"}:
        return "measured"
    if origins == {"test_fixture"}:
        return "test_fixture"
    return "unverified"


def _result_origin(summaries: list[ServerSearchCandidateResult]) -> str:
    origins = {summary.result_origin for summary in summaries}
    if "unverified" in origins or not origins:
        return "unverified"
    if origins == {"measured"}:
        return "measured"
    if origins == {"test_fixture"}:
        return "test_fixture"
    return "unverified"


def _overall_quality_status(summaries: list[ServerSearchCandidateResult]) -> str:
    statuses = {summary.quality_result.aggregate.quality_gate_status for summary in summaries}
    if not statuses:
        return "not_evaluated"
    if "passed" in statuses and all(status in {"passed", "failed"} for status in statuses):
        return "passed"
    if "failed" in statuses:
        return "failed"
    if "execution_failed" in statuses:
        return "execution_failed"
    if "not_evaluated" in statuses:
        return "not_evaluated"
    return sorted(statuses)[0]


def _rejected_candidates(summaries: list[ServerSearchCandidateResult]) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.eligible:
            continue
        rejected.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": summary.candidate.candidate_id,
                "eligibility_status": summary.eligibility_status,
                "failure_reasons": list(summary.failure_reasons),
            }
        )
    return rejected


def _distribution_warnings(summaries: list[ServerSearchCandidateResult]) -> list[str]:
    warnings: list[str] = []
    for summary in summaries:
        for metric in summary.distributions.values():
            warnings.extend(f"{summary.candidate.candidate_id}:{metric.metric_name}:{warning}" for warning in metric.warnings)
    return sorted(set(warnings))


def _finalize_result(root: Path, result: ArmTuneServerSearchResult) -> ArmTuneServerSearchResult:
    artifacts = _collect_artifacts(root)
    final = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(root / "server_optimize_result.json", final.to_dict())
    write_artifact_manifest(root / "artifact_manifest.json", artifacts)
    validation = validate_artifact_manifest(root)
    atomic_write_json(root / "artifact_validation.json", validation.to_dict())
    final = replace(final, artifact_validation=validation)
    atomic_write_json(root / "server_optimize_result.json", final.to_dict())
    return final


def _collect_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    skipped = {"artifact_manifest.json", "artifact_validation.json", "server_optimize_result.json"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skipped:
            continue
        label = path.relative_to(root).as_posix().replace("/", "_").replace(".", "_")
        artifacts[label] = artifact_record(path, label=label, root=root)
    return artifacts


def _add_affinity_candidate(
    candidates: list[ServerSearchCandidate],
    seen: set[str],
    base: ServerSearchCandidate,
    mode: str,
    cpus: list[int],
) -> None:
    if not cpus:
        return
    mask = _mask(cpus)
    if mask == base.allowed_cpu_mask:
        return
    identity = f"{mode}:{mask}"
    if identity in seen:
        return
    seen.add(identity)
    candidates.append(
        replace(
            base,
            candidate_id=f"affinity-{mode}-{mask.replace(',', '_')}",
            stage="affinity",
            affinity_mode=mode,
            cpu_affinity=cpus,
            allowed_cpu_mask=mask,
            is_baseline=False,
            description=f"{mode} process CPU affinity",
        )
    )


def _spread(cpus: list[int], count: int) -> list[int]:
    if count >= len(cpus):
        return list(cpus)
    if count == 1:
        return [cpus[0]]
    selected: list[int] = []
    last_index = len(cpus) - 1
    for index in range(count):
        position = round(index * last_index / (count - 1))
        selected.append(cpus[position])
    return sorted(set(selected))


def _mask(cpus: list[int] | None) -> str:
    if not cpus:
        return "scheduler-default"
    return ",".join(str(cpu) for cpu in cpus)
