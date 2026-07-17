from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import ArmBenchmarkResult, HostInfo, LlamaBenchJobSpec, to_plain, utc_now
from autotune.arm_tune.search.candidates import (
    SearchCandidate,
    baseline_candidate,
    generate_affinity_candidates,
    generate_batch_candidates,
    generate_thread_candidates,
)
from autotune.arm_tune.search.statistics import MetricDistribution, NoiseBandDecision, classify_noise_band, summarize_metric
from autotune.arm_tune.targets.base import ArmTuneTarget
from autotune.arm_tune.targets.protocol import atomic_write_json


PRIMARY_METRIC = "decode_tokens_per_second"
SECONDARY_METRIC = "prefill_tokens_per_second"


@dataclass(frozen=True)
class SearchConfig:
    warmup_runs: int = 2
    search_runs: int = 5
    confirmation_runs: int = 10
    cooldown_seconds: float = 2.0
    random_seed: int = 42
    insert_paired_baseline_every: int = 4
    noise_band_percent: float = 2.0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SearchRepetitionRecord:
    sequence_index: int
    candidate_id: str
    repetition_index: int
    phase: str
    stage: str
    started_at_utc: str
    ended_at_utc: str
    validation_status: str
    execution_status: str
    parse_status: str
    benchmark_status: str
    result_origin: str
    prefill_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    artifact_directory: str
    host_fingerprint: str
    allowed_cpu_mask: str
    target_load: dict[str, Any] | None = None
    failure_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SearchStageResult:
    stage: str
    candidate_ids: list[str]
    selected_candidate_id: str
    decision: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class SearchRecommendation:
    status: str
    candidate_id: str | None
    message: str
    quality_gate_status: str = "not_evaluated"
    noise_band_decision: NoiseBandDecision | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ArmTuneSearchResult:
    run_id: str
    status: str
    primary_metric: str
    quality_gate_status: str
    random_seed: int
    search_config: SearchConfig
    baseline_candidate: SearchCandidate
    selected_candidate: SearchCandidate
    recommendation: SearchRecommendation
    candidates: list[SearchCandidate]
    repetitions: list[SearchRepetitionRecord]
    distributions: dict[str, dict[str, MetricDistribution]]
    stages: list[SearchStageResult]
    warnings: list[str] = field(default_factory=list)
    artifact_directory: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def run_search(
    *,
    target: ArmTuneTarget,
    base_job: LlamaBenchJobSpec,
    config: SearchConfig,
    run_id: str,
    optimize_dir: str | Path,
) -> ArmTuneSearchResult:
    root = Path(optimize_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "optimize_config.json", config.to_dict())
    inspection = target.inspect()
    topology = inspection.host.topology
    baseline = baseline_candidate(base_job)
    rng = random.Random(config.random_seed)
    repetitions: list[SearchRepetitionRecord] = []
    stages: list[SearchStageResult] = []
    candidates_by_id: dict[str, SearchCandidate] = {baseline.candidate_id: baseline}
    supported_flags: list[str] = []

    warmup_schedule = [baseline] * config.warmup_runs
    _run_schedule(
        target=target,
        base_job=base_job,
        run_id=run_id,
        root=root,
        config=config,
        schedule=warmup_schedule,
        phase="warmup",
        repetitions=repetitions,
    )

    thread_candidates = generate_thread_candidates(base_job, topology)
    for candidate in thread_candidates:
        candidates_by_id[candidate.candidate_id] = candidate
    stage1 = _run_stage(
        target=target,
        base_job=base_job,
        run_id=run_id,
        root=root,
        config=config,
        stage_name="thread_count",
        candidates=thread_candidates,
        baseline=baseline,
        rng=rng,
        repetitions=repetitions,
    )
    current = _select_best_candidate(thread_candidates, repetitions, baseline)
    stages.append(_stage_result("thread_count", thread_candidates, current, stage1, config))

    supported_flags = _first_supported_flags(repetitions)
    batch_candidates = generate_batch_candidates(current, base_job, supported_flags=supported_flags)
    for candidate in batch_candidates:
        candidates_by_id[candidate.candidate_id] = candidate
    if batch_candidates:
        stage_candidates = [current, *batch_candidates]
        stage2 = _run_stage(
            target=target,
            base_job=base_job,
            run_id=run_id,
            root=root,
            config=config,
            stage_name="batch_parameters",
            candidates=stage_candidates,
            baseline=baseline,
            rng=rng,
            repetitions=repetitions,
        )
        current = _select_best_candidate(stage_candidates, repetitions, current)
        stages.append(_stage_result("batch_parameters", stage_candidates, current, stage2, config))
    else:
        stages.append(
            SearchStageResult(
                stage="batch_parameters",
                candidate_ids=[],
                selected_candidate_id=current.candidate_id,
                decision={"status": "skipped", "reason": "no supported optional batch flags were inspected"},
            )
        )

    affinity_candidates = generate_affinity_candidates(current, topology)
    for candidate in affinity_candidates:
        candidates_by_id[candidate.candidate_id] = candidate
    if affinity_candidates:
        stage_candidates = [current, *affinity_candidates]
        stage3 = _run_stage(
            target=target,
            base_job=base_job,
            run_id=run_id,
            root=root,
            config=config,
            stage_name="affinity",
            candidates=stage_candidates,
            baseline=baseline,
            rng=rng,
            repetitions=repetitions,
        )
        current = _select_best_candidate(stage_candidates, repetitions, current)
        stages.append(_stage_result("affinity", stage_candidates, current, stage3, config))
    else:
        stages.append(
            SearchStageResult(
                stage="affinity",
                candidate_ids=[],
                selected_candidate_id=current.candidate_id,
                decision={"status": "skipped", "reason": "no meaningful affinity candidates for visible topology"},
            )
        )

    confirmation_schedule = [baseline] * config.confirmation_runs
    if current.candidate_id != baseline.candidate_id:
        confirmation_schedule.extend([current] * config.confirmation_runs)
    _run_schedule(
        target=target,
        base_job=base_job,
        run_id=run_id,
        root=root,
        config=config,
        schedule=confirmation_schedule,
        phase="confirmation",
        repetitions=repetitions,
    )

    candidates = list(candidates_by_id.values())
    distributions = _all_distributions(candidates, repetitions)
    baseline_distribution = distributions.get(baseline.candidate_id, {}).get(PRIMARY_METRIC)
    current_distribution = distributions.get(current.candidate_id, {}).get(PRIMARY_METRIC)
    if baseline_distribution is None:
        baseline_distribution = summarize_metric(PRIMARY_METRIC, [], failed_count=0)
    if current_distribution is None:
        current_distribution = summarize_metric(PRIMARY_METRIC, [], failed_count=0)
    decision = classify_noise_band(
        baseline=baseline_distribution,
        candidate=current_distribution,
        noise_band_percent=config.noise_band_percent,
    )
    recommendation = _recommendation(current, baseline, decision)
    warnings = _collect_warnings(distributions)
    result = ArmTuneSearchResult(
        run_id=run_id,
        status="completed" if repetitions else "failed",
        primary_metric=PRIMARY_METRIC,
        quality_gate_status="not_evaluated",
        random_seed=config.random_seed,
        search_config=config,
        baseline_candidate=baseline,
        selected_candidate=current,
        recommendation=recommendation,
        candidates=candidates,
        repetitions=repetitions,
        distributions=distributions,
        stages=stages,
        warnings=warnings,
        artifact_directory=str(root),
    )
    atomic_write_json(root / "search_plan.json", _search_plan(config, candidates, stages))
    atomic_write_json(root / "search_result.json", result.to_dict())
    return result


def build_stage_schedule(
    *,
    candidates: list[SearchCandidate],
    baseline: SearchCandidate,
    repetitions_per_candidate: int,
    insert_paired_baseline_every: int,
    rng: random.Random,
) -> list[SearchCandidate]:
    if repetitions_per_candidate < 1:
        return []
    nonbaseline = [candidate for candidate in candidates if candidate.candidate_id != baseline.candidate_id]
    rng.shuffle(nonbaseline)
    schedule: list[SearchCandidate] = [baseline] * repetitions_per_candidate
    nonbaseline_seen = 0
    for candidate in nonbaseline:
        for _ in range(repetitions_per_candidate):
            schedule.append(candidate)
            nonbaseline_seen += 1
            if insert_paired_baseline_every > 0 and nonbaseline_seen % insert_paired_baseline_every == 0:
                schedule.append(baseline)
    return schedule


def _run_stage(
    *,
    target: ArmTuneTarget,
    base_job: LlamaBenchJobSpec,
    run_id: str,
    root: Path,
    config: SearchConfig,
    stage_name: str,
    candidates: list[SearchCandidate],
    baseline: SearchCandidate,
    rng: random.Random,
    repetitions: list[SearchRepetitionRecord],
) -> list[SearchRepetitionRecord]:
    before = len(repetitions)
    schedule = build_stage_schedule(
        candidates=candidates,
        baseline=baseline,
        repetitions_per_candidate=config.search_runs,
        insert_paired_baseline_every=config.insert_paired_baseline_every,
        rng=rng,
    )
    _run_schedule(
        target=target,
        base_job=base_job,
        run_id=run_id,
        root=root,
        config=config,
        schedule=schedule,
        phase=stage_name,
        repetitions=repetitions,
    )
    return repetitions[before:]


def _run_schedule(
    *,
    target: ArmTuneTarget,
    base_job: LlamaBenchJobSpec,
    run_id: str,
    root: Path,
    config: SearchConfig,
    schedule: list[SearchCandidate],
    phase: str,
    repetitions: list[SearchRepetitionRecord],
) -> None:
    candidate_counts: dict[str, int] = {}
    for local_index, candidate in enumerate(schedule, start=1):
        candidate_counts[candidate.candidate_id] = candidate_counts.get(candidate.candidate_id, 0) + 1
        sequence_index = len(repetitions) + 1
        repetition_index = candidate_counts[candidate.candidate_id]
        repetition_dir = root / "repetitions" / f"{sequence_index:04d}"
        benchmark_job = replace(
            base_job,
            run_id=f"{run_id}_{sequence_index:04d}",
            target_run_directory=str(repetition_dir),
            benchmark=candidate.settings_for(base_job.benchmark),
        )
        started = utc_now()
        result = target.run_benchmark(benchmark_job, repetition_dir)
        ended = utc_now()
        repetitions.append(_record_repetition(sequence_index, repetition_index, phase, candidate, started, ended, result))
        if config.cooldown_seconds > 0 and local_index < len(schedule):
            time.sleep(config.cooldown_seconds)


def _record_repetition(
    sequence_index: int,
    repetition_index: int,
    phase: str,
    candidate: SearchCandidate,
    started: str,
    ended: str,
    result: ArmBenchmarkResult,
) -> SearchRepetitionRecord:
    metrics = _metrics(result)
    return SearchRepetitionRecord(
        sequence_index=sequence_index,
        candidate_id=candidate.candidate_id,
        repetition_index=repetition_index,
        phase=phase,
        stage=candidate.stage,
        started_at_utc=started,
        ended_at_utc=ended,
        validation_status=result.validation_status,
        execution_status=result.execution_status,
        parse_status=result.parse_status,
        benchmark_status=result.benchmark_status,
        result_origin=result.result_origin,
        prefill_tokens_per_second=metrics.get(SECONDARY_METRIC),
        decode_tokens_per_second=metrics.get(PRIMARY_METRIC),
        artifact_directory=result.artifact_directory,
        host_fingerprint=_host_fingerprint(result.host),
        allowed_cpu_mask=candidate.allowed_cpu_mask,
        target_load=None,
        failure_reason="" if result.benchmark_status == "passed" else result.message,
    )


def _metrics(result: ArmBenchmarkResult) -> dict[str, float]:
    parse_result = result.trial.parse_result if result.trial and result.trial.parse_result else None
    if parse_result is None:
        return {}
    return {metric.name: metric.value for metric in parse_result.metrics if metric.validation_status == "passed"}


def _host_fingerprint(host: HostInfo) -> str:
    return "|".join(
        [
            host.architecture.system,
            host.architecture.normalized_machine,
            host.architecture.kernel_release,
            host.cpu_identity.model_name,
            str(host.topology.logical_cpus),
            host.topology.online_cpus,
        ]
    )


def _first_supported_flags(repetitions: list[SearchRepetitionRecord]) -> list[str]:
    for record in repetitions:
        result_path = Path(record.artifact_directory) / "benchmark_result.json"
        if not result_path.exists():
            continue
        try:
            import json

            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        flags = payload.get("inspection", {}).get("supported_flags", [])
        if isinstance(flags, list):
            return [str(flag) for flag in flags]
    return []


def _select_best_candidate(
    candidates: list[SearchCandidate],
    repetitions: list[SearchRepetitionRecord],
    fallback: SearchCandidate,
) -> SearchCandidate:
    distributions = _all_distributions(candidates, repetitions)
    best = fallback
    best_value = distributions.get(best.candidate_id, {}).get(PRIMARY_METRIC)
    best_median = best_value.median if best_value else None
    for candidate in candidates:
        distribution = distributions.get(candidate.candidate_id, {}).get(PRIMARY_METRIC)
        if distribution is None or distribution.failed_count > 0 or distribution.median is None:
            continue
        if best_median is None or distribution.median > best_median:
            best = candidate
            best_median = distribution.median
    return best


def _all_distributions(
    candidates: list[SearchCandidate],
    repetitions: list[SearchRepetitionRecord],
) -> dict[str, dict[str, MetricDistribution]]:
    result: dict[str, dict[str, MetricDistribution]] = {}
    for candidate in candidates:
        records = [
            record
            for record in repetitions
            if record.candidate_id == candidate.candidate_id and record.phase != "warmup"
        ]
        failed = sum(1 for record in records if record.benchmark_status != "passed")
        decode = [record.decode_tokens_per_second for record in records if record.decode_tokens_per_second is not None]
        prefill = [record.prefill_tokens_per_second for record in records if record.prefill_tokens_per_second is not None]
        result[candidate.candidate_id] = {
            PRIMARY_METRIC: summarize_metric(PRIMARY_METRIC, decode, failed_count=failed),
            SECONDARY_METRIC: summarize_metric(SECONDARY_METRIC, prefill, failed_count=failed),
        }
    return result


def _stage_result(
    stage: str,
    candidates: list[SearchCandidate],
    selected: SearchCandidate,
    stage_records: list[SearchRepetitionRecord],
    config: SearchConfig,
) -> SearchStageResult:
    distributions = _all_distributions(candidates, stage_records)
    baseline_distribution = distributions.get("baseline", {}).get(PRIMARY_METRIC)
    selected_distribution = distributions.get(selected.candidate_id, {}).get(PRIMARY_METRIC)
    if baseline_distribution is None:
        baseline_distribution = summarize_metric(PRIMARY_METRIC, [], failed_count=0)
    if selected_distribution is None:
        selected_distribution = summarize_metric(PRIMARY_METRIC, [], failed_count=0)
    decision = classify_noise_band(
        baseline=baseline_distribution,
        candidate=selected_distribution,
        noise_band_percent=config.noise_band_percent,
    )
    return SearchStageResult(
        stage=stage,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
        selected_candidate_id=selected.candidate_id,
        decision=decision.to_dict(),
    )


def _recommendation(
    selected: SearchCandidate,
    baseline: SearchCandidate,
    decision: NoiseBandDecision,
) -> SearchRecommendation:
    if selected.candidate_id == baseline.candidate_id:
        return SearchRecommendation(
            status="baseline_retained",
            candidate_id=baseline.candidate_id,
            message="No non-baseline candidate exceeded the configured noise band.",
            noise_band_decision=decision,
        )
    if decision.classification == "improved":
        return SearchRecommendation(
            status="recommended",
            candidate_id=selected.candidate_id,
            message="Best observed performance candidate under the tested llama-bench configuration.",
            noise_band_decision=decision,
        )
    return SearchRecommendation(
        status="inconclusive",
        candidate_id=selected.candidate_id,
        message="Best observed candidate is not recommended as an improvement because it is not outside the configured noise band.",
        noise_band_decision=decision,
    )


def _collect_warnings(distributions: dict[str, dict[str, MetricDistribution]]) -> list[str]:
    warnings: list[str] = []
    for candidate_id, metrics in distributions.items():
        for metric in metrics.values():
            for warning in metric.warnings:
                warnings.append(f"{candidate_id}:{metric.metric_name}:{warning}")
    return warnings


def _search_plan(
    config: SearchConfig,
    candidates: list[SearchCandidate],
    stages: list[SearchStageResult],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "armtune_search_plan",
        "random_seed": config.random_seed,
        "search_config": config.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "stages": [stage.to_dict() for stage in stages],
    }
