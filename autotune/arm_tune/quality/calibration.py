from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest
from autotune.arm_tune.config import parse_benchmark_config
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.aggregation import aggregate_quality_results
from autotune.arm_tune.quality.boottrace import boottrace_prompt_contract
from autotune.arm_tune.quality.normalization import NormalizationPolicy, normalization_policy_from_payload
from autotune.arm_tune.quality.schemas import CandidateQualityResult, QualityAggregateResult, QualityCase, QualityCaseResult, QualitySuiteManifest
from autotune.arm_tune.quality.suite import load_quality_suite, prompt_suite_from_quality_cases, suite_snapshot_payload
from autotune.arm_tune.quality.validators import validate_quality_case
from autotune.arm_tune.schemas import BenchmarkArtifactManifest, LlamaServerBenchmarkResult, LlamaServerJobSpec, TargetSpec, to_plain, utc_now
from autotune.arm_tune.targets.base import ArmTuneTarget
from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.utils.config import load_config


QUALITY_CALIBRATION_OPERATION = "quality_calibrate.llama_server"
CALIBRATION_EXTRA_KEYS = {"quality_gate", "normalization"}
QUALITY_GATE_KEYS = {"enabled", "suite_path"}


@dataclass(frozen=True)
class StructuredCapabilityStatus:
    grammar: str
    json_schema: str
    response_format: str
    openai_compatible_structured_output: str
    seed: str
    temperature: str
    deterministic_sampling: str
    probe_status: str
    evidence: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class QualityCalibrationConfig:
    target: TargetSpec
    job: LlamaServerJobSpec
    quality_cases: list[QualityCase]
    quality_manifest: QualitySuiteManifest
    quality_suite_payload: dict[str, Any]
    normalization_policy: NormalizationPolicy
    raw_payload: dict[str, Any]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class QualityCalibrationResult:
    run_id: str
    operation: str
    target_type: str
    result_origin: str
    suite_case_count: int
    request_success_count: int
    measured_request_count: int
    raw_json_valid_count: int
    normalized_json_valid_count: int
    required_quality_cases_passed: int
    required_quality_cases_total: int
    critical_failure_count: int
    failure_category_histogram: dict[str, int]
    quality_gate_status: str
    suitability_state: str
    suitability_reasons: list[str]
    structured_capability: StructuredCapabilityStatus
    quality_result: CandidateQualityResult
    benchmark_result: LlamaServerBenchmarkResult
    artifacts: BenchmarkArtifactManifest = BenchmarkArtifactManifest()
    artifact_directory: str = ""
    artifact_validation: Any = None
    created_at_utc: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def load_quality_calibration_config(path: str | Path) -> QualityCalibrationConfig:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("quality calibration config must be an object")
    unknown = set(payload) - (
        {
            "schema_version",
            "target",
            "result_origin",
            "runtime",
            "executable",
            "model",
            "server",
            "workload",
            "artifacts",
        }
        | CALIBRATION_EXTRA_KEYS
    )
    if unknown:
        raise ArmTuneConfigError(f"unknown quality calibration config field(s): {sorted(unknown)}")
    quality_gate = payload.get("quality_gate", {})
    if quality_gate is None:
        quality_gate = {}
    if not isinstance(quality_gate, dict):
        raise ArmTuneConfigError("quality_gate must be an object")
    _reject_unknown(quality_gate, QUALITY_GATE_KEYS, "quality_gate")
    if not bool(quality_gate.get("enabled", True)):
        raise ArmTuneConfigError("quality_gate.enabled must be true for quality calibration")
    suite_path = quality_gate.get("suite_path")
    if not isinstance(suite_path, str) or not suite_path.strip():
        raise ArmTuneConfigError("quality_gate.suite_path is required for quality calibration")
    try:
        normalization_policy = normalization_policy_from_payload(payload.get("normalization"))
    except ValueError as exc:
        raise ArmTuneConfigError(str(exc)) from exc
    benchmark_payload = {key: value for key, value in payload.items() if key not in CALIBRATION_EXTRA_KEYS}
    target, job = parse_benchmark_config(benchmark_payload)
    if not isinstance(job, LlamaServerJobSpec):
        raise ArmTuneConfigError("quality calibration requires runtime.adapter='llama_server'")
    _, cases, manifest, suite_payload = load_quality_suite(suite_path)
    return QualityCalibrationConfig(
        target=target,
        job=job,
        quality_cases=cases,
        quality_manifest=manifest,
        quality_suite_payload=suite_payload,
        normalization_policy=normalization_policy,
        raw_payload=dict(payload),
    )


def run_quality_calibration(
    *,
    target: ArmTuneTarget,
    config: QualityCalibrationConfig,
    run_id: str,
    calibration_dir: str | Path,
) -> QualityCalibrationResult:
    root = Path(calibration_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "quality_calibration_job.json", _job_payload(target, config, run_id))
    atomic_write_json(root / "effective_config.json", _effective_config_payload(target, config))
    atomic_write_json(root / "quality_suite_manifest.json", config.quality_manifest.to_dict())
    atomic_write_json(root / "quality_suite_snapshot.json", config.quality_suite_payload)
    atomic_write_json(root / "normalization_policy.json", config.normalization_policy.to_dict())
    contract = boottrace_prompt_contract(
        suite_manifest=config.quality_manifest,
        cases=config.quality_cases,
        generation_settings=config.job.server.to_dict(),
        normalization_policy=config.normalization_policy.to_dict(),
    )
    atomic_write_json(root / "prompt_contract.json", contract)
    prompt_suite_path = _write_prompt_suite_snapshot(root, target, config.quality_cases)
    job = replace(
        config.job,
        run_id=run_id,
        target_type=target.spec.type,
        remote_module=target.spec.remote_module,
        target_run_directory=str(root),
        workload=(
            replace(config.job.workload, prompt_suite_path=str(prompt_suite_path))
            if target.spec.type == "local" and prompt_suite_path is not None
            else config.job.workload
        ),
    )
    benchmark_result = target.run_server_benchmark(job, root / "benchmark")
    quality_result = evaluate_benchmark_quality(
        benchmark_result=benchmark_result,
        candidate_id="calibration",
        cases=config.quality_cases,
        manifest=config.quality_manifest,
        normalization_policy=config.normalization_policy,
    )
    capability = structured_capability_from_benchmark(benchmark_result)
    result = _calibration_result(
        run_id=run_id,
        target=target,
        benchmark_result=benchmark_result,
        quality_result=quality_result,
        manifest=config.quality_manifest,
        capability=capability,
        root=root,
    )
    atomic_write_json(root / "quality_calibration_result.json", result.to_dict())
    artifacts = _collect_artifacts(root)
    result = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(root / "quality_calibration_result.json", result.to_dict())
    validation = validate_artifact_manifest(root)
    result = replace(result, artifact_validation=validation)
    atomic_write_json(root / "artifact_validation.json", validation.to_dict())
    artifacts = _collect_artifacts(root)
    result = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(root / "quality_calibration_result.json", result.to_dict())
    return result


def evaluate_benchmark_quality(
    *,
    benchmark_result: LlamaServerBenchmarkResult,
    candidate_id: str,
    cases: list[QualityCase],
    manifest: QualitySuiteManifest,
    normalization_policy: NormalizationPolicy | None = None,
) -> CandidateQualityResult:
    case_by_id = {case.case_id: case for case in cases}
    case_results: list[QualityCaseResult] = []
    observed_cases: set[str] = set()
    for request in benchmark_result.requests:
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
                normalization_policy=normalization_policy,
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
                    normalization_policy=normalization_policy,
                )
            )
    return CandidateQualityResult(
        candidate_id=candidate_id,
        suite_manifest=manifest,
        aggregate=aggregate_quality_results(case_results),
        cases=case_results,
    )


def structured_capability_from_benchmark(result: LlamaServerBenchmarkResult) -> StructuredCapabilityStatus:
    inspection = result.inspection
    if inspection is None or inspection.help_status != "passed":
        return StructuredCapabilityStatus(
            grammar="probe_failed",
            json_schema="probe_failed",
            response_format="probe_failed",
            openai_compatible_structured_output="probe_failed",
            seed="probe_failed",
            temperature="probe_failed",
            deterministic_sampling="probe_failed",
            probe_status="probe_failed",
            evidence={},
        )
    flags = set(inspection.supported_flags)
    seed = "supported" if "--seed" in flags else "unsupported"
    temperature = "supported" if flags.intersection({"--temp", "--temperature"}) else "unsupported"
    deterministic = "supported" if seed == "supported" and temperature == "supported" else "unverified"
    grammar = "supported" if flags.intersection({"--grammar", "--grammar-file"}) else "unsupported"
    json_schema = "supported" if flags.intersection({"--json-schema", "--schema"}) else "unsupported"
    response_format = "supported" if "--response-format" in flags else "unsupported"
    openai_structured = "unverified"
    if response_format == "supported" or json_schema == "supported":
        openai_structured = "supported"
    return StructuredCapabilityStatus(
        grammar=grammar,
        json_schema=json_schema,
        response_format=response_format,
        openai_compatible_structured_output=openai_structured,
        seed=seed,
        temperature=temperature,
        deterministic_sampling=deterministic,
        probe_status="completed",
        evidence={
            "schema_version": SCHEMA_VERSION,
            "supported_flags": sorted(flags),
            "source_commit": inspection.source_commit,
            "runtime_build": inspection.runtime_build,
            "fixture_marker_detected": inspection.fixture_marker_detected,
        },
    )


def _calibration_result(
    *,
    run_id: str,
    target: ArmTuneTarget,
    benchmark_result: LlamaServerBenchmarkResult,
    quality_result: CandidateQualityResult,
    manifest: QualitySuiteManifest,
    capability: StructuredCapabilityStatus,
    root: Path,
) -> QualityCalibrationResult:
    measured = [request for request in benchmark_result.requests if request.phase == "measured"]
    request_success = sum(1 for request in measured if request.request_status == "completed" and request.stream_parse_status == "parsed")
    raw_valid = sum(1 for item in quality_result.cases if item.parser_status == "raw_json_valid")
    normalized_valid = sum(1 for item in quality_result.cases if item.parser_status == "normalized_json_valid")
    required_passed, required_total = _required_case_counts(quality_result.cases, manifest)
    suitability, reasons = _suitability(benchmark_result, quality_result)
    return QualityCalibrationResult(
        run_id=run_id,
        operation=QUALITY_CALIBRATION_OPERATION,
        target_type=target.spec.type,
        result_origin=benchmark_result.result_origin,
        suite_case_count=manifest.case_count,
        request_success_count=request_success,
        measured_request_count=len(measured),
        raw_json_valid_count=raw_valid,
        normalized_json_valid_count=normalized_valid,
        required_quality_cases_passed=required_passed,
        required_quality_cases_total=required_total,
        critical_failure_count=quality_result.aggregate.critical_validator_failures,
        failure_category_histogram=dict(quality_result.aggregate.failure_category_histogram),
        quality_gate_status=quality_result.aggregate.quality_gate_status,
        suitability_state=suitability,
        suitability_reasons=reasons,
        structured_capability=capability,
        quality_result=quality_result,
        benchmark_result=benchmark_result,
        artifact_directory=str(root),
        created_at_utc=utc_now(),
    )


def _suitability(
    benchmark_result: LlamaServerBenchmarkResult,
    quality_result: CandidateQualityResult,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if benchmark_result.validation_status != "passed":
        reasons.append(f"validation_status={benchmark_result.validation_status}")
    if benchmark_result.startup_status != "ready" or benchmark_result.readiness_status != "ready":
        reasons.append("server lifecycle was not ready")
    if benchmark_result.request_status != "completed" or benchmark_result.stream_parse_status != "parsed":
        reasons.append("request execution or stream parsing failed")
    if benchmark_result.cleanup_status not in {"passed", "forced"}:
        reasons.append(f"cleanup_status={benchmark_result.cleanup_status}")
    if benchmark_result.artifact_validation.validation_status != "passed":
        reasons.append("benchmark artifact validation failed")
    if reasons:
        return "execution_failed", sorted(set(reasons))
    aggregate = quality_result.aggregate
    if aggregate.quality_gate_status == "passed":
        return "suitable", []
    if not quality_result.cases:
        return "inconclusive", ["no quality case results were produced"]
    return "unsuitable", aggregate.failure_reasons or ["quality gate failed"]


def _required_case_counts(cases: list[QualityCaseResult], manifest: QualitySuiteManifest) -> tuple[int, int]:
    grouped: dict[str, list[QualityCaseResult]] = defaultdict(list)
    for result in cases:
        if result.required:
            grouped[result.case_id].append(result)
    passed = sum(1 for results in grouped.values() if all(item.status == "passed" for item in results))
    return passed, manifest.required_case_count


def _write_prompt_suite_snapshot(root: Path, target: ArmTuneTarget, cases: list[QualityCase]) -> Path | None:
    if not cases:
        return None
    prompt_suite = prompt_suite_from_quality_cases(cases)
    path = root / "prompt_suite_snapshot.json"
    atomic_write_json(path, prompt_suite)
    return path.resolve() if target.spec.type == "local" else path


def _job_payload(target: ArmTuneTarget, config: QualityCalibrationConfig, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": QUALITY_CALIBRATION_OPERATION,
        "run_id": run_id,
        "target": target.spec.sanitized_dict(),
        "base_job": config.job.to_dict(),
        "quality_suite_manifest": config.quality_manifest.to_dict(),
        "normalization_policy": config.normalization_policy.to_dict(),
    }


def _effective_config_payload(target: ArmTuneTarget, config: QualityCalibrationConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target.spec.sanitized_dict(),
        "runtime_adapter": "llama_server",
        "server": config.job.server.to_dict(),
        "workload": config.job.workload.to_dict(),
        "quality_gate": {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "suite_path": config.quality_manifest.source_path,
        },
        "normalization": config.normalization_policy.to_dict(),
    }


def _collect_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    skipped = {"artifact_manifest.json", "artifact_validation.json", "quality_calibration_result.json"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skipped:
            continue
        label = path.relative_to(root).as_posix().replace("/", "__")
        artifacts[label] = artifact_record(path, label=label, root=root)
    write_artifact_manifest(root / "artifact_manifest.json", artifacts)
    return artifacts


def _reject_unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ArmTuneConfigError(f"unknown {label} field(s): {sorted(unknown)}")
