from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.build_profiles.builder import inspect_build, run_build_profile, write_build_plan
from autotune.arm_tune.build_profiles.comparison import run_profile_comparison
from autotune.arm_tune.config import AFFINITY_MODE_CHOICES, default_local_target, load_benchmark_config, load_optimize_config, load_target_spec
from autotune.arm_tune.errors import ArmTuneError
from autotune.arm_tune.passport.generator import generate_passport
from autotune.arm_tune.passport.verification import verify_passport
from autotune.arm_tune.quality.calibration import QualityCalibrationResult, load_quality_calibration_config, run_quality_calibration
from autotune.arm_tune.schemas import (
    ArmBenchmarkResult,
    DoctorStatuses,
    GenerationMetricDistribution,
    LlamaBenchParseResult,
    LlamaServerBenchmarkResult,
    LlamaServerJobSpec,
    ProductionValidation,
    StatusCheck,
    TargetSpec,
)
from autotune.arm_tune.search.runner import SearchConfig, run_search
from autotune.arm_tune.server_search.runner import ArmTuneServerSearchResult, PRIMARY_METRIC as SERVER_PRIMARY_METRIC, SECONDARY_METRIC as SERVER_SECONDARY_METRIC, run_server_search
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.arm_tune.targets.ssh import SshArmTarget
from autotune.arm_tune.targets.worker import main as worker_main
from autotune.resource.budget import ResourceBudget
from autotune.resource.executor_capabilities import collect_executor_capabilities
from autotune.resource.run_state import create_run, finish_run


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except ArmTuneError as exc:
        print(f"ArmTune error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotuneai arm",
        description="ArmTune Arm platform inspection, reproducible benchmarking, runtime tuning, and Performance Passport generation.",
    )
    subparsers = parser.add_subparsers(dest="arm_command")

    doctor = subparsers.add_parser("doctor", help="Validate ArmTune controller and target configuration.")
    doctor.add_argument("--target-config")
    doctor.add_argument("--require-production", action="store_true")
    doctor.set_defaults(handler=_cmd_doctor)

    inspect_host = subparsers.add_parser("inspect-host", help="Inspect an ArmTune local or SSH target host.")
    inspect_host.add_argument("--target-config")
    inspect_host.set_defaults(handler=_cmd_inspect_host)

    benchmark = subparsers.add_parser("benchmark", help="Run a fixed ArmTune benchmark job.")
    benchmark.add_argument("--config", required=True)
    benchmark.set_defaults(handler=_cmd_benchmark)

    optimize = subparsers.add_parser("optimize", help="Run bounded ArmTune runtime performance search.")
    optimize.add_argument("--config", required=True)
    optimize.add_argument("--warmup-runs", type=int)
    optimize.add_argument("--search-runs", type=int)
    optimize.add_argument("--confirmation-runs", type=int)
    optimize.add_argument("--cooldown-seconds", type=float)
    optimize.add_argument("--random-seed", type=int)
    optimize.add_argument("--insert-paired-baseline-every", type=int)
    optimize.add_argument("--noise-band-percent", type=float)
    optimize.add_argument("--max-candidates", type=int)
    optimize.add_argument("--affinity-modes", help="Comma-separated affinity modes to include in search.")
    optimize.set_defaults(handler=_cmd_optimize)

    quality_calibrate = subparsers.add_parser(
        "quality-calibrate",
        help="Run optional experimental deterministic quality calibration for a fixed llama-server profile.",
    )
    quality_calibrate.add_argument("--config", required=True)
    quality_calibrate.set_defaults(handler=_cmd_quality_calibrate)

    build_plan = subparsers.add_parser("build-plan", help="Create a versioned ArmTune build-profile plan without executing a build.")
    build_plan.add_argument("--config", required=True)
    build_plan.set_defaults(handler=_cmd_build_plan)

    build_profile = subparsers.add_parser("build-profile", help="Inspect existing build-profile artifacts and write build evidence.")
    build_profile.add_argument("--config", required=True)
    build_profile.set_defaults(handler=_cmd_build_profile)

    inspect_build_cmd = subparsers.add_parser("inspect-build", help="Inspect existing build artifacts for ArmTune build evidence.")
    inspect_build_cmd.add_argument("--config", required=True)
    inspect_build_cmd.set_defaults(handler=_cmd_inspect_build)

    profile_compare = subparsers.add_parser("profile-compare", help="Compare build profiles using artifact-backed measurements.")
    profile_compare.add_argument("--config", required=True)
    profile_compare.set_defaults(handler=_cmd_profile_compare)

    passport = subparsers.add_parser("passport", help="Generate an ArmTune Performance Passport from a profile comparison result.")
    passport.add_argument("--comparison-result", required=True)
    passport.set_defaults(handler=_cmd_passport)

    verify_passport_cmd = subparsers.add_parser("verify-passport", help="Verify an ArmTune Performance Passport.")
    verify_passport_cmd.add_argument("--passport", required=True)
    verify_passport_cmd.set_defaults(handler=_cmd_verify_passport)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-spec", required=True)
    worker.add_argument("--run-dir", required=True)
    worker.set_defaults(handler=_cmd_worker)
    _hide_subparser_help(subparsers, "worker")

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    target_configured = bool(getattr(args, "target_config", None))
    spec = _target_spec_from_args(args)
    local_validation = LocalArmTarget(default_local_target()).validate_production_target()
    configured_validation: ProductionValidation | None = None
    configured_prerequisites: dict[str, Any] | None = None
    configured_error: str | None = None
    if target_configured:
        target = _target_from_spec(spec)
        try:
            if spec.type == "ssh":
                prerequisites = target.check_prerequisites()
                configured_prerequisites = prerequisites.to_dict()
                configured_validation = prerequisites.production_validation
            else:
                configured_validation = target.validate_production_target()
        except ArmTuneError as exc:
            configured_error = str(exc)
    statuses = DoctorStatuses(
        controller_status=StatusCheck(
            status="ready",
            reason="controller diagnostics completed; controller architecture is not production benchmark evidence",
        ),
        local_target_status=_status_from_validation(local_validation),
        configured_target_status=_configured_status(
            configured=target_configured,
            validation=configured_validation,
            error=configured_error,
        ),
        production_benchmark_ready=_production_ready(
            target_configured=target_configured,
            local_validation=local_validation,
            configured_validation=configured_validation,
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "armtune_doctor",
        "target": spec.sanitized_dict(),
        "controller_status": statuses.controller_status.to_dict(),
        "local_target_status": statuses.local_target_status.to_dict(),
        "configured_target_status": statuses.configured_target_status.to_dict(),
        "production_benchmark_ready": statuses.production_benchmark_ready,
        "controller_diagnostics": {
            "executor_capabilities": collect_executor_capabilities(),
        },
        "configured_target_prerequisites": configured_prerequisites,
        "production_validation": configured_validation.to_dict() if configured_validation else None,
        "notes": _doctor_notes(spec, statuses),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_production and not statuses.production_benchmark_ready:
        return 2
    return 0


def _cmd_inspect_host(args: argparse.Namespace) -> int:
    spec = _target_spec_from_args(args)
    target = _target_from_spec(spec)
    result = target.inspect()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    spec, job = load_benchmark_config(args.config)
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "benchmark", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    is_server = isinstance(job, LlamaServerJobSpec)
    benchmark_dir = run_dir / "armtune" / ("server_benchmark" if is_server else "benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    target = _target_from_spec(spec)
    job = replace(
        job,
        run_id=manifest.run_id,
        target_type=spec.type,
        remote_module=spec.remote_module,
        target_run_directory=str(benchmark_dir),
    )
    try:
        if isinstance(job, LlamaServerJobSpec):
            result = target.run_server_benchmark(job, benchmark_dir)
        else:
            result = target.run_benchmark(job, benchmark_dir)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = _server_benchmark_exit_code(result) if isinstance(result, LlamaServerBenchmarkResult) else _benchmark_exit_code(result)
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_server_benchmark_summary(result) if isinstance(result, LlamaServerBenchmarkResult) else _benchmark_summary(result))
    return exit_code


def _cmd_optimize(args: argparse.Namespace) -> int:
    spec, job, search_config = load_optimize_config(args.config)
    search_config = _search_config_from_args(search_config, args)
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "optimize", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    is_server = isinstance(job, LlamaServerJobSpec)
    optimize_dir = run_dir / "armtune" / ("server_optimize" if is_server else "optimize")
    optimize_dir.mkdir(parents=True, exist_ok=True)
    target = _target_from_spec(spec)
    job = replace(
        job,
        run_id=manifest.run_id,
        target_type=spec.type,
        remote_module=spec.remote_module,
        target_run_directory=str(optimize_dir),
    )
    try:
        if isinstance(job, LlamaServerJobSpec):
            result = run_server_search(
                target=target,
                base_job=job,
                config=search_config,
                run_id=manifest.run_id,
                optimize_dir=optimize_dir,
            )
        else:
            result = run_search(
                target=target,
                base_job=job,
                config=search_config,
                run_id=manifest.run_id,
                optimize_dir=optimize_dir,
            )
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    if isinstance(result, ArmTuneServerSearchResult):
        exit_code = _server_optimize_exit_code(result)
    else:
        exit_code = 0 if result.status == "completed" else 1
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_server_optimize_summary(result) if isinstance(result, ArmTuneServerSearchResult) else _optimize_summary(result))
    return exit_code


def _cmd_quality_calibrate(args: argparse.Namespace) -> int:
    config = load_quality_calibration_config(args.config)
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "quality-calibrate", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    calibration_dir = run_dir / "armtune" / "quality_calibration"
    target = _target_from_spec(config.target)
    try:
        result = run_quality_calibration(
            target=target,
            config=config,
            run_id=manifest.run_id,
            calibration_dir=calibration_dir,
        )
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = _quality_calibration_exit_code(result)
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_quality_calibration_summary(result))
    return exit_code


def _cmd_build_plan(args: argparse.Namespace) -> int:
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "build-plan", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    output_dir = run_dir / "armtune" / "build_profile"
    try:
        plan = write_build_plan(config_path=args.config, run_id=manifest.run_id, output_dir=output_dir)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    finish_run(run_dir, manifest, "completed", 0)
    print("\n".join([
        f"run_id: {plan.run_id}",
        f"operation: {plan.operation}",
        f"result_origin: {plan.result_origin}",
        f"profile_count: {len(plan.profiles)}",
        f"execute_build: {str(plan.execute_build).lower()}",
        f"artifact_directory: {output_dir}",
    ]))
    return 0


def _cmd_build_profile(args: argparse.Namespace) -> int:
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "build-profile", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    output_dir = run_dir / "armtune" / "build_profile"
    try:
        result = run_build_profile(config_path=args.config, run_id=manifest.run_id, output_dir=output_dir)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = 0 if all(profile.build_status == "completed" for profile in result.profiles) else 1
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_build_profile_summary(result))
    return exit_code


def _cmd_inspect_build(args: argparse.Namespace) -> int:
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "inspect-build", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    output_dir = run_dir / "armtune" / "build_profile"
    try:
        result = inspect_build(config_path=args.config, run_id=manifest.run_id, output_dir=output_dir)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = 0 if all(profile.binary and profile.binary.validation_status == "passed" for profile in result.profiles) else 1
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_build_profile_summary(result))
    return exit_code


def _cmd_profile_compare(args: argparse.Namespace) -> int:
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "profile-compare", "--config", args.config],
        ResourceBudget(enforce=False),
    )
    output_dir = run_dir / "armtune" / "profile_comparison"
    try:
        result = run_profile_comparison(config_path=args.config, run_id=manifest.run_id, output_dir=output_dir)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = 0 if result.recommendation.status in {"recommended", "inconclusive"} else 1
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print(_profile_comparison_summary(result))
    return exit_code


def _cmd_passport(args: argparse.Namespace) -> int:
    run_dir, manifest = create_run(
        ["autotuneai", "arm", "passport", "--comparison-result", args.comparison_result],
        ResourceBudget(enforce=False),
    )
    output_dir = run_dir / "armtune" / "passport"
    try:
        passport, json_path, markdown_path = generate_passport(
            comparison_result_path=args.comparison_result,
            run_id=manifest.run_id,
            output_dir=output_dir,
        )
        verification = verify_passport(json_path)
    except ArmTuneError:
        finish_run(run_dir, manifest, "failed", 2)
        raise
    except Exception as exc:
        finish_run(run_dir, manifest, "failed", 3)
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    exit_code = 0 if verification.verification_status == "passed" else 1
    finish_run(run_dir, manifest, "completed" if exit_code == 0 else "failed", exit_code)
    print("\n".join([
        f"run_id: {passport.run_id}",
        f"claim_eligibility_state: {passport.claim_eligibility_state}",
        f"result_origin: {passport.result_origin}",
        f"verification_status: {verification.verification_status}",
        f"json_path: {json_path}",
        f"markdown_path: {markdown_path}",
    ]))
    return exit_code


def _cmd_verify_passport(args: argparse.Namespace) -> int:
    try:
        verification = verify_passport(args.passport)
    except Exception as exc:
        print(f"ArmTune runtime error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(verification.to_dict(), indent=2, sort_keys=True))
    return 0 if verification.verification_status == "passed" else 1


def _cmd_worker(args: argparse.Namespace) -> int:
    return worker_main(["--job-spec", args.job_spec, "--run-dir", args.run_dir])


def _search_config_from_args(config: SearchConfig, args: argparse.Namespace) -> SearchConfig:
    updates: dict[str, object] = {}
    for attr in [
        "warmup_runs",
        "search_runs",
        "confirmation_runs",
        "cooldown_seconds",
        "random_seed",
        "insert_paired_baseline_every",
        "noise_band_percent",
        "max_candidates",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            updates[attr] = value
    if getattr(args, "affinity_modes", None) is not None:
        updates["affinity_modes"] = _parse_affinity_modes(args.affinity_modes)
    merged = replace(config, **updates) if updates else config
    if merged.warmup_runs < 0:
        raise ArmTuneError("warmup_runs must be >= 0")
    if merged.search_runs < 1:
        raise ArmTuneError("search_runs must be >= 1")
    if merged.confirmation_runs < 1:
        raise ArmTuneError("confirmation_runs must be >= 1")
    if merged.cooldown_seconds < 0:
        raise ArmTuneError("cooldown_seconds must be >= 0")
    if merged.insert_paired_baseline_every < 0:
        raise ArmTuneError("insert_paired_baseline_every must be >= 0")
    if merged.noise_band_percent <= 0:
        raise ArmTuneError("noise_band_percent must be > 0")
    if merged.max_candidates is not None and merged.max_candidates < 1:
        raise ArmTuneError("max_candidates must be >= 1")
    if merged.affinity_modes is not None:
        invalid = [mode for mode in merged.affinity_modes if mode not in AFFINITY_MODE_CHOICES]
        if invalid:
            raise ArmTuneError(f"affinity_modes contains unsupported mode(s): {invalid}")
    return merged


def _parse_affinity_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in AFFINITY_MODE_CHOICES]
    if invalid:
        raise ArmTuneError(f"affinity_modes contains unsupported mode(s): {invalid}")
    return list(dict.fromkeys(modes))


def _target_spec_from_args(args: argparse.Namespace) -> TargetSpec:
    if getattr(args, "target_config", None):
        return load_target_spec(args.target_config)
    return default_local_target()


def _target_from_spec(spec: TargetSpec):
    if spec.type == "local":
        return LocalArmTarget(spec)
    if spec.type == "ssh":
        return SshArmTarget(spec)
    raise ArmTuneError(f"unsupported target type: {spec.type}")


def _hide_subparser_help(subparsers: argparse._SubParsersAction, name: str) -> None:
    subparsers._choices_actions = [action for action in subparsers._choices_actions if action.dest != name]


def _status_from_validation(validation: ProductionValidation) -> StatusCheck:
    return StatusCheck(
        status="ready" if validation.valid else "unsupported",
        reason=validation.reason,
    )


def _configured_status(
    *,
    configured: bool,
    validation: ProductionValidation | None,
    error: str | None,
) -> StatusCheck:
    if not configured:
        return StatusCheck(status="not_configured", reason="no target config was supplied")
    if error:
        return StatusCheck(status="error", reason=error)
    if validation is None:
        return StatusCheck(status="error", reason="configured target validation did not produce a result")
    return _status_from_validation(validation)


def _production_ready(
    *,
    target_configured: bool,
    local_validation: ProductionValidation,
    configured_validation: ProductionValidation | None,
) -> bool:
    if target_configured:
        return bool(configured_validation and configured_validation.valid)
    return local_validation.valid


def _doctor_notes(spec: TargetSpec, statuses: DoctorStatuses) -> list[str]:
    notes = [
        "Host capabilities and workload execution evidence are separate ArmTune schemas.",
        "No benchmark or workload execution evidence is collected by arm doctor.",
        "Controller readiness is separate from production benchmark readiness.",
    ]
    if spec.type == "ssh" and spec.strict_host_key_checking:
        notes.append("SSH strict host-key checking is enabled.")
    if spec.type == "local" and statuses.production_benchmark_ready:
        notes.append("The local machine is a Linux Arm64 production benchmark target.")
    if not statuses.production_benchmark_ready:
        notes.append("Production benchmark execution requires a configured Linux Arm64 target.")
    return notes


def _benchmark_exit_code(result: ArmBenchmarkResult) -> int:
    if result.validation_status != "passed":
        return 2
    if result.execution_status == "completed" and result.parse_status == "parsed" and result.benchmark_status == "passed":
        return 0
    return 1


def _server_benchmark_exit_code(result: LlamaServerBenchmarkResult) -> int:
    if result.cleanup_status == "failed":
        return 4
    if result.validation_status != "passed" or result.inspection_status == "failed":
        return 2
    if (
        result.startup_status == "ready"
        and result.readiness_status == "ready"
        and result.request_status == "completed"
        and result.stream_parse_status == "parsed"
        and result.benchmark_status == "passed"
    ):
        return 0
    return 1


def _benchmark_summary(result: ArmBenchmarkResult) -> str:
    prefill, decode = _summary_metrics(result.trial.parse_result if result.trial else None)
    binary_hash = result.executable.sha256[:12] if result.executable and result.executable.sha256 else "Not available"
    model_hash = result.model.sha256[:12] if result.model and result.model.sha256 else "Not available"
    binary_name = result.executable.basename if result.executable else "Not available"
    model_name = result.model.basename if result.model else "Not available"
    architecture = result.host.architecture.normalized_machine
    lines = [
        f"run_id: {result.run_id}",
        f"target_type: {result.target_type}",
        f"target_architecture: {architecture}",
        f"result_origin: {result.result_origin}",
        f"binary: {binary_name} sha256={binary_hash}",
        f"model: {model_name} sha256={model_hash}",
        f"execution_status: {result.execution_status}",
        f"parse_status: {result.parse_status}",
        f"prefill_tokens_per_second: {_format_metric(prefill)}",
        f"decode_tokens_per_second: {_format_metric(decode)}",
        f"artifact_directory: {result.artifact_directory}",
    ]
    return "\n".join(lines)


def _server_benchmark_summary(result: LlamaServerBenchmarkResult) -> str:
    architecture = result.host.architecture.normalized_machine
    measured = [request for request in result.requests if request.phase == "measured"]
    failed = [
        request
        for request in measured
        if request.request_status != "completed" or request.stream_parse_status != "parsed"
    ]
    ttft = result.distributions.get("ttft_ms")
    e2e = result.distributions.get("end_to_end_latency_ms")
    lines = [
        f"run_id: {result.run_id}",
        f"target_type: {result.target_type}",
        f"target_architecture: {architecture}",
        f"result_origin: {result.result_origin}",
        f"server_startup_status: {result.startup_status}",
        f"server_readiness_status: {result.readiness_status}",
        f"measured_request_count: {len(measured)}",
        f"failed_request_count: {len(failed)}",
        f"median_ttft_ms: {_format_distribution(ttft)}",
        f"median_end_to_end_latency_ms: {_format_distribution(e2e)}",
        f"cleanup_status: {result.cleanup_status}",
        f"quality_gate_status: {result.quality_gate_status}",
        f"artifact_directory: {result.artifact_directory}",
    ]
    return "\n".join(lines)


def _optimize_summary(result) -> str:
    decision = result.recommendation.noise_band_decision
    delta = "Not available" if decision is None or decision.delta_percent is None else f"{decision.delta_percent:.3f}%"
    classification = decision.classification if decision else "not_comparable"
    lines = [
        f"run_id: {result.run_id}",
        f"search_status: {result.status}",
        f"quality_gate_status: {result.quality_gate_status}",
        f"primary_metric: {result.primary_metric}",
        f"selected_candidate: {result.selected_candidate.candidate_id}",
        f"recommendation_status: {result.recommendation.status}",
        f"noise_band_classification: {classification}",
        f"delta_percent: {delta}",
        f"artifact_directory: {result.artifact_directory}",
    ]
    return "\n".join(lines)


def _server_optimize_exit_code(result: ArmTuneServerSearchResult) -> int:
    if result.recommendation.status in {"recommended", "inconclusive"}:
        return 0
    return 1


def _server_optimize_summary(result: ArmTuneServerSearchResult) -> str:
    baseline = _server_candidate_result(result, result.baseline_candidate_id)
    selected = _server_candidate_result(result, result.selected_candidate_id)
    decision = result.recommendation.noise_band_decision
    improvement = "Not available" if decision is None or decision.delta_percent is None else f"{decision.delta_percent:.3f}%"
    noise = "Not available" if decision is None else decision.classification
    selected_quality = selected.quality_result.aggregate if selected else None
    required_passed, required_total = _required_quality_case_counts(selected) if selected else (0, result.quality_suite_manifest.required_case_count)
    lines = [
        f"run_id: {result.run_id}",
        f"target_architecture: {result.target_architecture}",
        f"result_origin: {result.result_origin}",
        f"adapter: {result.runtime_adapter}",
        f"candidate_count: {result.candidate_count}",
        f"eligible_candidate_count: {result.eligible_candidate_count}",
        f"failed_candidate_count: {result.failed_candidate_count}",
        f"quality_suite_id: {result.quality_suite_manifest.suite_id}",
        f"required_quality_cases_passed: {required_passed}/{required_total}",
        f"baseline_median_ttft_ms: {_server_metric(baseline, SERVER_PRIMARY_METRIC)}",
        f"selected_median_ttft_ms: {_server_metric(selected, SERVER_PRIMARY_METRIC)}",
        f"observed_improvement_percent: {improvement}",
        f"selected_median_end_to_end_latency_ms: {_server_metric(selected, SERVER_SECONDARY_METRIC)}",
        f"noise_band_classification: {noise}",
        f"recommendation_state: {result.recommendation.status}",
        f"quality_gate_status: {selected_quality.quality_gate_status if selected_quality else result.quality_gate_status}",
        f"cleanup_status: {_server_cleanup_status(result)}",
        f"artifact_directory: {result.artifact_directory}",
    ]
    return "\n".join(lines)


def _quality_calibration_exit_code(result: QualityCalibrationResult) -> int:
    if result.benchmark_result.cleanup_status == "failed":
        return 4
    if result.suitability_state == "suitable":
        return 0
    if result.suitability_state in {"unsuitable", "inconclusive", "execution_failed"}:
        return 1
    return 3


def _quality_calibration_summary(result: QualityCalibrationResult) -> str:
    return "\n".join(
        [
            f"run_id: {result.run_id}",
            f"target_type: {result.target_type}",
            f"result_origin: {result.result_origin}",
            f"suite_case_count: {result.suite_case_count}",
            f"request_success_count: {result.request_success_count}",
            f"raw_json_valid_count: {result.raw_json_valid_count}",
            f"normalized_json_valid_count: {result.normalized_json_valid_count}",
            f"required_quality_cases_passed: {result.required_quality_cases_passed}/{result.required_quality_cases_total}",
            f"critical_failure_count: {result.critical_failure_count}",
            f"failure_category_histogram: {json.dumps(result.failure_category_histogram, sort_keys=True)}",
            f"quality_gate_status: {result.quality_gate_status}",
            f"model_profile_suitability: {result.suitability_state}",
            f"artifact_directory: {result.artifact_directory}",
        ]
    )


def _build_profile_summary(result) -> str:
    completed = sum(1 for profile in result.profiles if profile.build_status == "completed")
    kleidiai = {profile.profile_id: profile.kleidiai_status for profile in result.profiles}
    return "\n".join(
        [
            f"run_id: {result.run_id}",
            f"operation: {result.operation}",
            f"result_origin: {result.result_origin}",
            f"profile_count: {len(result.profiles)}",
            f"completed_profile_count: {completed}",
            f"kleidiai_status: {json.dumps(kleidiai, sort_keys=True)}",
            f"artifact_directory: {result.artifact_directory}",
        ]
    )


def _profile_comparison_summary(result) -> str:
    return "\n".join(
        [
            f"run_id: {result.run_id}",
            f"operation: {result.operation}",
            f"result_origin: {result.result_origin}",
            f"baseline_profile_id: {result.baseline_profile_id}",
            f"profile_count: {len(result.profiles)}",
            f"eligible_profile_count: {sum(1 for profile in result.profiles if profile.eligible)}",
            f"recommendation_state: {result.recommendation.status}",
            f"recommended_profile_id: {result.recommendation.profile_id or 'Not available'}",
            f"artifact_directory: {result.artifact_directory}",
        ]
    )


def _server_candidate_result(result: ArmTuneServerSearchResult, candidate_id: str | None):
    if candidate_id is None:
        return None
    for candidate in result.candidates:
        if candidate.candidate.candidate_id == candidate_id:
            return candidate
    return None


def _server_metric(candidate, metric_name: str) -> str:
    if candidate is None:
        return "Not available"
    distribution = candidate.distributions.get(metric_name)
    if distribution is None or distribution.median is None:
        return "Not available"
    return f"{distribution.median:.3f}"


def _required_quality_case_counts(candidate) -> tuple[int, int]:
    by_case: dict[str, list[object]] = {}
    for case in candidate.quality_result.cases:
        if case.required:
            by_case.setdefault(case.case_id, []).append(case)
    passed = sum(1 for results in by_case.values() if all(getattr(item, "status", "") == "passed" for item in results))
    return passed, candidate.quality_result.suite_manifest.required_case_count


def _server_cleanup_status(result: ArmTuneServerSearchResult) -> str:
    statuses = {run.cleanup_status for candidate in result.candidates for run in candidate.benchmark_runs}
    if not statuses:
        return "Not available"
    if statuses <= {"passed", "forced"}:
        return "passed"
    if "failed" in statuses:
        return "failed"
    return ",".join(sorted(statuses))


def _summary_metrics(parse_result: LlamaBenchParseResult | None) -> tuple[float | None, float | None]:
    if parse_result is None:
        return None, None
    prefill = None
    decode = None
    for metric in parse_result.metrics:
        if metric.name == "prefill_tokens_per_second":
            prefill = metric.value
        if metric.name == "decode_tokens_per_second":
            decode = metric.value
    return prefill, decode


def _format_metric(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.3f}"


def _format_distribution(distribution: GenerationMetricDistribution | None) -> str:
    if distribution is None or distribution.median is None:
        return "Not available"
    return f"{distribution.median:.3f}"
