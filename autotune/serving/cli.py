from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.serving import SERVING_SCHEMA_VERSION
from autotune.serving.capabilities import discover_server_capabilities
from autotune.serving.network import create_phase21d2_network_plan
from autotune.serving.schemas import (
    ServerCapabilityReport,
    ServerEndpointCapability,
    ServerOptionDefinition,
    ServerRuntimeIdentity,
    ServerRuntimeProfile,
    ServingExperimentPlan,
    ServingExperimentResult,
    ServingMetricSummary,
    ServingTrafficProfile,
)
from autotune.serving.search import compute_pareto_result, create_serving_plan, run_fixture_serving_search, serving_passport_extension
from autotune.serving.workloads import get_standard_workload, standard_workloads


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser(
        "serve",
        help="Plan and run workload-aware llama-server serving autotuning.",
        add_help=False,
    )
    serve.add_argument("--help", action="store_true", dest="serve_help")
    serve.add_argument("serve_args", nargs=argparse.REMAINDER)
    serve.set_defaults(handler=_cmd_serve)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotuneai serve",
        description="Workload-aware serving autotuning preparation for llama-server.",
    )
    subparsers = parser.add_subparsers(dest="serve_command")

    doctor = subparsers.add_parser("doctor", help="Read-only serving autotuning readiness check.")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--debug", action="store_true")
    doctor.set_defaults(handler=_cmd_doctor)

    workload = subparsers.add_parser("workload", help="Inspect built-in serving workload definitions.")
    workload_sub = workload.add_subparsers(dest="workload_command")
    show = workload_sub.add_parser("show", help="Show a workload profile.")
    show.add_argument("workload_id")
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=_cmd_workload_show)

    plan = subparsers.add_parser("plan", help="Create a serving experiment plan without running a server.")
    plan.add_argument("--workload", required=True)
    plan.add_argument("--traffic", choices=["closed-loop", "open-loop", "burst", "shared-prefix"], default="closed-loop")
    plan.add_argument("--concurrency", type=int, default=1)
    plan.add_argument("--candidate-cap", type=int, default=12)
    plan.add_argument("--output", default="results/serving/serving_plan.json")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--debug", action="store_true")
    plan.set_defaults(handler=_cmd_plan)

    run = subparsers.add_parser("run", help="Run a serving plan. Phase 21D0 supports fixture execution only.")
    run.add_argument("--plan", required=True)
    run.add_argument("--fixture", action="store_true", help="Run deterministic fixture serving evidence.")
    run.add_argument("--output-dir", default="results/serving/run")
    run.add_argument("--debug", action="store_true")
    run.set_defaults(handler=_cmd_run)

    search = subparsers.add_parser("search", help="Run staged serving search. Phase 21D0 supports fixture execution only.")
    search.add_argument("--workload", required=True)
    search.add_argument("--traffic", choices=["closed-loop", "open-loop", "burst", "shared-prefix"], default="closed-loop")
    search.add_argument("--concurrency", type=int, default=1)
    search.add_argument("--candidate-cap", type=int, default=12)
    search.add_argument("--fixture", action="store_true")
    search.add_argument("--output-dir", default="results/serving/search")
    search.add_argument("--debug", action="store_true")
    search.set_defaults(handler=_cmd_search)

    compare = subparsers.add_parser("compare", help="Compare two serving result JSON files without merging protocols.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", default="results/serving/serving_compare.json")
    compare.set_defaults(handler=_cmd_compare)

    pareto = subparsers.add_parser("pareto", help="Show a serving Pareto result.")
    pareto.add_argument("--experiment", required=True)
    pareto.set_defaults(handler=_cmd_pareto)

    network = subparsers.add_parser("network-plan", help="Create the future Phase 21D2 remote-network plan only.")
    network.add_argument("--output", default="results/serving/phase21d2_network_plan.json")
    network.set_defaults(handler=_cmd_network_plan)

    cleanup = subparsers.add_parser("cleanup", help="Read-only cleanup guidance for serving experiments.")
    cleanup.set_defaults(handler=_cmd_cleanup)

    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.serve_help:
        build_parser().print_help()
        return 0
    return main(args.serve_args)


def _cmd_doctor(args: argparse.Namespace) -> int:
    capability = discover_server_capabilities(help_text="")
    payload = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "kind": "serving_doctor",
        "phase": "21D0",
        "ready": True,
        "aws_accessed": False,
        "remote_accessed": False,
        "default_commands_read_only": True,
        "standard_workloads": sorted(standard_workloads()),
        "capability_strategy": "exact binary --help, startup logs, endpoint probes, request fields, and build/backend evidence in real phases",
        "local_placeholder_capability_digest": capability.capability_digest,
    }
    _print_payload(payload, as_json=args.json)
    return 0


def _cmd_workload_show(args: argparse.Namespace) -> int:
    workload = get_standard_workload(args.workload_id)
    payload = workload.to_dict()
    _print_payload(payload, as_json=args.json)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = create_serving_plan(
        workload_id=args.workload,
        traffic_mode=args.traffic,
        concurrency=args.concurrency,
        candidate_cap=args.candidate_cap,
    )
    path = atomic_write_json(args.output, plan.to_dict())
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"experiment_id: {plan.experiment_id}")
        print(f"workload_id: {plan.workload.workload_id}")
        print(f"traffic_mode: {plan.traffic.traffic_mode}")
        print(f"candidate_count: {len(plan.candidates)}")
        print(f"result_origin: {plan.result_origin}")
        print(f"plan_path: {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.fixture:
        raise SystemExit("Phase 21D0 serve run requires --fixture; real server execution starts in Phase 21D1")
    plan = _load_plan(args.plan)
    results, pareto = run_fixture_serving_search(plan)
    selected = results[0]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "serving_experiment_result.json", selected.to_dict())
    atomic_write_json(root / "serving_pareto_result.json", pareto.to_dict())
    print(f"experiment_id: {plan.experiment_id}")
    print(f"candidate_id: {selected.candidate_id}")
    print(f"result_origin: {selected.result_origin}")
    print(f"quality_guard_status: {selected.quality_guard_status}")
    print(f"output_dir: {root}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.fixture:
        raise SystemExit("Phase 21D0 serve search requires --fixture; real C7g execution starts in Phase 21D1")
    plan = create_serving_plan(
        workload_id=args.workload,
        traffic_mode=args.traffic,
        concurrency=args.concurrency,
        candidate_cap=args.candidate_cap,
    )
    results, pareto = run_fixture_serving_search(plan)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "serving_plan.json", plan.to_dict())
    for result in results:
        atomic_write_json(root / "candidates" / result.candidate_id / "serving_experiment_result.json", result.to_dict())
    atomic_write_json(root / "serving_pareto_result.json", pareto.to_dict())
    passport_extension = serving_passport_extension(plan=plan, pareto=pareto, result=results[0])
    atomic_write_json(root / "serving_passport_extension.json", passport_extension)
    print(f"experiment_id: {plan.experiment_id}")
    print(f"candidate_count: {len(results)}")
    print(f"pareto_frontier_count: {len(pareto.frontier_points)}")
    print(f"result_origin: {pareto.result_origin}")
    print(f"output_dir: {root}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = _load_result(args.baseline)
    candidate = _load_result(args.candidate)
    if baseline.workload_id != candidate.workload_id or baseline.traffic_profile.traffic_mode != candidate.traffic_profile.traffic_mode:
        raise SystemExit("serving compare refuses mixed workload or traffic protocols")
    payload = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "baseline_candidate_id": baseline.candidate_id,
        "candidate_id": candidate.candidate_id,
        "workload_id": baseline.workload_id,
        "traffic_mode": baseline.traffic_profile.traffic_mode,
        "aggregate_output_tokens_per_second": {
            "baseline": _metric_value(baseline, "aggregate_output_tokens_per_second"),
            "candidate": _metric_value(candidate, "aggregate_output_tokens_per_second"),
        },
        "ttft_p95_ms": {
            "baseline": _metric_p95(baseline, "ttft_ms"),
            "candidate": _metric_p95(candidate, "ttft_ms"),
        },
        "protocols_combined": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_pareto(args: argparse.Namespace) -> int:
    path = Path(args.experiment)
    if path.is_dir():
        path = path / "serving_pareto_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_network_plan(args: argparse.Namespace) -> int:
    plan = create_phase21d2_network_plan()
    path = atomic_write_json(args.output, plan.to_dict())
    print(f"network_plan_path: {path}")
    print("loopback_results_excluded_from_remote_claims: true")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "cleanup_mode": "read_only_phase21d0",
        "real_server_cleanup": "Phase 21D1 requires graceful termination, forced termination if needed, port-release verification, and residual-worker refusal.",
        "aws_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _print_payload(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, sort_keys=True)}")
        else:
            print(f"{key}: {value}")


def _load_plan(path: str | Path) -> ServingExperimentPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    capability = _capability_from_payload(raw["capability_report"])
    workload = get_standard_workload(raw["workload"]["workload_id"])
    traffic = ServingTrafficProfile(**_without_schema(raw["traffic"]))
    candidates = [ServerRuntimeProfile(**_without_schema(item)) for item in raw["candidates"]]
    return ServingExperimentPlan(
        experiment_id=str(raw["experiment_id"]),
        workload=workload,
        traffic=traffic,
        candidates=candidates,
        capability_report=capability,
        result_origin=str(raw.get("result_origin", "planned")),
        candidate_cap=raw.get("candidate_cap"),
        warnings=list(raw.get("warnings", [])),
    )


def _load_result(path: str | Path) -> ServingExperimentResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    traffic = ServingTrafficProfile(**_without_schema(raw["traffic_profile"]))
    runtime = ServerRuntimeProfile(**_without_schema(raw["runtime_profile"]))
    metrics = {key: ServingMetricSummary(**_without_schema(value)) for key, value in raw.get("metrics", {}).items()}
    return ServingExperimentResult(
        experiment_id=str(raw["experiment_id"]),
        candidate_id=str(raw["candidate_id"]),
        workload_id=str(raw["workload_id"]),
        traffic_profile=traffic,
        runtime_profile=runtime,
        result_origin=str(raw["result_origin"]),
        request_count=int(raw["request_count"]),
        failed_request_count=int(raw["failed_request_count"]),
        metrics=metrics,
        quality_guard_status=str(raw.get("quality_guard_status", "not_evaluated")),
        cleanup_status=str(raw.get("cleanup_status", "not_evaluated")),
        client_validity=dict(raw.get("client_validity", {})),
    )


def _capability_from_payload(raw: dict[str, Any]) -> ServerCapabilityReport:
    identity = ServerRuntimeIdentity(**_without_schema(raw["runtime_identity"]))
    options = {
        key: ServerOptionDefinition(**_without_schema(value))
        for key, value in raw.get("options", {}).items()
    }
    endpoints = {
        key: ServerEndpointCapability(**_without_schema(value))
        for key, value in raw.get("endpoints", {}).items()
    }
    return ServerCapabilityReport(
        runtime_identity=identity,
        options=options,
        endpoints=endpoints,
        supported_request_fields=list(raw.get("supported_request_fields", [])),
        supported_flags=list(raw.get("supported_flags", [])),
        capability_digest=str(raw.get("capability_digest", "")),
    )


def _metric_value(result: ServingExperimentResult, metric: str) -> float | None:
    summary = result.metrics.get(metric)
    return summary.p50 if summary else None


def _metric_p95(result: ServingExperimentResult, metric: str) -> float | None:
    summary = result.metrics.get(metric)
    return summary.p95 if summary else None


def _without_schema(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "schema_version"}
