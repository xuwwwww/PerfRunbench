from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotune.recommendation.apply import (
    ApplyOptions,
    apply_recommendation_bundle,
    promote_recommendation_bundle,
    recommendation_status,
    rollback_recommendation_bundle,
)
from autotune.recommendation.bundle import (
    generate_bundle_from_source,
    load_recommendation_bundle,
    recommendation_path,
    validate_recommendation_bundle,
    write_recommendation_bundle,
)
from autotune.recommendation.schemas import RecommendationError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except RecommendationError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotuneai recommendation", description="Validate and apply guarded PerfRunbench recommendations.")
    subparsers = parser.add_subparsers(dest="recommendation_command")

    generate = subparsers.add_parser("generate", help="Generate a RecommendationBundle from an eligible source result.")
    generate.add_argument("--passport", required=True)
    generate.add_argument("--source-result")
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--selected-config", required=True)
    generate.add_argument("--baseline-config")
    generate.add_argument("--objective-metric", default="throughput")
    generate.add_argument("--objective-direction", choices=["maximize", "minimize"], default="maximize")
    generate.add_argument("--scope", choices=["process", "system"], default="process")
    generate.add_argument("--output")
    generate.set_defaults(handler=_cmd_generate)

    show = subparsers.add_parser("show", help="Show a RecommendationBundle summary.")
    _add_bundle_args(show)
    show.set_defaults(handler=_cmd_show)

    validate = subparsers.add_parser("validate", help="Validate a RecommendationBundle and source Passport digest.")
    _add_bundle_args(validate)
    validate.set_defaults(handler=_cmd_validate)

    apply = subparsers.add_parser("apply", help="Apply a recommendation through dry-run or guarded canary.")
    _add_bundle_args(apply)
    apply.add_argument("--scope", choices=["process", "system"], default="process")
    apply.add_argument("--lease-minutes", type=int, default=30)
    apply.add_argument("--dry-run", action="store_true")
    apply.add_argument("--takeover", action="store_true")
    apply.add_argument("--canary-runs", type=int, default=1)
    apply.add_argument("--system-tuning-sudo", action="store_true")
    apply.set_defaults(handler=_cmd_apply)

    status = subparsers.add_parser("status", help="Show recommendation lock and lease state.")
    _add_bundle_args(status)
    status.set_defaults(handler=_cmd_status)

    promote = subparsers.add_parser("promote", help="Promote an active recommendation lease.")
    _add_bundle_args(promote)
    promote.set_defaults(handler=_cmd_promote)

    rollback = subparsers.add_parser("rollback", help="Rollback a recommendation lease.")
    _add_bundle_args(rollback)
    rollback.set_defaults(handler=_cmd_rollback)
    return parser


def _add_bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--bundle")
    parser.add_argument("--root", default=str(Path(".autotuneai") / "recommendations"))


def _cmd_generate(args: argparse.Namespace) -> int:
    output_path = Path(args.output) if args.output else recommendation_path(args.run_id, root=args.root)
    bundle = generate_bundle_from_source(
        passport_path=args.passport,
        source_result_path=args.source_result,
        source_run_id=args.run_id,
        selected_configuration_path=args.selected_config,
        baseline_configuration_path=args.baseline_config,
        objective_metric=args.objective_metric,
        objective_direction=args.objective_direction,
        apply_scope=args.scope,
        output_path=output_path,
    )
    print(json.dumps(_summary(bundle), indent=2, sort_keys=True))
    print(f"Wrote recommendation bundle to {output_path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    bundle = _load_from_args(args)
    print(json.dumps(_summary(bundle), indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    bundle = _load_from_args(args)
    result = validate_recommendation_bundle(bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def _cmd_apply(args: argparse.Namespace) -> int:
    bundle_path = _bundle_path_from_args(args)
    result = apply_recommendation_bundle(
        bundle_path,
        options=ApplyOptions(
            scope=args.scope,
            dry_run=args.dry_run,
            lease_minutes=args.lease_minutes,
            takeover=args.takeover,
            canary_runs=args.canary_runs,
            root=Path(args.root),
            system_tuning_sudo=args.system_tuning_sudo,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"dry_run_passed", "applied"} else 1


def _cmd_status(args: argparse.Namespace) -> int:
    result = recommendation_status(_bundle_path_from_args(args), root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    result = promote_recommendation_bundle(_bundle_path_from_args(args), root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    result = rollback_recommendation_bundle(_bundle_path_from_args(args), root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_from_args(args: argparse.Namespace) -> dict:
    return load_recommendation_bundle(_bundle_path_from_args(args))


def _bundle_path_from_args(args: argparse.Namespace) -> Path:
    if args.bundle:
        return Path(args.bundle)
    if not args.run_id:
        raise RecommendationError("--run-id or --bundle is required")
    return recommendation_path(args.run_id, root=args.root)


def _summary(bundle: dict) -> dict:
    selected = bundle.get("selected_configuration", {})
    launch_spec = selected.get("launch_spec", {}) if isinstance(selected, dict) else {}
    env_keys = sorted((launch_spec.get("env") or {}).keys()) if isinstance(launch_spec.get("env"), dict) else []
    return {
        "schema_version": bundle.get("schema_version"),
        "bundle_id": bundle.get("bundle_id"),
        "source_run_id": bundle.get("source_run_id"),
        "source_passport_path": bundle.get("source_passport_path"),
        "source_passport_sha256": bundle.get("source_passport_sha256"),
        "status": bundle.get("status"),
        "objective": bundle.get("objective"),
        "apply_scope": bundle.get("apply_scope"),
        "selected_configuration_id": selected.get("configuration_id") if isinstance(selected, dict) else None,
        "adapter": launch_spec.get("adapter"),
        "argv": launch_spec.get("argv"),
        "env_keys": env_keys,
        "expires_at": bundle.get("expires_at"),
        "recommendation_wording": bundle.get("recommendation_wording"),
    }
