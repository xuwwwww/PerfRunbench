from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotune.recommendation.schemas import atomic_write_json, load_json
from autotune.system_profiles.catalog import collect_capabilities, create_plan, list_profiles, staged_search_policy
from autotune.system_profiles.lease import apply_system_profile_plan, rollback_system_profile, verify_system_profile
from autotune.system_profiles.protocols import list_protocols
from autotune.system_profiles.schemas import HostStateSnapshot, SystemProfileError
from autotune.system_profiles.snapshot import collect_host_state_snapshot


def add_system_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    system = subparsers.add_parser("system", help="Inspect, plan, and fake-apply guarded Linux host system profiles.", add_help=False)
    system.add_argument("--help", action="store_true", dest="system_help")
    system.add_argument("system_args", nargs=argparse.REMAINDER)
    system.set_defaults(handler=_cmd_system_dispatch)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except SystemProfileError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"schema_version": "1.0", "status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"system profile error: {exc}")
        return 1


def _cmd_system_dispatch(args: argparse.Namespace) -> int:
    system_args = list(args.system_args)
    if getattr(args, "system_help", False):
        system_args.insert(0, "--help")
    return main(system_args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotuneai system")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    _root_json(doctor)
    doctor.set_defaults(handler=_cmd_doctor)

    snapshot = subparsers.add_parser("snapshot")
    _root_json(snapshot)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--snapshot-marker", default=None)
    snapshot.set_defaults(handler=_cmd_snapshot)

    plan = subparsers.add_parser("plan")
    _root_json(plan)
    plan.add_argument("--catalog", default="safe-system")
    plan.add_argument("--profile")
    plan.add_argument("--output")
    plan.set_defaults(handler=_cmd_plan)

    show = subparsers.add_parser("show-profile")
    _root_json(show)
    show.add_argument("profile_id")
    show.set_defaults(handler=_cmd_show_profile)

    apply = subparsers.add_parser("apply")
    _root_json(apply)
    apply.add_argument("--profile", required=True)
    apply.add_argument("--lease-minutes", type=int, default=30)
    apply.add_argument("--snapshot", required=True)
    apply.add_argument("--lease-output", required=True)
    apply.add_argument("--allow-real-system", action="store_true")
    apply.set_defaults(handler=_cmd_apply)

    verify = subparsers.add_parser("verify")
    _root_json(verify)
    verify.add_argument("--profile", required=True)
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--lease", required=True)
    verify.add_argument("--allow-real-system", action="store_true")
    verify.set_defaults(handler=_cmd_verify)

    rollback = subparsers.add_parser("rollback")
    _root_json(rollback)
    rollback.add_argument("--snapshot", required=True)
    rollback.add_argument("--lease", required=True)
    rollback.add_argument("--allow-real-system", action="store_true")
    rollback.add_argument("--output")
    rollback.set_defaults(handler=_cmd_rollback)

    boot = subparsers.add_parser("boot-plan")
    _root_json(boot)
    boot.add_argument("--profile", dest="profile_id")
    boot.add_argument("--allow-boot-change", action="store_true")
    boot.add_argument("--user-snapshot-marker")
    boot.add_argument("--output")
    boot.set_defaults(handler=_cmd_boot_plan)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--run-id", required=True)
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(handler=_cmd_compare)
    return parser


def _root_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", help="Fake filesystem root for tests and Phase 21A validation.")
    parser.add_argument("--json", action="store_true")


def _cmd_doctor(args: argparse.Namespace) -> int:
    snapshot = collect_host_state_snapshot(args.root)
    capabilities = collect_capabilities(snapshot, args.root)
    payload = {"schema_version": "1.0", "status": "passed", "snapshot_digest": snapshot.snapshot_digest, "capabilities": capabilities.to_dict(), "protocols": list_protocols()}
    _print(payload, json_output=args.json)
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    snapshot = collect_host_state_snapshot(args.root, snapshot_marker=args.snapshot_marker)
    atomic_write_json(args.output, snapshot.to_dict())
    payload = {"schema_version": "1.0", "status": "passed", "output": args.output, "snapshot_digest": snapshot.snapshot_digest}
    _print(payload, json_output=args.json)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    snapshot = collect_host_state_snapshot(args.root)
    if args.profile:
        payload = create_plan(args.profile, snapshot, args.root, catalog=args.catalog).to_dict()
    else:
        payload = {
            "schema_version": "1.0",
            "catalog": args.catalog,
            "snapshot_digest": snapshot.snapshot_digest,
            "profiles": list_profiles(snapshot, args.root),
            "staged_search_policy": staged_search_policy(snapshot, args.root),
        }
    if args.output:
        atomic_write_json(args.output, payload)
    _print(payload, json_output=True if args.json or args.output is None else False)
    return 0


def _cmd_show_profile(args: argparse.Namespace) -> int:
    snapshot = collect_host_state_snapshot(args.root)
    payload = create_plan(args.profile_id, snapshot, args.root).profile.to_dict()
    _print(payload, json_output=True)
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    snapshot = _load_snapshot(args.snapshot)
    plan = create_plan(args.profile, snapshot, args.root)
    result = apply_system_profile_plan(plan, snapshot, root=args.root, lease_minutes=args.lease_minutes, allow_real_system=args.allow_real_system)
    atomic_write_json(args.lease_output, result.lease.to_dict())
    payload = result.to_dict()
    payload["lease_output"] = args.lease_output
    _print(payload, json_output=True)
    return 0 if result.apply_status == "applied" else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    snapshot = _load_snapshot(args.snapshot)
    plan = create_plan(args.profile, snapshot, args.root)
    lease = _load_lease(args.lease)
    payload = verify_system_profile(plan, lease, root=args.root, allow_real_system=args.allow_real_system)
    _print(payload, json_output=True)
    return 0 if payload["verification_status"] == "passed" else 1


def _cmd_rollback(args: argparse.Namespace) -> int:
    snapshot = _load_snapshot(args.snapshot)
    lease = _load_lease(args.lease)
    report = rollback_system_profile(snapshot, lease, root=args.root, allow_real_system=args.allow_real_system)
    payload = report.to_dict()
    if args.output:
        atomic_write_json(args.output, payload)
    _print(payload, json_output=True)
    return 0 if report.rollback_status == "passed" else 1


def _cmd_boot_plan(args: argparse.Namespace) -> int:
    from autotune.system_profiles.boot import build_boot_profile, list_boot_profiles

    snapshot = collect_host_state_snapshot(args.root)
    if args.profile_id:
        payload = build_boot_profile(
            args.profile_id,
            snapshot,
            root=args.root,
            allow_boot_change=args.allow_boot_change,
            user_snapshot_marker=args.user_snapshot_marker,
        ).to_dict()
    else:
        payload = {"schema_version": "1.0", "profiles": list_boot_profiles(snapshot, root=args.root)}
    if args.output:
        atomic_write_json(args.output, payload)
    _print(payload, json_output=True)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "status": "planning_only",
        "protocols": list_protocols(),
        "score_combination": "prohibited",
        "message": "System profiles are compared per protocol; throughput, latency, startup, and jitter are not combined.",
    }
    _print(payload, json_output=True)
    return 0


def _load_snapshot(path: str | Path) -> HostStateSnapshot:
    raw = load_json(path)
    return HostStateSnapshot(**raw)


def _load_lease(path: str | Path):
    from autotune.system_profiles.schemas import LeaseChangeRecord, SystemTuningLease

    raw = load_json(path)
    changes = [LeaseChangeRecord(**item) for item in raw.get("changes", [])]
    return SystemTuningLease(**{**raw, "changes": changes})


def _print(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status', 'passed')}")
        if "snapshot_digest" in payload:
            print(f"snapshot_digest: {payload['snapshot_digest']}")
