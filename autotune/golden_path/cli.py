from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotune.golden_path.cache import gc_cache, invalidate_cache_entry, list_cache_entries, show_cache_entry
from autotune.golden_path.config import DEFAULT_CACHE_ROOT, cache_root, load_golden_path_config
from autotune.golden_path.runner import run_auto, run_best, tune_command


def add_tune_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tune = subparsers.add_parser("tune", help="Tune a command and cache an eligible RecommendationBundle.")
    tune.add_argument("--config", required=True)
    tune.add_argument("--json", action="store_true")
    tune.add_argument("workload", nargs=argparse.REMAINDER)
    tune.set_defaults(handler=_cmd_tune)


def add_cache_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cache = subparsers.add_parser("cache", help="Inspect and maintain the golden-path recommendation cache.")
    cache.add_argument("--root", default=str(DEFAULT_CACHE_ROOT))
    cache_sub = cache.add_subparsers(dest="cache_command")
    list_cmd = cache_sub.add_parser("list", help="List cached recommendations.")
    list_cmd.set_defaults(handler=_cmd_cache_list)
    show_cmd = cache_sub.add_parser("show", help="Show one cached recommendation entry.")
    show_cmd.add_argument("cache_id")
    show_cmd.set_defaults(handler=_cmd_cache_show)
    invalidate_cmd = cache_sub.add_parser("invalidate", help="Mark a cache entry revoked.")
    invalidate_cmd.add_argument("cache_id")
    invalidate_cmd.set_defaults(handler=_cmd_cache_invalidate)
    gc_cmd = cache_sub.add_parser("gc", help="Remove revoked, expired, and corrupt cache entries.")
    gc_cmd.set_defaults(handler=_cmd_cache_gc)


def command_after_separator(command: list[str], usage: str) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit(usage)
    return command


def run_best_from_args(config_path: str, workload: list[str]) -> int:
    command = command_after_separator(workload, "Usage: autotuneai run --best --config <config> -- <command>")
    result = run_best(config_path, command)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return int(result.get("return_code", 1)) if result.get("status") == "completed" else 1


def run_auto_from_args(config_path: str, workload: list[str]) -> int:
    command = command_after_separator(workload, "Usage: autotuneai run --auto --config <config> -- <command>")
    result = run_auto(config_path, command)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return int(result.get("return_code", 1)) if result.get("status") == "completed" else 1


def _cmd_tune(args: argparse.Namespace) -> int:
    command = command_after_separator(args.workload, "Usage: autotuneai tune --config <config> -- <command>")
    result = tune_command(args.config, command)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return 0 if result.get("status") in {"recommended", "inconclusive"} else 1


def _cmd_cache_list(args: argparse.Namespace) -> int:
    entries = list_cache_entries(Path(args.root))
    print(json.dumps({"entries": [_cache_summary(entry) for entry in entries]}, indent=2, sort_keys=True))
    return 0


def _cmd_cache_show(args: argparse.Namespace) -> int:
    print(json.dumps(_cache_summary(show_cache_entry(Path(args.root), args.cache_id), include_paths=True), indent=2, sort_keys=True))
    return 0


def _cmd_cache_invalidate(args: argparse.Namespace) -> int:
    entry = invalidate_cache_entry(Path(args.root), args.cache_id)
    print(json.dumps(_cache_summary(entry, include_paths=True), indent=2, sort_keys=True))
    return 0


def _cmd_cache_gc(args: argparse.Namespace) -> int:
    print(json.dumps(gc_cache(Path(args.root)), indent=2, sort_keys=True))
    return 0


def _summary(result: dict) -> dict:
    keys = [
        "schema_version",
        "status",
        "run_id",
        "selected_candidate_id",
        "recommendation_bundle_id",
        "cache_state",
        "newly_tuned",
        "return_code",
        "artifact_dir",
        "performance_passport_path",
        "recommendation_bundle_path",
        "cache_entry",
        "reason",
    ]
    summary = {key: result.get(key) for key in keys if key in result}
    evidence = result.get("statistical_evidence")
    if isinstance(evidence, dict):
        summary["statistical_evidence"] = {
            "status": evidence.get("status"),
            "classification": evidence.get("classification"),
            "metric": evidence.get("metric"),
            "baseline_median": evidence.get("baseline_median"),
            "selected_median": evidence.get("selected_median"),
            "improvement_percent": evidence.get("improvement_percent"),
        }
    lookup = result.get("cache_lookup")
    if isinstance(lookup, dict):
        summary["cache_lookup"] = {key: value for key, value in lookup.items() if key != "entries"}
    apply_result = result.get("apply_result")
    if isinstance(apply_result, dict):
        summary["apply_result"] = {
            "status": apply_result.get("status"),
            "bundle_id": apply_result.get("bundle_id"),
            "canary_status": apply_result.get("canary", {}).get("status") if isinstance(apply_result.get("canary"), dict) else None,
        }
    return summary


def _cache_summary(entry: dict, *, include_paths: bool = False) -> dict:
    summary = {
        "schema_version": entry.get("schema_version"),
        "cache_id": entry.get("cache_id"),
        "bundle_id": entry.get("bundle_id"),
        "revoked": entry.get("revoked"),
        "expires_at": entry.get("expires_at"),
        "source_run_id": entry.get("source_run_id"),
        "selected_configuration_id": entry.get("selected_configuration_id"),
        "summary": entry.get("summary"),
    }
    if include_paths:
        summary["bundle_path"] = entry.get("bundle_path")
    return summary


def config_cache_root(path: str) -> Path:
    return cache_root(load_golden_path_config(path))

