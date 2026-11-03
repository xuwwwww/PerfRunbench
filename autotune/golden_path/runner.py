from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.golden_path.budget import SearchBudget
from autotune.golden_path.cache import insert_cache_entry, lookup_cache
from autotune.golden_path.config import (
    apply_section,
    budget_from_config,
    cache_policy,
    cache_root,
    candidate_configs,
    load_golden_path_config,
    objective_from_config,
    ttl_days,
    tuning_section,
    workload_section,
)
from autotune.golden_path.identity import (
    build_workload_identity,
    environment_fingerprint_for_command,
    launch_spec_for_command,
    validate_env_overlay,
)
from autotune.golden_path.metrics import MetricSourceError, load_metrics_for_run, objective_value
from autotune.golden_path.policy import should_canary, should_retune
from autotune.recommendation.apply import ApplyOptions, apply_recommendation_bundle, rollback_recommendation_bundle
from autotune.recommendation.bundle import create_recommendation_bundle, write_recommendation_bundle
from autotune.recommendation.schemas import atomic_write_json, file_sha256, stable_digest, validate_launch_spec
from autotune.recommendation.statistics import StatisticalGuardConfig, evaluate_statistical_evidence
from autotune.resource.advanced_tuning import AdvancedRunOptions
from autotune.resource.workload_runner import run_with_budget


def tune_command(config_path: str | Path, command: list[str]) -> dict[str, Any]:
    config = load_golden_path_config(config_path)
    return tune_command_config(config, command, config_path=config_path)


def tune_command_config(
    config: dict[str, Any],
    command: list[str],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if workload_section(config).get("adapter") == "pytorch-training":
        from autotune.training_adapter.runner import tune_training_command_config

        return tune_training_command_config(config, command, config_path=config_path)
    objective = objective_from_config(config)
    tuning = tuning_section(config)
    search_budget = SearchBudget.from_minutes(float(tuning["budget_minutes"]))
    trial_timeout = _trial_timeout_seconds(config)
    candidates = candidate_configs(config)
    if len(candidates) < 2:
        result = _base_result(config_path, command, config)
        result.update(
            {
                "status": "inconclusive",
                "reason": "no non-baseline candidates configured",
                "candidates": candidates,
                "search_budget": search_budget.to_dict(),
            }
        )
        _write_tuning_artifacts(result)
        return result
    result = _base_result(config_path, command, config)
    root = Path(result["artifact_dir"])
    trials: list[dict[str, Any]] = []
    warmups = int(tuning["warmup_runs"])
    for index in range(warmups):
        if not search_budget.can_start(trial_timeout, phase="warmup", candidate_id="baseline"):
            break
        trials.append(
            _run_trial(
                command,
                config,
                candidates[0],
                phase="warmup",
                role="baseline",
                index=index,
                timeout_seconds=search_budget.bounded_timeout(trial_timeout),
            )
        )
        search_budget.cooldown(float(tuning.get("cooldown_seconds", 0.0)))
    search_candidates = candidates[1:]
    random.Random(int(tuning["shuffle_seed"])).shuffle(search_candidates)
    search_order = [candidates[0], *search_candidates]
    for candidate in search_order:
        for index in range(int(tuning["search_runs"])):
            if not search_budget.can_start(trial_timeout, phase="search", candidate_id=str(candidate["id"])):
                break
            trials.append(
                _run_trial(
                    command,
                    config,
                    candidate,
                    phase="search",
                    role=_role(candidate),
                    index=index,
                    timeout_seconds=search_budget.bounded_timeout(trial_timeout),
                )
            )
            search_budget.cooldown(float(tuning.get("cooldown_seconds", 0.0)))
        if search_budget.exhausted:
            break
    selected = _select_candidate(search_order, trials, objective)
    completed_confirmation_pairs = 0
    if selected["id"] != "baseline":
        for index in range(int(tuning["confirmation_runs"])):
            pair_id = f"confirm_{index:03d}"
            pair = [candidates[0], selected]
            if index % 2 == 1:
                pair.reverse()
            pair_budget = (2 * trial_timeout) + float(tuning.get("cooldown_seconds", 0.0))
            if not search_budget.can_start(pair_budget, phase="confirmation", candidate_id=str(selected["id"])):
                break
            for candidate in pair:
                trials.append(
                    _run_trial(
                        command,
                        config,
                        candidate,
                        phase="confirmation",
                        role=_role(candidate),
                        index=index,
                        pair_id=pair_id,
                        timeout_seconds=search_budget.bounded_timeout(trial_timeout),
                    )
                )
                search_budget.cooldown(float(tuning.get("cooldown_seconds", 0.0)))
            completed_confirmation_pairs += 1
    else:
        # Verify that the selected baseline itself runs successfully before
        # reporting baseline retention. Do not present this as a paired gain.
        for index in range(int(tuning["min_valid_trials"])):
            if not search_budget.can_start(trial_timeout, phase="baseline_confirmation", candidate_id="baseline"):
                break
            trials.append(
                _run_trial(
                    command,
                    config,
                    candidates[0],
                    phase="confirmation",
                    role="baseline",
                    index=index,
                    timeout_seconds=search_budget.bounded_timeout(trial_timeout),
                )
            )
            search_budget.cooldown(float(tuning.get("cooldown_seconds", 0.0)))
    statistical_evidence = evaluate_statistical_evidence(
        {"trials": trials},
        StatisticalGuardConfig(
            metric=objective["metric"],
            direction=objective["direction"],
            min_valid_trials=int(tuning["min_valid_trials"]),
            noise_band_percent=float(tuning["noise_band_percent"]),
            variability_threshold_percent=_optional_float(tuning.get("variability_threshold_percent")),
        ),
    )
    evidence_status = statistical_evidence["status"]
    required_confirmation_pairs = int(tuning["confirmation_runs"]) if selected["id"] != "baseline" else 0
    if selected["id"] == "baseline" and not search_budget.exhausted:
        # Keeping the baseline is a valid search outcome; it is not evidence
        # of an improvement and does not require paired confirmation.
        baseline_samples = [
            trial for trial in trials
            if trial.get("candidate_id") == "baseline"
            and trial.get("phase") == "confirmation"
            and trial.get("trial_status") == "valid"
            and isinstance(trial.get("objective_metric_value"), (int, float))
        ]
        if len(baseline_samples) >= int(tuning["min_valid_trials"]):
            evidence_status = "baseline_retained"
            statistical_evidence["status"] = "baseline_retained"
            statistical_evidence["classification"] = "baseline_selected_no_improvement_claim"
            statistical_evidence["baseline_confirmation_samples"] = len(baseline_samples)
            statistical_evidence["note"] = (
                "Baseline ranked first in screening and passed baseline-only "
                "confirmation. No improvement claim or paired comparison was made."
            )
        else:
            evidence_status = "inconclusive"
            statistical_evidence["classification"] = "insufficient_baseline_confirmation_samples"
    if search_budget.exhausted or completed_confirmation_pairs < required_confirmation_pairs:
        evidence_status = "inconclusive"
        statistical_evidence["status"] = "inconclusive"
        statistical_evidence["classification"] = "time_budget_exhausted"
    result.update(
        {
            "status": evidence_status,
            "candidates": candidates,
            "search_order": [candidate["id"] for candidate in search_order],
            "selected_candidate_id": selected["id"],
            "trials": trials,
            "statistical_evidence": statistical_evidence,
            "search_budget": search_budget.to_dict(),
            "confirmation_pairs_completed": completed_confirmation_pairs,
            "confirmation_pairs_required": required_confirmation_pairs,
        }
    )
    passport_path = root / "performance_passport.json"
    _write_passport(result, passport_path)
    result["performance_passport_path"] = str(passport_path)
    if evidence_status == "recommended" and selected["id"] != "baseline":
        bundle_path = root / "recommendation_bundle.json"
        selected_env = dict(selected["env"])
        baseline_config = _configuration_payload("baseline", command, config, candidates[0]["env"], config_path=config_path)
        selected_config = _configuration_payload(selected["id"], command, config, selected_env, config_path=config_path)
        bundle = create_recommendation_bundle(
            source_run_id=result["run_id"],
            source_passport_path=passport_path,
            status="eligible_preliminary",
            objective=objective,
            baseline_configuration=baseline_config,
            selected_configuration=selected_config,
            environment_fingerprint=result["environment_fingerprint"],
            workload_fingerprint=result["workload_identity"],
            constraints=_constraints(config, objective, statistical_evidence),
            statistical_evidence=statistical_evidence,
            apply_scope="process",
            rollback_configuration={"type": "process_exit", "system_restore_run_id": None},
            source_result_path=root / "tuning_result.json",
            provenance={"result_origin": "measured", "protocol": "golden_path_command_v1"},
        )
        write_recommendation_bundle(bundle, bundle_path)
        cache_entry = insert_cache_entry(
            root=cache_root(config),
            bundle_path=bundle_path,
            bundle=bundle,
            workload_identity=result["workload_identity"],
            environment_fingerprint=result["environment_fingerprint"],
            tuning_result=result,
            ttl_days=ttl_days(config),
        )
        result["recommendation_bundle_path"] = str(bundle_path)
        result["recommendation_bundle_id"] = bundle["bundle_id"]
        result["cache_entry"] = {"cache_id": cache_entry["cache_id"], "path": str(cache_root(config) / f"{cache_entry['cache_id']}.json")}
    _write_tuning_artifacts(result)
    return result


def run_best(config_path: str | Path, command: list[str]) -> dict[str, Any]:
    config = load_golden_path_config(config_path)
    lookup = _lookup(config_path, command, config)
    if lookup["state"] not in {"exact_match", "soft_drift"}:
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "reason": "no usable eligible cached recommendation", "cache_lookup": _public_lookup(lookup)}
    return _apply_and_launch(config, command, lookup, newly_tuned=False, force_canary=True)


def run_auto(config_path: str | Path, command: list[str]) -> dict[str, Any]:
    config = load_golden_path_config(config_path)
    return run_auto_config(config, command, config_path=config_path)


def run_auto_config(
    config: dict[str, Any],
    command: list[str],
    *,
    config_path: str | Path | None = None,
    launch_without_recommendation: bool = False,
) -> dict[str, Any]:
    lookup = _lookup(config_path, command, config)
    policy = cache_policy(config)
    if lookup["state"] in {"ambiguous", "revoked"}:
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "reason": f"cache state {lookup['state']} requires explicit intervention", "cache_lookup": _public_lookup(lookup)}
    newly_tuned = False
    if should_retune(str(lookup["state"]), policy["retune_policy"]):
        tuned = tune_command_config(config, command, config_path=config_path)
        if tuned.get("status") != "recommended" or "cache_entry" not in tuned:
            if tuned.get("status") == "baseline_retained":
                # A verified safe fallback, without pretending a better profile exists.
                return _launch_without_recommendation(config, command, tuned)
            if launch_without_recommendation:
                return _launch_without_recommendation(config, command, tuned)
            detail = tuned.get("reason")
            reason = "retune did not produce an eligible recommendation"
            if detail:
                reason = f"{reason}: {detail}"
            return {"schema_version": SCHEMA_VERSION, "status": "failed", "reason": reason, "tuning_result": _public_tuning(tuned)}
        lookup = _lookup(config_path, command, config)
        newly_tuned = True
    if lookup["state"] not in {"exact_match", "soft_drift"}:
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "reason": "cache lookup did not produce a reusable recommendation", "cache_lookup": _public_lookup(lookup)}
    result = _apply_and_launch(config, command, lookup, newly_tuned=newly_tuned, force_canary=False)
    result["auto_status"] = "newly_tuned" if newly_tuned else "cache_hit"
    return result


def _lookup(config_path: str | Path, command: list[str], config: dict[str, Any]) -> dict[str, Any]:
    identity = build_workload_identity(command, config, config_path=config_path)
    fingerprint = environment_fingerprint_for_command(command, config, config_path=config_path)
    return lookup_cache(root=cache_root(config), workload_identity=identity, environment_fingerprint=fingerprint)


def _apply_and_launch(config: dict[str, Any], command: list[str], lookup: dict[str, Any], *, newly_tuned: bool, force_canary: bool) -> dict[str, Any]:
    bundle = lookup["bundle"]
    apply = apply_section(config)
    policy = cache_policy(config)
    canary_needed = force_canary or should_canary(str(lookup["state"]), policy["canary_policy"], newly_tuned=newly_tuned)
    canary_runs = max(1, int(apply.get("canary_runs", 1))) if canary_needed else 0
    apply_result = apply_recommendation_bundle(
        bundle,
        options=ApplyOptions(
            scope="process",
            dry_run=False,
            lease_minutes=max(1, int(apply.get("lease_minutes", 30))),
            takeover=True,
            canary_runs=canary_runs,
            root=cache_root(config) / "apply_state",
        ),
    )
    if apply_result.get("status") != "applied":
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "cache_lookup": _public_lookup(lookup), "apply_result": apply_result}
    launch_spec = bundle.get("selected_configuration", {}).get("launch_spec", {})
    runtime_launch_spec = _runtime_launch_spec(launch_spec)
    selected_env = {str(key): str(value) for key, value in runtime_launch_spec.get("env", {}).items()} if isinstance(runtime_launch_spec.get("env"), dict) else {}
    launch_command = [str(item) for item in runtime_launch_spec.get("argv", command)]
    launch_cwd = runtime_launch_spec.get("cwd")
    try:
        return_code, run_dir = run_with_budget(
            launch_command,
            budget_from_config(config),
            sample_interval_seconds=float(config.get("sample_interval_seconds", 0.5)),
            hard_kill=bool(config.get("hard_kill", False)),
            advanced_options=AdvancedRunOptions(extra_env=selected_env),
            capture_output=True,
            cwd=Path(str(launch_cwd)) if isinstance(launch_cwd, str) and launch_cwd else _cwd(config),
        )
    finally:
        rollback = rollback_recommendation_bundle(bundle, root=cache_root(config) / "apply_state")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "run_dir": str(run_dir),
        "cache_state": lookup["state"],
        "newly_tuned": newly_tuned,
        "apply_result": apply_result,
        "rollback": rollback,
    }


def _run_trial(
    command: list[str],
    config: dict[str, Any],
    candidate: dict[str, Any],
    *,
    phase: str,
    role: str,
    index: int,
    pair_id: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    objective = objective_from_config(config)
    env = {
        **{str(key): str(value) for key, value in candidate["env"].items()},
        "PERFRUNBENCH_CANDIDATE_ID": str(candidate["id"]),
        "PERFRUNBENCH_TRIAL_PHASE": phase,
        "PERFRUNBENCH_TRIAL_ROLE": role,
    }
    if pair_id:
        env["PERFRUNBENCH_PAIR_ID"] = pair_id
    validate_env_overlay(env)
    started_at = time.perf_counter()
    try:
        return_code, run_dir = run_with_budget(
            command,
            budget_from_config(config),
            sample_interval_seconds=float(config.get("sample_interval_seconds", 0.5)),
            hard_kill=bool(config.get("hard_kill", False)),
            advanced_options=AdvancedRunOptions(extra_env=env),
            capture_output=True,
            cwd=_cwd(config),
            timeout_seconds=timeout_seconds,
        )
        elapsed_seconds = time.perf_counter() - started_at
        metrics = load_metrics_for_run(run_dir, config, cwd=_cwd(config), elapsed_seconds=elapsed_seconds)
        metric_value = objective_value(metrics, objective["metric"])
        failed = return_code != 0 or metric_value is None
        error = None if not failed else "command_failed_or_metric_missing"
    except (RuntimeError, MetricSourceError, ValueError) as exc:
        return_code = 1
        run_dir = Path("")
        metrics = {}
        metric_value = None
        failed = True
        error = str(exc)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["id"],
        "role": role,
        "phase": phase,
        "run_index": index,
        "pair_id": pair_id,
        "run_dir": str(run_dir) if str(run_dir) else None,
        "return_code": return_code,
        "metrics": metrics,
        "objective_metric_value": metric_value,
        "trial_status": "failed" if failed else "valid",
        "failed": failed,
        "error": error,
    }


def _base_result(config_path: str | Path, command: list[str], config: dict[str, Any]) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(".autotuneai") / "runs" / run_id / "golden_path"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "golden_path_tuning_result",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_origin": "measured",
        "config_path": str(Path(config_path)) if config_path is not None else None,
        "config_origin": "file" if config_path is not None else "in_memory_default",
        "artifact_dir": str(root),
        "objective": objective_from_config(config),
        "workload_identity": build_workload_identity(command, config, config_path=config_path),
        "environment_fingerprint": environment_fingerprint_for_command(command, config, config_path=config_path),
        "protocol": {
            "schema_version": SCHEMA_VERSION,
            "name": "golden_path_command_v1",
            "warmup_exclusion": True,
            "paired_confirmation": True,
            "process_scoped_apply_only": True,
        },
    }


def _select_candidate(candidates: list[dict[str, Any]], trials: list[dict[str, Any]], objective: dict[str, str]) -> dict[str, Any]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        values = [
            float(trial["objective_metric_value"])
            for trial in trials
            if trial.get("candidate_id") == candidate["id"]
            and trial.get("phase") == "search"
            and trial.get("trial_status") == "valid"
            and isinstance(trial.get("objective_metric_value"), (int, float))
        ]
        if values:
            ranked.append((median(values), candidate))
    if not ranked:
        return candidates[0]
    reverse = objective["direction"] == "maximize"
    ranked.sort(key=lambda item: item[0], reverse=reverse)
    return ranked[0][1]


def _configuration_payload(
    configuration_id: str,
    command: list[str],
    config: dict[str, Any],
    env: dict[str, str],
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "launch_spec": launch_spec_for_command(command, config, env=env, config_path=config_path),
    }


def _constraints(config: dict[str, Any], objective: dict[str, str], evidence: dict[str, Any]) -> dict[str, Any]:
    apply = apply_section(config)
    canary = apply.get("canary", {}) if isinstance(apply.get("canary"), dict) else {}
    metric_source = config.get("metrics", {}).get("source") if isinstance(config.get("metrics"), dict) else None
    if not canary and metric_source != "process_wall_time":
        baseline = evidence.get("baseline_median")
        if isinstance(baseline, (int, float)):
            canary = {"throughput_metric": objective["metric"], "baseline_throughput_floor": baseline}
    return {"schema_version": SCHEMA_VERSION, "canary": canary}


def _write_passport(result: dict[str, Any], path: Path) -> None:
    passport = {
        "schema_version": SCHEMA_VERSION,
        "passport_type": "golden_path_command_performance_passport",
        "run_id": result["run_id"],
        "result_origin": result["result_origin"],
        "claim_eligibility_state": "eligible_preliminary" if result.get("status") == "recommended" else "inconclusive",
        "objective": result["objective"],
        "workload_identity_digest": result["workload_identity"]["digest"],
        "environment_fingerprint_digest": result["environment_fingerprint"]["digest"],
        "statistical_evidence": result.get("statistical_evidence", {}),
        "recommendation_wording": "Best observed eligible configuration under the tested workload and measurement protocol.",
    }
    passport["passport_sha256"] = stable_digest({key: value for key, value in passport.items() if key != "passport_sha256"})
    atomic_write_json(path, passport)


def _write_tuning_artifacts(result: dict[str, Any]) -> None:
    root = Path(result["artifact_dir"])
    atomic_write_json(root / "tuning_result.json", result)
    manifest = _artifact_manifest(root)
    atomic_write_json(root / "artifact_manifest.json", manifest)
    atomic_write_json(root / "artifact_validation.json", _validate_manifest(root, manifest))


def _artifact_manifest(root: Path) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(root.glob("*.json")):
        if path.name in {"artifact_manifest.json", "artifact_validation.json"}:
            continue
        artifacts[path.name] = {
            "relative_path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {"schema_version": SCHEMA_VERSION, "artifacts": artifacts}


def _validate_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for name, record in manifest.get("artifacts", {}).items():
        path = root / str(record.get("relative_path"))
        if not path.exists():
            issues.append({"artifact": name, "issue": "missing"})
        elif file_sha256(path) != record.get("sha256"):
            issues.append({"artifact": name, "issue": "sha256_mismatch"})
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_status": "passed" if not issues else "failed",
        "checked_artifact_count": len(manifest.get("artifacts", {})),
        "issues": issues,
    }


def _public_lookup(lookup: dict[str, Any]) -> dict[str, Any]:
    public = dict(lookup)
    public.pop("bundle", None)
    return public


def _public_tuning(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "selected_candidate_id": result.get("selected_candidate_id"),
        "statistical_evidence": result.get("statistical_evidence"),
        "artifact_dir": result.get("artifact_dir"),
    }


def _role(candidate: dict[str, Any]) -> str:
    return "baseline" if candidate["id"] == "baseline" else "selected"


def _cwd(config: dict[str, Any]) -> Path:
    workload = workload_section(config)
    cwd = workload.get("cwd")
    return Path(str(cwd)).resolve() if isinstance(cwd, str) and cwd else Path.cwd().resolve()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _runtime_launch_spec(launch_spec: dict[str, Any]) -> dict[str, Any]:
    metadata = launch_spec.get("metadata", {}) if isinstance(launch_spec.get("metadata"), dict) else {}
    full_run = metadata.get("full_run_launch_spec")
    if isinstance(full_run, dict):
        validate_launch_spec(full_run)
        return full_run
    validate_launch_spec(launch_spec)
    return launch_spec


def _trial_timeout_seconds(config: dict[str, Any]) -> float:
    value = workload_section(config).get("timeout_seconds", 60.0)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = 60.0
    return max(0.001, timeout)


def _launch_without_recommendation(
    config: dict[str, Any],
    command: list[str],
    tuning_result: dict[str, Any],
) -> dict[str, Any]:
    return_code, run_dir = run_with_budget(
        command,
        budget_from_config(config),
        sample_interval_seconds=float(config.get("sample_interval_seconds", 0.5)),
        hard_kill=bool(config.get("hard_kill", False)),
        capture_output=True,
        cwd=_cwd(config),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if return_code == 0 else "failed",
        "auto_status": "no_eligible_candidate",
        "newly_tuned": False,
        "return_code": return_code,
        "run_dir": str(run_dir),
        "reason": (
            "baseline ranked first; no tuned improvement claimed"
            if tuning_result.get("status") == "baseline_retained"
            else "tuning completed without an eligible recommendation; baseline command executed"
        ),
        "tuning_result": _public_tuning(tuning_result),
    }
