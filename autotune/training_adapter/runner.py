from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.golden_path.budget import SearchBudget
from autotune.golden_path.cache import insert_cache_entry
from autotune.golden_path.config import (
    budget_from_config,
    cache_root,
    load_golden_path_config,
    objective_from_config,
    ttl_days,
    tuning_section,
)
from autotune.golden_path.identity import build_workload_identity, environment_fingerprint_for_command, validate_env_overlay
from autotune.recommendation.bundle import create_recommendation_bundle, write_recommendation_bundle
from autotune.recommendation.schemas import atomic_write_json, file_sha256, stable_digest
from autotune.recommendation.statistics import StatisticalGuardConfig, evaluate_statistical_evidence
from autotune.resource.advanced_tuning import AdvancedRunOptions
from autotune.resource.workload_runner import run_with_budget
from autotune.training_adapter.bindings import CandidatePlan, build_candidate_plan, launch_spec_from_plan
from autotune.training_adapter.candidates import generate_training_candidates
from autotune.training_adapter.config import (
    has_bounded_slice,
    progress_config,
    search_config,
    slice_config,
    validate_training_config,
    workload_cwd,
)
from autotune.training_adapter.errors import TrainingAdapterError, TrainingBindingError, TrainingMetricError
from autotune.training_adapter.metrics import failure_diagnostics_from_exception, load_training_result_for_run, protocol_digest


def tune_training_command(config_path: str | Path, command: list[str]) -> dict[str, Any]:
    config = load_golden_path_config(config_path)
    return tune_training_command_config(config, command, config_path=config_path)


def tune_training_command_config(
    config: dict[str, Any],
    command: list[str],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_training_config(config)
    result = _base_result(config_path, command, config)
    root = Path(result["artifact_dir"])
    if not has_bounded_slice(config):
        result.update({"status": "failed", "reason": "pytorch-training tuning requires bounded slice control"})
        _write_tuning_artifacts(result)
        return result
    try:
        candidates, stage_statuses = generate_training_candidates(config)
    except TrainingAdapterError as exc:
        result.update({"status": "failed", "reason": str(exc)})
        _write_tuning_artifacts(result)
        return result
    plans = _build_plans(command, config, candidates, root)
    result["candidates"] = [plan.to_dict() for plan in plans]
    result["stage_statuses"] = stage_statuses
    if len(plans) < 2:
        result.update({"status": "no_eligible_candidate", "reason": "no non-baseline training candidates generated"})
        _write_tuning_artifacts(result)
        return result
    tuning = tuning_section(config)
    search_budget = SearchBudget.from_minutes(float(tuning["budget_minutes"]))
    trial_timeout = float(slice_config(config).get("max_wall_time_seconds", 300.0))
    trials: list[dict[str, Any]] = []
    baseline = plans[0]
    for index in range(int(tuning["warmup_runs"])):
        if not search_budget.can_start(trial_timeout, phase="warmup", candidate_id="baseline"):
            break
        trials.append(
            _run_trial(
                baseline,
                config,
                phase="warmup",
                role="baseline",
                index=index,
                timeout_seconds=search_budget.bounded_timeout(trial_timeout),
            )
        )
        search_budget.cooldown(float(search_config(config).get("cooldown_seconds", 0.0)))
    search_order = [baseline, *_shuffled(plans[1:], config)]
    for plan in search_order:
        for index in range(int(tuning["search_runs"])):
            if not search_budget.can_start(trial_timeout, phase="search", candidate_id=plan.candidate_id):
                break
            trials.append(
                _run_trial(
                    plan,
                    config,
                    phase="search",
                    role=_role(plan),
                    index=index,
                    timeout_seconds=search_budget.bounded_timeout(trial_timeout),
                )
            )
            search_budget.cooldown(float(search_config(config).get("cooldown_seconds", 0.0)))
        if search_budget.exhausted:
            break
    selected = _select_candidate(search_order, trials, objective_from_config(config))
    completed_confirmation_pairs = 0
    if selected.candidate_id != "baseline":
        for index in range(int(tuning["confirmation_runs"])):
            pair_id = f"confirm_{index:03d}"
            pair = [baseline, selected]
            if index % 2 == 1:
                pair.reverse()
            pair_budget = (2 * trial_timeout) + float(search_config(config).get("cooldown_seconds", 0.0))
            if not search_budget.can_start(pair_budget, phase="confirmation", candidate_id=selected.candidate_id):
                break
            for plan in pair:
                trials.append(
                    _run_trial(
                        plan,
                        config,
                        phase="confirmation",
                        role=_role(plan),
                        index=index,
                        pair_id=pair_id,
                        timeout_seconds=search_budget.bounded_timeout(trial_timeout),
                    )
                )
                search_budget.cooldown(float(search_config(config).get("cooldown_seconds", 0.0)))
            completed_confirmation_pairs += 1
    statistical_evidence = evaluate_statistical_evidence(
        {"trials": trials},
        StatisticalGuardConfig(
            metric=objective_from_config(config)["metric"],
            direction=objective_from_config(config)["direction"],
            min_valid_trials=int(tuning["min_valid_trials"]),
            noise_band_percent=float(tuning["noise_band_percent"]),
            variability_threshold_percent=_optional_float(tuning.get("variability_threshold_percent")),
        ),
    )
    evidence_status = statistical_evidence["status"]
    required_confirmation_pairs = int(tuning["confirmation_runs"]) if selected.candidate_id != "baseline" else 0
    if search_budget.exhausted or completed_confirmation_pairs < required_confirmation_pairs:
        evidence_status = "inconclusive"
        statistical_evidence["status"] = "inconclusive"
        statistical_evidence["classification"] = "time_budget_exhausted"
    result.update(
        {
            "status": evidence_status if selected.recommendation_eligible else "no_eligible_candidate",
            "selected_candidate_id": selected.candidate_id,
            "search_order": [plan.candidate_id for plan in search_order],
            "trials": trials,
            "statistical_evidence": statistical_evidence,
            "selected_candidate_recommendation_eligible": selected.recommendation_eligible,
            "search_budget": search_budget.to_dict(),
            "confirmation_pairs_completed": completed_confirmation_pairs,
            "confirmation_pairs_required": required_confirmation_pairs,
        }
    )
    passport_path = root / "performance_passport.json"
    _write_passport(result, passport_path)
    result["performance_passport_path"] = str(passport_path)
    if result["status"] == "recommended" and selected.candidate_id != "baseline":
        bundle_path = root / "recommendation_bundle.json"
        baseline_config = _configuration_payload(baseline, config, config_path=config_path)
        selected_config = _configuration_payload(selected, config, config_path=config_path)
        bundle = create_recommendation_bundle(
            source_run_id=result["run_id"],
            source_passport_path=passport_path,
            status="eligible_preliminary",
            objective=objective_from_config(config),
            baseline_configuration=baseline_config,
            selected_configuration=selected_config,
            environment_fingerprint=result["environment_fingerprint"],
            workload_fingerprint=result["workload_identity"],
            constraints=_constraints(config, statistical_evidence),
            statistical_evidence=statistical_evidence,
            apply_scope="process",
            rollback_configuration={"type": "process_exit", "system_restore_run_id": None},
            source_result_path=root / "tuning_result.json",
            provenance={"result_origin": "measured", "protocol": "pytorch_training_adapter_v1"},
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


def _build_plans(command: list[str], config: dict[str, Any], candidates: list[dict[str, Any]], root: Path) -> list[CandidatePlan]:
    plans = []
    for candidate in candidates:
        try:
            plans.append(build_candidate_plan(command, config, candidate, plan_root=root / "candidate_plans"))
        except TrainingBindingError as exc:
            plans.append(
                CandidatePlan(
                    candidate_id=str(candidate.get("id") or "invalid_candidate"),
                    stage=str(candidate.get("stage") or "invalid"),
                    knob_values=dict(candidate.get("knob_values", {})),
                    slice_argv=list(command),
                    full_argv=list(command),
                    recommendation_eligible=False,
                    guard_results=[
                        *list(candidate.get("guard_results", [])),
                        {
                            "schema_version": "1.0",
                            "stage": "candidate",
                            "status": "fail",
                            "code": "binding_failed",
                            "message": str(exc),
                            "evidence_refs": [],
                            "recoverable": True,
                        },
                    ],
                )
            )
    return plans


def _run_trial(
    plan: CandidatePlan,
    config: dict[str, Any],
    *,
    phase: str,
    role: str,
    index: int,
    pair_id: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    objective = objective_from_config(config)
    env = {
        **plan.env,
        "PERFRUNBENCH_CANDIDATE_ID": plan.candidate_id,
        "PERFRUNBENCH_TRIAL_PHASE": phase,
        "PERFRUNBENCH_TRIAL_ROLE": role,
    }
    if pair_id:
        env["PERFRUNBENCH_PAIR_ID"] = pair_id
    validate_env_overlay({str(key): str(value) for key, value in env.items()})
    blocking_guards = [guard for guard in plan.guard_results if guard.get("status") == "fail"]
    if blocking_guards:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": plan.candidate_id,
            "role": role,
            "phase": phase,
            "run_index": index,
            "pair_id": pair_id,
            "run_dir": None,
            "return_code": 1,
            "metrics": {},
            "objective_metric_value": None,
            "trial_status": "failed",
            "failed": True,
            "error": "configuration_error",
            "training_diagnostics": {
                "schema_version": "1.0",
                "failure_class": "configuration_error",
                "guard_results": blocking_guards,
            },
            "semantic_change_state": plan.semantic_change_state,
            "recommendation_eligible": plan.recommendation_eligible,
        }
    try:
        return_code, run_dir = run_with_budget(
            plan.slice_argv,
            budget_from_config(config),
            sample_interval_seconds=float(config.get("sample_interval_seconds", 0.5)),
            hard_kill=bool(config.get("hard_kill", False)),
            advanced_options=AdvancedRunOptions(extra_env=env),
            capture_output=True,
            cwd=workload_cwd(config),
            timeout_seconds=timeout_seconds or float(slice_config(config).get("max_wall_time_seconds", 300.0)),
        )
        parsed = load_training_result_for_run(run_dir, config, return_code=return_code)
        metrics = parsed["metrics"]
        diagnostics = parsed["diagnostics"]
        metric_value = _objective_value(metrics, objective["metric"])
        failed = metric_value is None or return_code != 0 or not plan.recommendation_eligible
        error = None if not failed else diagnostics.get("failure_class") or "candidate_not_recommendation_eligible"
    except TrainingMetricError as exc:
        return_code = 1
        run_dir = Path("")
        metrics = {}
        diagnostics = failure_diagnostics_from_exception(exc)
        metric_value = None
        failed = True
        error = diagnostics.get("failure_class") or "invalid_metric"
    except Exception as exc:
        return_code = 1
        run_dir = Path("")
        metrics = {}
        diagnostics = {"schema_version": "1.0", "failure_class": "configuration_error", "semantic_failure_categories": [str(exc)]}
        metric_value = None
        failed = True
        error = str(exc)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": plan.candidate_id,
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
        "training_diagnostics": diagnostics,
        "semantic_change_state": plan.semantic_change_state,
        "recommendation_eligible": plan.recommendation_eligible,
    }


def _base_result(config_path: str | Path | None, command: list[str], config: dict[str, Any]) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(".autotuneai") / "runs" / run_id / "pytorch_training"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pytorch_training_tuning_result",
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
            "name": "pytorch_training_adapter_v1",
            "protocol_digest": protocol_digest(config),
            "warmup_exclusion": True,
            "paired_confirmation": True,
            "bounded_slice_required_for_tuning": True,
            "process_scoped_apply_only": True,
            "progress": progress_config(config),
            "slice": slice_config(config),
        },
    }


def _select_candidate(plans: list[CandidatePlan], trials: list[dict[str, Any]], objective: dict[str, str]) -> CandidatePlan:
    ranked: list[tuple[float, CandidatePlan]] = []
    for plan in plans:
        if not plan.recommendation_eligible:
            continue
        values = [
            float(trial["objective_metric_value"])
            for trial in trials
            if trial.get("candidate_id") == plan.candidate_id
            and trial.get("phase") == "search"
            and trial.get("trial_status") == "valid"
            and isinstance(trial.get("objective_metric_value"), (int, float))
        ]
        if values:
            ranked.append((median(values), plan))
    if not ranked:
        return plans[0]
    ranked.sort(key=lambda item: item[0], reverse=objective["direction"] == "maximize")
    return ranked[0][1]


def _configuration_payload(plan: CandidatePlan, config: dict[str, Any], *, config_path: str | Path | None) -> dict[str, Any]:
    bounded_spec = launch_spec_from_plan(plan, config, full_run=False, config_path=config_path)
    full_spec = launch_spec_from_plan(plan, config, full_run=True, config_path=config_path)
    bounded_spec.setdefault("metadata", {})["full_run_launch_spec"] = full_spec
    bounded_spec["correctness_contract"] = {
        "validators": [
            {"type": "exit_code", "expected": 0},
            {"type": "required_metric", "metric": objective_from_config(config)["metric"]},
            {"type": "metric_finite", "metric": objective_from_config(config)["metric"]},
            {"type": "cleanup_passed"},
        ]
    }
    return {
        "configuration_id": plan.candidate_id,
        "launch_spec": bounded_spec,
        "adapter_metadata": {
            "schema_version": "1.0",
            "adapter": "pytorch-training",
            "knob_values": plan.knob_values,
            "effective_batch_size": plan.effective_batch_size,
            "baseline_effective_batch_size": plan.baseline_effective_batch_size,
            "semantic_change_state": plan.semantic_change_state,
            "slice_protocol_digest": protocol_digest(config),
        },
    }


def _constraints(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    baseline = evidence.get("baseline_median")
    canary = {}
    if isinstance(baseline, (int, float)):
        canary = {"throughput_metric": objective_from_config(config)["metric"], "baseline_throughput_floor": baseline}
    return {
        "schema_version": SCHEMA_VERSION,
        "canary": canary,
        "adapter": "pytorch-training",
        "bounded_slice_required": True,
        "semantic_changes_recommended_by_default": False,
    }


def _write_passport(result: dict[str, Any], path: Path) -> None:
    passport = {
        "schema_version": SCHEMA_VERSION,
        "passport_type": "pytorch_training_performance_passport",
        "run_id": result["run_id"],
        "result_origin": result["result_origin"],
        "claim_eligibility_state": "eligible_preliminary" if result.get("status") == "recommended" else result.get("status", "inconclusive"),
        "objective": result["objective"],
        "workload_identity_digest": result["workload_identity"]["digest"],
        "environment_fingerprint_digest": result["environment_fingerprint"]["digest"],
        "protocol_digest": result["protocol"]["protocol_digest"],
        "statistical_evidence": result.get("statistical_evidence", {}),
        "recommendation_wording": "Best observed eligible configuration under the tested training workload, slice protocol, and measurement protocol.",
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


def _role(plan: CandidatePlan) -> str:
    return "baseline" if plan.candidate_id == "baseline" else "selected"


def _shuffled(plans: list[CandidatePlan], config: dict[str, Any]) -> list[CandidatePlan]:
    copied = list(plans)
    random.Random(int(search_config(config).get("shuffle_seed", 0))).shuffle(copied)
    return copied


def _objective_value(metrics: dict[str, Any], metric: str) -> float | None:
    raw = metrics.get(metric)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
