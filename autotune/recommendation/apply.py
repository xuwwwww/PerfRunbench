from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from autotune.recommendation.bundle import load_recommendation_bundle, validate_recommendation_bundle
from autotune.recommendation.correctness import correctness_passed, evaluate_correctness_contract, load_metrics_from_observation
from autotune.recommendation.fingerprints import collect_environment_fingerprint, compare_environment_fingerprints
from autotune.recommendation.guards import GuardResult, guard_dicts, guard_fail, guard_pass, utc_now
from autotune.recommendation.schemas import RECOMMENDATION_SCHEMA_VERSION, RecommendationError, atomic_write_json, validate_launch_spec
from autotune.system_tuner.runtime import SystemTuningError, apply_system_tuning, restore_system_tuning


LOCKS_DIR = Path(".autotuneai") / "recommendations" / "locks"
LEASES_DIR = Path(".autotuneai") / "recommendations" / "leases"


Runner = Callable[[list[str], dict[str, Any]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ApplyOptions:
    scope: str = "process"
    dry_run: bool = False
    lease_minutes: int = 30
    takeover: bool = False
    canary_runs: int = 1
    root: Path = Path(".autotuneai") / "recommendations"
    runner: Runner | None = None
    system_tuning_sudo: bool = False


def apply_recommendation_bundle(
    bundle_or_path: dict[str, Any] | str | Path,
    *,
    options: ApplyOptions | None = None,
) -> dict[str, Any]:
    options = options or ApplyOptions()
    bundle_path = Path(bundle_or_path) if isinstance(bundle_or_path, (str, Path)) else None
    bundle = load_recommendation_bundle(bundle_path) if bundle_path else dict(bundle_or_path)
    guards: list[GuardResult] = []
    validation = validate_recommendation_bundle(bundle)
    guards.extend(GuardResult.from_dict(item) for item in validation["guard_results"])
    selected = bundle.get("selected_configuration", {})
    launch_spec = selected.get("launch_spec", {}) if isinstance(selected, dict) else {}
    try:
        validate_launch_spec(launch_spec)
    except RecommendationError as exc:
        guards.append(guard_fail("apply", "invalid_launch_spec", str(exc), recoverable=True))
    observed_fingerprint = _observed_fingerprint_for_launch(launch_spec)
    drift = compare_environment_fingerprints(bundle.get("environment_fingerprint", {}), observed_fingerprint)
    guards.extend(GuardResult.from_dict(item) for item in drift["guard_results"])

    result_dir = _result_dir(bundle, bundle_path, options.root)
    result = _base_result(bundle, result_dir, options, observed_fingerprint, drift, guards)
    if options.dry_run:
        result["status"] = "dry_run_passed" if not _has_fail(guards) else "dry_run_failed"
        result["launch_spec"] = _public_launch_spec(launch_spec)
        atomic_write_json(result_dir / "apply_result.json", result)
        return result
    if _has_fail(guards):
        result["status"] = "preflight_failed"
        atomic_write_json(result_dir / "apply_result.json", result)
        return result

    lock = acquire_recommendation_lock(bundle, scope=options.scope, lease_minutes=options.lease_minutes, root=options.root, takeover=options.takeover)
    result["lock"] = lock
    system_restore_run_dir: str | None = None
    try:
        if options.scope == "system":
            system_restore_run_dir = _apply_system_scope(selected, options=options)
            guards.append(guard_pass("apply", "system_scope_applied", "System-scoped allowlisted profile was applied."))
        canary = _run_canary(bundle, launch_spec, options=options, result_dir=result_dir)
        guards.extend(GuardResult.from_dict(item) for item in canary["guard_results"])
        result["canary"] = canary
        if canary["status"] != "passed" or _has_fail(guards):
            rollback = rollback_recommendation_bundle(bundle, root=options.root, system_restore_run_dir=system_restore_run_dir)
            result["rollback"] = rollback
            result["status"] = "rolled_back_after_canary_failure"
            _release_lock(bundle, root=options.root, state="rolled_back")
        else:
            lease = _activate_lease(bundle, scope=options.scope, lease_minutes=options.lease_minutes, root=options.root, system_restore_run_dir=system_restore_run_dir)
            result["lease"] = lease
            result["status"] = "applied"
            _update_lock(bundle, root=options.root, state="active", lease_expires_at=lease["expires_at"])
    except Exception as exc:
        rollback = rollback_recommendation_bundle(bundle, root=options.root, system_restore_run_dir=system_restore_run_dir)
        result["rollback"] = rollback
        result["status"] = "rolled_back_after_apply_failure"
        guards.append(guard_fail("apply", "apply_failed", str(exc), recoverable=True))
        _release_lock(bundle, root=options.root, state="rolled_back")
    result["guard_results"] = guard_dicts(guards)
    atomic_write_json(result_dir / "apply_result.json", result)
    return result


def recommendation_status(
    bundle_or_path: dict[str, Any] | str | Path,
    *,
    root: str | Path = Path(".autotuneai") / "recommendations",
) -> dict[str, Any]:
    bundle = load_recommendation_bundle(bundle_or_path) if isinstance(bundle_or_path, (str, Path)) else dict(bundle_or_path)
    lock_path = _lock_path(bundle, Path(root))
    lease_path = _lease_path(bundle, Path(root))
    lock = _load_json_if_exists(lock_path)
    lease = _load_json_if_exists(lease_path)
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "lock_state": classify_lock(lock)["state"] if lock else "none",
        "lease_state": _lease_state(lease) if lease else "none",
        "lock": lock,
        "lease": lease,
    }


def promote_recommendation_bundle(
    bundle_or_path: dict[str, Any] | str | Path,
    *,
    root: str | Path = Path(".autotuneai") / "recommendations",
) -> dict[str, Any]:
    bundle = load_recommendation_bundle(bundle_or_path) if isinstance(bundle_or_path, (str, Path)) else dict(bundle_or_path)
    lease_path = _lease_path(bundle, Path(root))
    lease = _load_json_if_exists(lease_path)
    if not lease:
        raise RecommendationError("cannot promote without an active lease")
    lease["state"] = "promoted"
    lease["promoted_at"] = utc_now()
    atomic_write_json(lease_path, lease)
    _update_lock(bundle, root=Path(root), state="promoted", lease_expires_at=lease.get("expires_at"))
    return lease


def rollback_recommendation_bundle(
    bundle_or_path: dict[str, Any] | str | Path,
    *,
    root: str | Path = Path(".autotuneai") / "recommendations",
    system_restore_run_dir: str | None = None,
) -> dict[str, Any]:
    bundle = load_recommendation_bundle(bundle_or_path) if isinstance(bundle_or_path, (str, Path)) else dict(bundle_or_path)
    root_path = Path(root)
    lease_path = _lease_path(bundle, root_path)
    lease = _load_json_if_exists(lease_path) or {}
    restore_run_dir = system_restore_run_dir or lease.get("system_restore_run_dir")
    restore_result: list[dict[str, Any]] = []
    if restore_run_dir:
        restore_result = restore_system_tuning(restore_run_dir)
    lease.update(
        {
            "schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "bundle_id": bundle.get("bundle_id"),
            "state": "rolled_back",
            "rolled_back_at": utc_now(),
            "system_restore_run_dir": restore_run_dir,
            "restore_result": restore_result,
        }
    )
    atomic_write_json(lease_path, lease)
    _update_lock(bundle, root=root_path, state="rolled_back", lease_expires_at=lease.get("expires_at"))
    return lease


def acquire_recommendation_lock(
    bundle: dict[str, Any],
    *,
    scope: str,
    lease_minutes: int,
    root: str | Path = Path(".autotuneai") / "recommendations",
    takeover: bool = False,
) -> dict[str, Any]:
    root_path = Path(root)
    path = _lock_path(bundle, root_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_json_if_exists(path)
    if existing:
        state = classify_lock(existing)
        if state["state"] == "active" and not takeover:
            raise RecommendationError(f"recommendation lock is active: {path}")
        if state["state"] == "active" and takeover:
            raise RecommendationError("active lock cannot be overwritten by takeover")
        if not takeover:
            raise RecommendationError(f"recommendation lock exists in {state['state']} state; use explicit takeover")
    lock = {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "source_run_id": bundle.get("source_run_id"),
        "scope": scope,
        "state": "active",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "lease_expires_at": _future_utc(lease_minutes),
    }
    atomic_write_json(path, lock)
    return lock


def classify_lock(lock: dict[str, Any] | None) -> dict[str, Any]:
    if not lock:
        return {"state": "none", "reason": "lock not found"}
    state = str(lock.get("state") or "orphan")
    if state not in {"active"}:
        return {"state": state, "reason": "recorded non-active state"}
    expires_at = _parse_utc(lock.get("lease_expires_at"))
    if expires_at and expires_at < datetime.now(timezone.utc):
        return {"state": "expired", "reason": "lease deadline passed"}
    hostname = lock.get("hostname")
    pid = lock.get("pid")
    if hostname != socket.gethostname():
        return {"state": "orphan", "reason": "lock belongs to another host"}
    if isinstance(pid, int) and not _pid_alive(pid):
        return {"state": "dead", "reason": "owner pid is not alive"}
    return {"state": "active", "reason": "owner appears alive"}


def _run_canary(bundle: dict[str, Any], launch_spec: dict[str, Any], *, options: ApplyOptions, result_dir: Path) -> dict[str, Any]:
    guards: list[GuardResult] = []
    run_records = []
    for run_index in range(max(1, options.canary_runs)):
        observation = _run_launch_spec(launch_spec, result_dir=result_dir, run_index=run_index, runner=options.runner)
        correctness = evaluate_correctness_contract(launch_spec.get("correctness_contract", {}), observation, stage="apply")
        guards.extend(correctness)
        guards.extend(_canary_metric_guards(bundle, observation))
        run_records.append(_public_observation(observation))
    if all(record.get("exit_code") == 0 for record in run_records) and correctness_passed(guards):
        status = "passed"
        guards.append(guard_pass("post_apply", "canary_passed", "Canary passed correctness and metric guards."))
    else:
        status = "failed"
        guards.append(guard_fail("post_apply", "canary_failed", "Canary failed correctness or metric guards.", recoverable=True))
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "status": status,
        "runs": run_records,
        "guard_results": guard_dicts(guards),
    }


def _run_launch_spec(
    launch_spec: dict[str, Any],
    *,
    result_dir: Path,
    run_index: int,
    runner: Runner | None,
) -> dict[str, Any]:
    argv = [str(item) for item in launch_spec["argv"]]
    cwd = launch_spec.get("cwd")
    cwd_path = Path(cwd) if isinstance(cwd, str) and cwd else None
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in launch_spec.get("env", {}).items()})
    env["AUTOTUNEAI_RECOMMENDATION_CANARY_RUN_INDEX"] = str(run_index)
    timeout = float(launch_spec.get("timeout_seconds", 300.0))
    kwargs = {
        "cwd": str(cwd_path) if cwd_path else None,
        "env": env,
        "timeout": timeout,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    completed = runner(argv, kwargs) if runner else subprocess.run(argv, **kwargs)
    stdout_path = result_dir / f"canary_{run_index}_stdout.txt"
    stderr_path = result_dir / f"canary_{run_index}_stderr.txt"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    observation = {
        "exit_code": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "cleanup_passed": True,
        "request_completed": completed.returncode == 0,
        "stream_parsed": True,
    }
    metrics_path = _metrics_path(launch_spec, cwd_path)
    if metrics_path:
        observation["metrics_path"] = str(metrics_path)
        observation["outputs"] = {"metrics": str(metrics_path)}
    metrics = _load_metrics_from_stdout(completed.stdout or "") or load_metrics_from_observation(observation)
    if metrics:
        observation["metrics"] = metrics
    return observation


def _canary_metric_guards(bundle: dict[str, Any], observation: dict[str, Any]) -> list[GuardResult]:
    constraints = bundle.get("constraints", {})
    canary = constraints.get("canary", {}) if isinstance(constraints, dict) else {}
    metrics = load_metrics_from_observation(observation)
    guards: list[GuardResult] = []
    throughput_metric = canary.get("throughput_metric")
    floor = canary.get("baseline_throughput_floor")
    if isinstance(throughput_metric, str) and isinstance(floor, (int, float)):
        value = metrics.get(throughput_metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= floor:
            guards.append(guard_pass("apply", "canary_throughput_floor", f"{throughput_metric} met the canary floor."))
        else:
            guards.append(guard_fail("apply", "canary_throughput_floor", f"{throughput_metric} did not meet the canary floor.", recoverable=True))
    latency_metric = canary.get("latency_metric")
    ceiling = canary.get("baseline_latency_ceiling")
    if isinstance(latency_metric, str) and isinstance(ceiling, (int, float)):
        value = metrics.get(latency_metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= ceiling:
            guards.append(guard_pass("apply", "canary_latency_ceiling", f"{latency_metric} met the canary ceiling."))
        else:
            guards.append(guard_fail("apply", "canary_latency_ceiling", f"{latency_metric} did not meet the canary ceiling.", recoverable=True))
    return guards


def _apply_system_scope(selected_configuration: dict[str, Any], *, options: ApplyOptions) -> str:
    profile = selected_configuration.get("system_profile")
    if not isinstance(profile, str) or not profile:
        raise RecommendationError("system scope requires selected_configuration.system_profile")
    try:
        run_dir, _result = apply_system_tuning(profile, use_sudo=options.system_tuning_sudo)
    except SystemTuningError as exc:
        raise RecommendationError(str(exc)) from exc
    return str(run_dir)


def _observed_fingerprint_for_launch(launch_spec: dict[str, Any]) -> dict[str, Any]:
    executable = Path(str(launch_spec["argv"][0]))
    executable_path = executable if executable.exists() and executable.is_file() else None
    metadata = launch_spec.get("metadata", {}) if isinstance(launch_spec.get("metadata"), dict) else {}
    return collect_environment_fingerprint(
        executable_path=executable_path,
        model_path=metadata.get("model_path"),
        workload_config_path=metadata.get("workload_config_path"),
        runtime_identity=metadata.get("runtime_identity") or launch_spec.get("adapter"),
        runtime_version=metadata.get("runtime_version"),
        container_image_digest=metadata.get("container_image_digest"),
        required_accelerator_identity=metadata.get("required_accelerator_identity"),
        allowed_cpu_mask=metadata.get("allowed_cpu_mask"),
        instance_identity=metadata.get("instance_identity"),
        tool_versions=metadata.get("tool_versions") if isinstance(metadata.get("tool_versions"), dict) else None,
    )


def _activate_lease(
    bundle: dict[str, Any],
    *,
    scope: str,
    lease_minutes: int,
    root: str | Path,
    system_restore_run_dir: str | None,
) -> dict[str, Any]:
    lease = {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "source_run_id": bundle.get("source_run_id"),
        "scope": scope,
        "state": "active",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": utc_now(),
        "expires_at": _future_utc(lease_minutes),
        "process_scoped": scope == "process",
        "system_restore_run_dir": system_restore_run_dir,
    }
    atomic_write_json(_lease_path(bundle, Path(root)), lease)
    return lease


def _release_lock(bundle: dict[str, Any], *, root: str | Path, state: str) -> None:
    _update_lock(bundle, root=root, state=state, lease_expires_at=None)


def _update_lock(bundle: dict[str, Any], *, root: str | Path, state: str, lease_expires_at: str | None) -> None:
    path = _lock_path(bundle, Path(root))
    lock = _load_json_if_exists(path) or {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "source_run_id": bundle.get("source_run_id"),
        "scope": bundle.get("apply_scope"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": utc_now(),
    }
    lock.update({"state": state, "updated_at": utc_now(), "lease_expires_at": lease_expires_at})
    atomic_write_json(path, lock)


def _base_result(
    bundle: dict[str, Any],
    result_dir: Path,
    options: ApplyOptions,
    observed_fingerprint: dict[str, Any],
    drift: dict[str, Any],
    guards: list[GuardResult],
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "source_run_id": bundle.get("source_run_id"),
        "scope": options.scope,
        "dry_run": options.dry_run,
        "created_at": utc_now(),
        "observed_environment_fingerprint": observed_fingerprint,
        "environment_drift": drift,
        "guard_results": guard_dicts(guards),
    }


def _result_dir(bundle: dict[str, Any], bundle_path: Path | None, root: Path) -> Path:
    if bundle_path is not None:
        return bundle_path.parent / "apply"
    bundle_id = str(bundle.get("bundle_id") or "unknown")
    return root / str(bundle.get("source_run_id") or bundle_id) / "apply"


def _lock_path(bundle: dict[str, Any], root: Path) -> Path:
    return root / "locks" / f"{bundle.get('bundle_id')}.lock.json"


def _lease_path(bundle: dict[str, Any], root: Path) -> Path:
    return root / "leases" / f"{bundle.get('bundle_id')}.lease.json"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": "orphan", "error": "invalid JSON"}
    return raw if isinstance(raw, dict) else {"state": "orphan", "error": "non-object JSON"}


def _lease_state(lease: dict[str, Any]) -> str:
    if lease.get("state") != "active":
        return str(lease.get("state") or "unknown")
    expires_at = _parse_utc(lease.get("expires_at"))
    if expires_at and expires_at < datetime.now(timezone.utc):
        return "expired"
    return "active"


def _future_utc(minutes: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=max(1, minutes))).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _public_launch_spec(launch_spec: dict[str, Any]) -> dict[str, Any]:
    public = dict(launch_spec)
    if "env" in public:
        public["env"] = {key: "<redacted>" for key in public.get("env", {})}
    return public


def _public_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": observation.get("exit_code"),
        "stdout_path": observation.get("stdout_path"),
        "stderr_path": observation.get("stderr_path"),
        "metrics": observation.get("metrics", {}),
        "cleanup_passed": observation.get("cleanup_passed"),
        "request_completed": observation.get("request_completed"),
        "stream_parsed": observation.get("stream_parsed"),
    }


def _metrics_path(launch_spec: dict[str, Any], cwd_path: Path | None) -> Path | None:
    metadata = launch_spec.get("metadata", {}) if isinstance(launch_spec.get("metadata"), dict) else {}
    path_value = metadata.get("metrics_output_path")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute() and cwd_path is not None:
        path = cwd_path / path
    return path


def _load_metrics_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text.startswith("{"):
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return raw


def _has_fail(guards: list[GuardResult]) -> bool:
    return any(guard.status == "fail" for guard in guards)
