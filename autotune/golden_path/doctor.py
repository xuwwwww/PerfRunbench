from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from autotune.golden_path import SCHEMA_VERSION
from autotune.golden_path.cache import lookup_cache
from autotune.golden_path.config import (
    GoldenPathConfigError,
    apply_section,
    cache_root,
    candidate_configs,
    load_golden_path_config,
    metrics_section,
    objective_from_config,
    tuning_section,
    workload_section,
)
from autotune.golden_path.identity import (
    build_workload_identity,
    environment_fingerprint_for_command,
    resolve_executable,
    resolved_argv,
)
from autotune.golden_path.metrics import ALLOWED_METRIC_SOURCES
from autotune.recommendation.schemas import file_sha256
from autotune.resource.budget import ResourceBudget
from autotune.training_adapter.config import (
    has_bounded_slice,
    is_training_adapter,
    progress_config,
    search_config,
    slice_config,
    validate_training_config,
    visible_cpu_count,
    workload_cwd,
    world_size,
)
from autotune.training_adapter.candidates import generate_training_candidates
from autotune.training_adapter.errors import TrainingConfigError


@dataclass
class DoctorCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def run_doctor(config_path: str | Path, command: list[str], *, debug: bool = False) -> dict[str, Any]:
    checks: list[DoctorCheck] = []
    config: dict[str, Any] | None = None
    config_file = Path(config_path)
    checks.append(_check_platform())
    try:
        config = load_golden_path_config(config_file)
        checks.append(DoctorCheck("Config", "pass", "Configuration schema is valid.", {"schema_version": config.get("schema_version")}))
    except Exception as exc:
        checks.append(_exception_check("Config", "Configuration schema validation failed.", exc, debug=debug))
        return _payload(checks, config_path=config_file, adapter=None)

    adapter = str(workload_section(config).get("adapter") or "generic-command")
    checks.extend(
        [
            _check_adapter(config),
            _check_working_directory(config),
            _check_workload_command(config, command),
            _check_identity_files(config, config_file),
            _check_metric_contract(config),
            _check_objective(config),
            _check_tuning_limits(config),
            _check_cache_writability(config),
            _check_artifact_writability(),
            _check_apply_support(config),
            _check_resource_guards(config),
        ]
    )
    if is_training_adapter(config):
        checks.extend(_training_checks(config, command))
    else:
        checks.append(DoctorCheck("Bounded training slice", "not_available", "The selected adapter does not use the PyTorch training slice protocol."))
        checks.append(DoctorCheck("PyTorch", "not_available", "The selected adapter does not require PyTorch."))

    checks.append(_check_cache_match(config, config_file, command))
    return _payload(checks, config_path=config_file, adapter=adapter)


def format_doctor_result(result: dict[str, Any]) -> str:
    checks = result.get("checks", [])
    width = max([len(str(check.get("name", ""))) for check in checks] + [8])
    lines = ["PerfRunbench Doctor", ""]
    for check in checks:
        status = _display_status(str(check.get("status")))
        message = str(check.get("message") or "")
        suffix = f" ({message})" if status in {"Warning", "Fail", "Not Available"} and message else ""
        lines.append(f"{str(check.get('name')):<{width}} {status}{suffix}")
    budget = result.get("estimated_tune_budget", {})
    if isinstance(budget, dict) and budget:
        lines.append("")
        lines.append(
            "Estimated tune budget "
            f"{budget.get('unique_candidates')} candidates / "
            f"{budget.get('estimated_process_executions')} process executions"
        )
    lines.append("")
    lines.append(f"Ready: {'yes' if result.get('ready') else 'no'}")
    return "\n".join(lines)


def _payload(checks: list[DoctorCheck], *, config_path: Path, adapter: str | None) -> dict[str, Any]:
    failed = any(check.status == "fail" for check in checks)
    budget = _budget_from_checks(checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "perfrunbench_doctor_result",
        "config_path": str(config_path),
        "adapter": adapter,
        "ready": not failed,
        "checks": [check.to_dict() for check in checks],
        "estimated_tune_budget": budget,
    }


def _check_platform() -> DoctorCheck:
    return DoctorCheck(
        "Host",
        "pass",
        "Controller host was inspected.",
        {
            "system": platform.system(),
            "machine": platform.machine(),
            "logical_cpus": visible_cpu_count(),
            "python": platform.python_version(),
        },
    )


def _check_adapter(config: dict[str, Any]) -> DoctorCheck:
    adapter = str(workload_section(config).get("adapter") or "generic-command")
    if adapter in {"generic-command", "pytorch-training"}:
        return DoctorCheck("Adapter", "pass", f"Adapter {adapter} is supported.", {"adapter": adapter})
    return DoctorCheck("Adapter", "fail", f"Unsupported adapter: {adapter}", {"adapter": adapter})


def _check_working_directory(config: dict[str, Any]) -> DoctorCheck:
    try:
        cwd = workload_cwd(config) if is_training_adapter(config) else _generic_cwd(config)
    except Exception as exc:
        return DoctorCheck("Working directory", "fail", str(exc))
    if not cwd.exists():
        return DoctorCheck("Working directory", "fail", f"Working directory does not exist: {cwd.name}", {"cwd": str(cwd)})
    if not cwd.is_dir():
        return DoctorCheck("Working directory", "fail", f"Working directory is not a directory: {cwd.name}", {"cwd": str(cwd)})
    return DoctorCheck("Working directory", "pass", "Working directory exists.", {"cwd": str(cwd)})


def _check_workload_command(config: dict[str, Any], command: list[str]) -> DoctorCheck:
    if not command:
        return DoctorCheck("Workload command", "fail", "No workload command was provided after --.")
    cwd = workload_cwd(config) if is_training_adapter(config) else _generic_cwd(config)
    executable = resolve_executable(command[0], cwd=cwd)
    if executable is None:
        return DoctorCheck("Workload command", "fail", f"Executable could not be resolved: {command[0]}", {"argv": command})
    try:
        argv = resolved_argv(command, cwd=cwd)
    except Exception as exc:
        return DoctorCheck("Workload command", "fail", str(exc), {"argv": command})
    return DoctorCheck("Workload command", "pass", "Executable and argv resolved.", {"resolved_argv": argv})


def _check_identity_files(config: dict[str, Any], config_path: Path) -> DoctorCheck:
    workload = workload_section(config)
    cwd = workload_cwd(config) if is_training_adapter(config) else _generic_cwd(config)
    records = []
    missing = []
    for key in ("identity_files", "config_files"):
        raw = workload.get(key, [])
        if raw is None:
            continue
        if not isinstance(raw, list):
            return DoctorCheck("Identity files", "fail", f"workload.{key} must be a list")
        for item in raw:
            if not isinstance(item, str):
                return DoctorCheck("Identity files", "fail", f"workload.{key} entries must be strings")
            path = Path(item)
            target = path if path.is_absolute() else cwd / path
            if not target.exists() or not target.is_file():
                missing.append(item)
                records.append({"path": item, "sha256": "missing"})
                continue
            records.append({"path": item, "sha256": file_sha256(target)})
    if config_path.exists():
        records.append({"path": config_path.name, "sha256": file_sha256(config_path)})
    if missing:
        return DoctorCheck("Identity files", "fail", "Declared identity/config file is missing.", {"files": records, "missing": missing})
    if not records:
        return DoctorCheck("Identity files", "warning", "No identity files are declared; cache reuse may be too broad.", {"files": records})
    return DoctorCheck("Identity files", "pass", "Declared identity/config files were hashed.", {"files": records})


def _check_metric_contract(config: dict[str, Any]) -> DoctorCheck:
    try:
        metrics = metrics_section(config)
    except GoldenPathConfigError as exc:
        return DoctorCheck("Metric contract", "fail", str(exc))
    source = str(metrics.get("source") or "json_file")
    if source not in ALLOWED_METRIC_SOURCES:
        return DoctorCheck("Metric contract", "fail", f"Unsupported metrics source: {source}", {"source": source})
    if source == "stdout_json_prefix" and (not isinstance(metrics.get("prefix"), str) or not metrics.get("prefix")):
        return DoctorCheck("Metric contract", "fail", "stdout_json_prefix requires metrics.prefix.", {"source": source})
    if source in {"json_file", "jsonl_file", "adapter_result_file"} and not isinstance(metrics.get("path"), str):
        return DoctorCheck("Metric contract", "fail", "File-based metric sources require metrics.path.", {"source": source})
    return DoctorCheck("Metric contract", "pass", "Metric source is configured.", {"source": source})


def _check_objective(config: dict[str, Any]) -> DoctorCheck:
    try:
        objective = objective_from_config(config)
    except Exception as exc:
        return DoctorCheck("Objective", "fail", str(exc))
    return DoctorCheck("Objective", "pass", "Objective metric is configured.", objective)


def _check_tuning_limits(config: dict[str, Any]) -> DoctorCheck:
    try:
        tuning = tuning_section(config)
        if is_training_adapter(config):
            candidates, stage_statuses = generate_training_candidates(config)
        else:
            candidates = candidate_configs(config)
            stage_statuses = []
    except Exception as exc:
        return DoctorCheck("Estimated tune budget", "fail", str(exc))
    search_runs = int(tuning["search_runs"])
    warmups = int(tuning["warmup_runs"])
    confirmation_runs = int(tuning["confirmation_runs"])
    non_baseline = max(0, len(candidates) - 1)
    estimated = warmups + (len(candidates) * search_runs) + (confirmation_runs * 2 if non_baseline else 0)
    return DoctorCheck(
        "Estimated tune budget",
        "pass",
        f"{non_baseline} candidates / {estimated} process executions.",
        {
            "preset": tuning.get("preset"),
            "unique_candidates": non_baseline,
            "including_baseline": len(candidates),
            "estimated_process_executions": estimated,
            "stage_statuses": stage_statuses,
        },
    )


def _check_cache_writability(config: dict[str, Any]) -> DoctorCheck:
    root = cache_root(config)
    status, message = _writable_location(root)
    return DoctorCheck("Cache directory", status, message, {"path": str(root)})


def _check_artifact_writability() -> DoctorCheck:
    path = Path(".autotuneai") / "runs"
    status, message = _writable_location(path)
    return DoctorCheck("Artifact directory", status, message, {"path": str(path)})


def _check_apply_support(config: dict[str, Any]) -> DoctorCheck:
    apply = apply_section(config)
    canary_runs = max(0, int(apply.get("canary_runs", 1)))
    lease_minutes = max(1, int(apply.get("lease_minutes", 30)))
    return DoctorCheck(
        "Process apply",
        "pass",
        "Process-scoped recommendation apply is available.",
        {"scope": "process", "canary_runs": canary_runs, "lease_minutes": lease_minutes},
    )


def _check_resource_guards(config: dict[str, Any]) -> DoctorCheck:
    budget = ResourceBudget.from_config(config)
    virtual = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    details = {
        "budget": budget.to_record(total_cores=visible_cpu_count(), total_memory_mb=virtual.total / (1024 * 1024)),
        "available_memory_mb": round(virtual.available / (1024 * 1024), 3),
        "free_disk_mb": round(disk.free / (1024 * 1024), 3),
    }
    if budget.memory_budget_gb is not None and budget.memory_budget_gb > 0 and virtual.available < budget.memory_budget_gb * (1024**3):
        return DoctorCheck("Resource guards", "warning", "Configured memory budget exceeds currently available memory.", details)
    return DoctorCheck("Resource guards", "pass", "Memory, disk, and CPU guard configuration was inspected.", details)


def _training_checks(config: dict[str, Any], command: list[str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    try:
        validate_training_config(config)
        checks.append(DoctorCheck("Training config", "pass", "PyTorch training adapter config is valid."))
    except TrainingConfigError as exc:
        checks.append(DoctorCheck("Training config", "fail", str(exc)))
        return checks
    bounded = has_bounded_slice(config)
    checks.append(
        DoctorCheck(
            "Bounded training slice",
            "pass" if bounded else "fail",
            "Bounded tuning slice is configured." if bounded else "PyTorch training tuning requires bounded slice control.",
            {"slice": slice_config(config)},
        )
    )
    checks.append(_check_training_progress(config))
    checks.append(_check_effective_batch(config))
    checks.append(_check_torch(command, config))
    return checks


def _check_training_progress(config: dict[str, Any]) -> DoctorCheck:
    progress = progress_config(config)
    return DoctorCheck("Progress source", "pass", "Training progress source is configured.", progress)


def _check_effective_batch(config: dict[str, Any]) -> DoctorCheck:
    try:
        candidates, _stage_statuses = generate_training_candidates(config)
    except Exception as exc:
        return DoctorCheck("Effective batch", "fail", str(exc))
    baseline = candidates[0] if candidates else {}
    effective = baseline.get("effective_batch_size")
    if effective is None:
        return DoctorCheck("Effective batch", "warning", "Effective batch could not be calculated from declared knobs.", {"world_size": world_size(config)})
    if int(effective) <= 0:
        return DoctorCheck("Effective batch", "fail", "Effective batch must be positive.", {"effective_batch_size": effective})
    changed = [item["id"] for item in candidates if item.get("semantic_change_state") != "preserved" and item.get("recommendation_eligible")]
    if changed:
        return DoctorCheck("Effective batch", "fail", "Semantic-changing candidates cannot be eligible by default.", {"candidates": changed})
    return DoctorCheck("Effective batch", "pass", "Effective batch is preserved for eligible candidates.", {"baseline_effective_batch_size": effective})


def _check_torch(command: list[str], config: dict[str, Any]) -> DoctorCheck:
    if not command:
        return DoctorCheck("PyTorch", "warning", "No command was provided, so torch import could not be checked.")
    cwd = workload_cwd(config)
    executable = resolve_executable(command[0], cwd=cwd)
    if executable is None:
        return DoctorCheck("PyTorch", "warning", "Executable was not resolved, so torch import could not be checked.")
    probe = _probe_python_torch(executable)
    if probe["status"] == "pass":
        device = "CUDA" if probe.get("cuda_available") else "CPU"
        return DoctorCheck("PyTorch", "pass", f"PyTorch import passed ({device}).", probe)
    if probe["status"] == "not_available":
        return DoctorCheck("PyTorch", "not_available", "Command is not a Python interpreter; torch import was not checked.", probe)
    return DoctorCheck("PyTorch", "warning", "PyTorch is not importable in the selected interpreter.", probe)


def _check_cache_match(config: dict[str, Any], config_path: Path, command: list[str]) -> DoctorCheck:
    if not command:
        return DoctorCheck("Cache match", "not_available", "No command was provided for cache lookup.")
    try:
        identity = build_workload_identity(command, config, config_path=config_path)
        fingerprint = environment_fingerprint_for_command(command, config, config_path=config_path)
        lookup = lookup_cache(root=cache_root(config), workload_identity=identity, environment_fingerprint=fingerprint)
    except Exception as exc:
        return DoctorCheck("Cache match", "warning", f"Cache lookup could not be completed: {exc}")
    public_lookup = dict(lookup)
    public_lookup.pop("bundle", None)
    state = str(public_lookup.get("state"))
    if state in {"exact_match", "soft_drift"}:
        status = "pass"
    elif state == "missing":
        status = "not_available"
    else:
        status = "warning"
    return DoctorCheck("Cache match", status, f"Cache lookup state: {state}.", public_lookup)


def _writable_location(path: Path) -> tuple[str, str]:
    if path.exists() and not path.is_dir():
        return "fail", f"Path exists but is not a directory: {path.name}"
    probe_parent = path if path.exists() else _nearest_existing_parent(path)
    if probe_parent is None or not probe_parent.is_dir():
        return "fail", f"No existing parent directory for: {path}"
    try:
        with tempfile.NamedTemporaryFile(prefix=".perfrunbench_doctor_", dir=probe_parent, delete=True):
            pass
    except OSError as exc:
        return "fail", f"Directory is not writable: {exc}"
    return "pass", "Directory is writable or can be created by the current user."


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path.resolve().parent if path.is_absolute() else (Path.cwd() / path).resolve().parent
    while True:
        if current.exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _probe_python_torch(executable: Path) -> dict[str, Any]:
    if "python" not in executable.name.lower() and executable.name.lower() != "py.exe":
        return {"status": "not_available", "executable": str(executable)}
    code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({'status':'pass','torch_version':getattr(torch,'__version__','unknown'),"
        "'cuda_available':bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'status':'warning','error':str(exc)}))\n"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "warning", "executable": str(executable), "error": str(exc)}
    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"status": "warning", "error": (completed.stderr or completed.stdout or "torch probe produced no JSON").strip()}
    payload["executable"] = str(executable)
    payload["return_code"] = completed.returncode
    return payload


def _generic_cwd(config: dict[str, Any]) -> Path:
    workload = workload_section(config)
    cwd = workload.get("cwd")
    return Path(str(cwd)).resolve() if isinstance(cwd, str) and cwd else Path.cwd().resolve()


def _exception_check(name: str, message: str, exc: Exception, *, debug: bool) -> DoctorCheck:
    details = {"error_type": type(exc).__name__}
    if debug:
        details["error"] = repr(exc)
    return DoctorCheck(name, "fail", f"{message}: {exc}", details)


def _display_status(status: str) -> str:
    return {"pass": "Pass", "warning": "Warning", "fail": "Fail", "not_available": "Not Available"}.get(status, status)


def _budget_from_checks(checks: list[DoctorCheck]) -> dict[str, Any]:
    for check in checks:
        if check.name == "Estimated tune budget" and check.status == "pass":
            return {
                "unique_candidates": check.details.get("unique_candidates"),
                "including_baseline": check.details.get("including_baseline"),
                "estimated_process_executions": check.details.get("estimated_process_executions"),
                "preset": check.details.get("preset"),
            }
    return {}
