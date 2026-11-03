from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from typing import Any

from autotune.recommendation.guards import GuardResult, guard_dicts, guard_fail, guard_not_available, guard_pass, guard_warn, utc_now
from autotune.recommendation.schemas import RECOMMENDATION_SCHEMA_VERSION, file_sha256, stable_digest


UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
HARD_FIELDS = (
    "architecture",
    "cpu_model",
    "logical_cpu_count",
    "allowed_cpu_mask",
    "executable_sha256",
    "model_sha256",
    "runtime_identity",
    "runtime_version",
    "workload_config_sha256",
    "container_image_digest",
    "required_accelerator_identity",
)
SOFT_FIELDS = (
    "kernel_release",
    "available_memory_mb",
    "hostname",
    "instance_identity",
    "python_version",
    "tool_versions",
)


def collect_environment_fingerprint(
    *,
    executable_path: str | Path | None = None,
    model_path: str | Path | None = None,
    workload_config_path: str | Path | None = None,
    runtime_identity: str | None = None,
    runtime_version: str | None = None,
    container_image_digest: str | None = None,
    required_accelerator_identity: str | None = None,
    allowed_cpu_mask: str | None = None,
    instance_identity: str | None = None,
    tool_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    hard = {
        "architecture": _unknown_if_empty(platform.machine()),
        "cpu_model": _detect_cpu_model(),
        "logical_cpu_count": os.cpu_count() or UNKNOWN,
        "allowed_cpu_mask": _canonical_cpu_mask(allowed_cpu_mask or _current_affinity_mask()),
        "executable_sha256": _digest_or_unknown(executable_path),
        "model_sha256": _digest_or_not_applicable(model_path),
        "runtime_identity": runtime_identity or platform.python_implementation() or UNKNOWN,
        "runtime_version": runtime_version or platform.python_version() or UNKNOWN,
        "workload_config_sha256": _digest_or_not_applicable(workload_config_path),
        "container_image_digest": container_image_digest or NOT_APPLICABLE,
        "required_accelerator_identity": required_accelerator_identity or NOT_APPLICABLE,
    }
    soft = {
        "kernel_release": _unknown_if_empty(platform.release()),
        "available_memory_mb": _available_memory_mb(),
        "hostname": _unknown_if_empty(socket.gethostname()),
        "instance_identity": instance_identity or UNKNOWN,
        "python_version": _unknown_if_empty(platform.python_version()),
        "tool_versions": tool_versions or {},
    }
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "hard": hard,
        "soft": soft,
        "digest": stable_digest({"hard": hard, "soft": soft}),
    }


def compare_environment_fingerprints(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    expected_hard = dict(expected.get("hard", {}))
    observed_hard = dict(observed.get("hard", {}))
    expected_soft = dict(expected.get("soft", {}))
    observed_soft = dict(observed.get("soft", {}))
    guard_results: list[GuardResult] = []
    hard_mismatches = []
    soft_mismatches = []

    for field in HARD_FIELDS:
        expected_value = expected_hard.get(field, UNKNOWN)
        observed_value = observed_hard.get(field, UNKNOWN)
        if field == "allowed_cpu_mask":
            expected_value = _canonical_cpu_mask(expected_value)
            observed_value = _canonical_cpu_mask(observed_value)
        if _is_unknown(expected_value) or _is_unknown(observed_value):
            hard_mismatches.append({"field": field, "expected": expected_value, "observed": observed_value, "reason": "unknown"})
            guard_results.append(
                guard_fail(
                    "preflight",
                    "fingerprint_unknown",
                    f"Hard fingerprint field {field} is unknown and requires retuning before apply.",
                    evidence_refs=[f"environment_fingerprint.hard.{field}"],
                    recoverable=True,
                )
            )
            continue
        if str(expected_value) != str(observed_value):
            hard_mismatches.append({"field": field, "expected": expected_value, "observed": observed_value, "reason": "mismatch"})
            guard_results.append(
                guard_fail(
                    "preflight",
                    "stale_environment",
                    f"Hard fingerprint field {field} changed from {expected_value} to {observed_value}.",
                    evidence_refs=[f"environment_fingerprint.hard.{field}"],
                    recoverable=True,
                )
            )
        else:
            guard_results.append(
                guard_pass(
                    "preflight",
                    "fingerprint_hard_match",
                    f"Hard fingerprint field {field} matches.",
                    evidence_refs=[f"environment_fingerprint.hard.{field}"],
                )
            )

    for field in SOFT_FIELDS:
        expected_value = expected_soft.get(field, UNKNOWN)
        observed_value = observed_soft.get(field, UNKNOWN)
        if _is_unknown(expected_value) or _is_unknown(observed_value):
            guard_results.append(
                guard_not_available(
                    "preflight",
                    "fingerprint_soft_unknown",
                    f"Soft fingerprint field {field} is not available.",
                    evidence_refs=[f"environment_fingerprint.soft.{field}"],
                )
            )
            continue
        if field == "available_memory_mb" and _memory_drift_within_tolerance(expected_value, observed_value):
            guard_results.append(
                guard_pass(
                    "preflight",
                    "fingerprint_memory_within_tolerance",
                    "Available-memory drift is below the 5% / 256 MiB threshold.",
                    evidence_refs=[f"environment_fingerprint.soft.{field}"],
                )
            )
            continue
        if expected_value != observed_value:
            soft_mismatches.append({"field": field, "expected": expected_value, "observed": observed_value})
            guard_results.append(
                guard_warn(
                    "preflight",
                    "fingerprint_soft_mismatch",
                    f"Soft fingerprint field {field} changed.",
                    evidence_refs=[f"environment_fingerprint.soft.{field}"],
                    recoverable=True,
                )
            )
        else:
            guard_results.append(
                guard_pass(
                    "preflight",
                    "fingerprint_soft_match",
                    f"Soft fingerprint field {field} matches.",
                    evidence_refs=[f"environment_fingerprint.soft.{field}"],
                )
            )

    if hard_mismatches:
        status = "hard_mismatch"
        retune_required = True
    elif soft_mismatches:
        status = "soft_mismatch"
        retune_required = False
    else:
        status = "match"
        retune_required = False

    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "status": status,
        "retune_required": retune_required,
        "hard_mismatches": hard_mismatches,
        "soft_mismatches": soft_mismatches,
        "guard_results": guard_dicts(guard_results),
    }


def _detect_cpu_model() -> str:
    if platform.system() == "Linux":
        model = _cpuinfo_value(("model name", "Hardware", "Processor", "CPU implementer"))
        if model:
            return model
    processor = platform.processor()
    return _unknown_if_empty(processor)


def _cpuinfo_value(keys: tuple[str, ...]) -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = [item.strip() for item in line.split(":", 1)]
            if key in keys and value:
                return value
    except OSError:
        return None
    return None


def _current_affinity_mask() -> str:
    if hasattr(os, "sched_getaffinity"):
        try:
            values = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
            return ",".join(str(item) for item in values) if values else UNKNOWN
        except OSError:
            return UNKNOWN
    count = os.cpu_count()
    if not count:
        return UNKNOWN
    return ",".join(str(item) for item in range(count))


def _canonical_cpu_mask(value: Any) -> str:
    if value is None:
        return UNKNOWN
    text = str(value).strip()
    if not text:
        return UNKNOWN
    if text.lower() in {UNKNOWN, "not_available", NOT_APPLICABLE}:
        return text.lower()
    if text.lower().startswith("0x"):
        try:
            mask = int(text, 16)
        except ValueError:
            return text
        if mask <= 0:
            return UNKNOWN
        return ",".join(str(index) for index in range(mask.bit_length()) if mask & (1 << index))
    cpus: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            return text
        if "-" in part:
            left, right = [item.strip() for item in part.split("-", 1)]
            if not left.isdigit() or not right.isdigit():
                return text
            start = int(left)
            end = int(right)
            if start > end:
                return text
            cpus.update(range(start, end + 1))
            continue
        if not part.isdigit():
            return text
        cpus.add(int(part))
    if not cpus:
        return UNKNOWN
    return ",".join(str(item) for item in sorted(cpus))


def _available_memory_mb() -> int | str:
    if platform.system() == "Linux":
        path = Path("/proc/meminfo")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(int(parts[1]) / 1024)
        except (OSError, ValueError, IndexError):
            return UNKNOWN
    return UNKNOWN


def _digest_or_unknown(path: str | Path | None) -> str:
    if path is None:
        return UNKNOWN
    target = Path(path)
    if not target.exists() or not target.is_file():
        return UNKNOWN
    return file_sha256(target)


def _digest_or_not_applicable(path: str | Path | None) -> str:
    if path is None:
        return NOT_APPLICABLE
    return _digest_or_unknown(path)


def _unknown_if_empty(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or UNKNOWN


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", UNKNOWN, "not_available"}:
        return True
    return False


def _memory_drift_within_tolerance(expected: Any, observed: Any) -> bool:
    """Ignore ordinary MemAvailable fluctuations, not large resource shifts."""
    try:
        expected_mb = float(expected)
        observed_mb = float(observed)
    except (TypeError, ValueError):
        return False
    if not (expected_mb >= 0 and observed_mb >= 0):
        return False
    return abs(observed_mb - expected_mb) <= max(256.0, expected_mb * 0.05)
