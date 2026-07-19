from __future__ import annotations

from typing import Any

from autotune.recommendation.guards import GuardResult, guard_dicts, guard_fail, guard_not_available, guard_pass
from autotune.recommendation.schemas import RECOMMENDATION_SCHEMA_VERSION


DEFAULT_THRESHOLDS = {
    "background_cpu_percent_max": 25.0,
    "cpu_steal_percent_max": 1.0,
    "available_memory_mb_min": 512.0,
    "swap_delta_mb_max": 0.0,
    "free_disk_mb_min": 512.0,
    "gpu_occupancy_percent_max": 10.0,
}


OBSERVATION_FIELDS = (
    "load_average_1m",
    "background_cpu_percent",
    "cpu_steal_percent",
    "available_memory_mb",
    "swap_delta_mb",
    "free_disk_mb",
    "target_fingerprint_stable",
    "gpu_occupancy_percent",
    "frequency_governor",
    "thermal_throttling",
)


def classify_trial_contamination(
    observation: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    effective = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    results: list[GuardResult] = []
    contaminated = False

    if "load_average_1m" not in observation:
        results.append(guard_not_available("trial", "load_average_not_available", "load average was not available."))
    else:
        load_average = _number(observation.get("load_average_1m"))
        logical_cpus = _number(observation.get("logical_cpu_count"))
        if load_average is not None and logical_cpus is not None and load_average > logical_cpus:
            contaminated = True
            results.append(guard_fail("trial", "trial_contaminated", "load average exceeded visible logical CPUs.", recoverable=True))
        else:
            results.append(guard_pass("trial", "load_average_ok", "load average did not exceed visible logical CPUs."))

    contaminated |= _threshold_max(observation, results, "background_cpu_percent", effective["background_cpu_percent_max"])
    contaminated |= _threshold_max(observation, results, "cpu_steal_percent", effective["cpu_steal_percent_max"])
    contaminated |= _threshold_min(observation, results, "available_memory_mb", effective["available_memory_mb_min"])
    contaminated |= _threshold_max(observation, results, "swap_delta_mb", effective["swap_delta_mb_max"])
    contaminated |= _threshold_min(observation, results, "free_disk_mb", effective["free_disk_mb_min"])
    contaminated |= _threshold_max(observation, results, "gpu_occupancy_percent", effective["gpu_occupancy_percent_max"])

    if "target_fingerprint_stable" not in observation:
        results.append(guard_not_available("trial", "target_fingerprint_stability_not_available", "target fingerprint stability was not available."))
    elif observation.get("target_fingerprint_stable") is not True:
        contaminated = True
        results.append(guard_fail("trial", "trial_contaminated", "target fingerprint changed during the trial.", recoverable=True))
    else:
        results.append(guard_pass("trial", "target_fingerprint_stable", "target fingerprint remained stable."))

    if "thermal_throttling" not in observation:
        results.append(guard_not_available("trial", "thermal_not_available", "thermal throttling evidence was not available."))
    elif observation.get("thermal_throttling") is True:
        contaminated = True
        results.append(guard_fail("trial", "trial_contaminated", "thermal throttling was observed.", recoverable=True))
    else:
        results.append(guard_pass("trial", "thermal_ok", "thermal throttling was not observed."))

    if "frequency_governor" not in observation:
        results.append(guard_not_available("trial", "frequency_governor_not_available", "frequency/governor evidence was not available."))
    else:
        results.append(guard_pass("trial", "frequency_governor_recorded", "frequency/governor evidence was recorded."))

    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "trial_status": "contaminated" if contaminated else "valid",
        "observed_fields": {field: observation.get(field, "not_available") for field in OBSERVATION_FIELDS},
        "guard_results": guard_dicts(results),
    }


def _threshold_max(observation: dict[str, Any], results: list[GuardResult], field: str, maximum: float) -> bool:
    if field not in observation:
        results.append(guard_not_available("trial", f"{field}_not_available", f"{field} was not available."))
        return False
    value = _number(observation.get(field))
    if value is not None and value > maximum:
        results.append(guard_fail("trial", "trial_contaminated", f"{field}={value} exceeded {maximum}.", recoverable=True))
        return True
    results.append(guard_pass("trial", f"{field}_ok", f"{field} did not exceed {maximum}."))
    return False


def _threshold_min(observation: dict[str, Any], results: list[GuardResult], field: str, minimum: float) -> bool:
    if field not in observation:
        results.append(guard_not_available("trial", f"{field}_not_available", f"{field} was not available."))
        return False
    value = _number(observation.get(field))
    if value is not None and value < minimum:
        results.append(guard_fail("trial", "trial_contaminated", f"{field}={value} was below {minimum}.", recoverable=True))
        return True
    results.append(guard_pass("trial", f"{field}_ok", f"{field} was not below {minimum}."))
    return False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
