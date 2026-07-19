from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Any

from autotune.recommendation.guards import GuardResult, guard_dicts, guard_fail, guard_pass, guard_warn
from autotune.recommendation.schemas import RECOMMENDATION_SCHEMA_VERSION


@dataclass(frozen=True)
class StatisticalGuardConfig:
    metric: str
    direction: str = "maximize"
    min_valid_trials: int = 3
    noise_band_percent: float = 2.0
    variability_threshold_percent: float | None = None


def evaluate_statistical_evidence(evidence: dict[str, Any], config: StatisticalGuardConfig) -> dict[str, Any]:
    if config.direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    trials = evidence.get("trials", [])
    if not isinstance(trials, list):
        trials = []
    baseline_confirmation = _valid_trials(trials, role="baseline", phase="confirmation", metric=config.metric)
    selected_confirmation = _valid_trials(trials, role="selected", phase="confirmation", metric=config.metric)
    baseline_search = _valid_trials(trials, role="baseline", phase="search", metric=config.metric)
    selected_search = _valid_trials(trials, role="selected", phase="search", metric=config.metric)
    failed_trials = [trial for trial in trials if trial.get("failed") is True or trial.get("trial_status") == "failed"]
    contaminated_trials = [trial for trial in trials if trial.get("trial_status") == "contaminated"]
    warmup_trials = [trial for trial in trials if trial.get("phase") == "warmup"]

    guards: list[GuardResult] = []
    if failed_trials:
        guards.append(
            guard_warn(
                "trial",
                "failed_trials_preserved",
                f"{len(failed_trials)} failed trial(s) are preserved in statistical evidence.",
                recoverable=True,
            )
        )
    else:
        guards.append(guard_pass("trial", "failed_trials_preserved", "No failed trials were recorded."))

    if warmup_trials:
        guards.append(guard_pass("trial", "warmup_excluded", f"{len(warmup_trials)} warm-up trial(s) were excluded."))
    else:
        guards.append(guard_pass("trial", "warmup_excluded", "No warm-up trials were present."))

    if contaminated_trials:
        guards.append(
            guard_warn(
                "trial",
                "contaminated_trials_excluded",
                f"{len(contaminated_trials)} contaminated trial(s) were excluded from valid samples.",
                recoverable=True,
            )
        )
    else:
        guards.append(guard_pass("trial", "contaminated_trials_excluded", "No contaminated trials were recorded."))

    if len(baseline_confirmation) < config.min_valid_trials:
        guards.append(
            guard_fail(
                "recommendation",
                "insufficient_confirmation_trials",
                f"Baseline confirmation has {len(baseline_confirmation)} valid sample(s), below {config.min_valid_trials}.",
                recoverable=True,
            )
        )
    else:
        guards.append(guard_pass("recommendation", "min_valid_trials", "Baseline confirmation has enough valid samples."))

    if len(selected_confirmation) < config.min_valid_trials:
        guards.append(
            guard_fail(
                "recommendation",
                "insufficient_confirmation_trials",
                f"Selected confirmation has {len(selected_confirmation)} valid sample(s), below {config.min_valid_trials}.",
                recoverable=True,
            )
        )
    else:
        guards.append(guard_pass("recommendation", "min_valid_trials", "Selected confirmation has enough valid samples."))

    if not baseline_confirmation or not selected_confirmation:
        guards.append(
            guard_fail(
                "recommendation",
                "search_only_recommendation_rejected",
                "Fresh confirmation samples are required; search samples alone are not recommendation evidence.",
                recoverable=True,
            )
        )

    paired = _paired_confirmation_pairs(baseline_confirmation, selected_confirmation, config.metric)
    if paired:
        guards.append(guard_pass("recommendation", "paired_baseline_evidence", f"{len(paired)} paired confirmation sample(s) were found."))
    else:
        guards.append(
            guard_fail(
                "recommendation",
                "missing_paired_baseline_evidence",
                "No paired confirmation baseline evidence was found.",
                recoverable=True,
            )
        )

    baseline_values = [_metric_value(trial, config.metric) for trial in baseline_confirmation]
    selected_values = [_metric_value(trial, config.metric) for trial in selected_confirmation]
    baseline_values = [value for value in baseline_values if value is not None]
    selected_values = [value for value in selected_values if value is not None]
    baseline_median = median(baseline_values) if baseline_values else None
    selected_median = median(selected_values) if selected_values else None
    improvement_percent = _improvement_percent(selected_median, baseline_median, config.direction)
    classification = "inconclusive"
    status = "inconclusive"

    if improvement_percent is None:
        guards.append(
            guard_fail(
                "recommendation",
                "missing_improvement_evidence",
                "Median improvement could not be computed.",
                recoverable=True,
            )
        )
    elif improvement_percent <= config.noise_band_percent:
        guards.append(
            guard_fail(
                "recommendation",
                "improvement_within_noise_band",
                f"Median improvement {improvement_percent:.6f}% is not outside the {config.noise_band_percent:.6f}% noise band.",
                recoverable=True,
            )
        )
    else:
        guards.append(
            guard_pass(
                "recommendation",
                "improvement_outside_noise_band",
                f"Median improvement {improvement_percent:.6f}% is outside the {config.noise_band_percent:.6f}% noise band.",
            )
        )

    consistency = _confirmation_direction_consistent(paired, direction=config.direction)
    if paired and consistency:
        guards.append(guard_pass("recommendation", "consistent_confirmation_direction", "Paired confirmation direction was consistent."))
    elif paired:
        guards.append(
            guard_fail(
                "recommendation",
                "inconsistent_confirmation_direction",
                "Paired confirmation samples did not consistently favor the selected configuration.",
                recoverable=True,
            )
        )

    variability = _variability_percent(selected_values)
    if config.variability_threshold_percent is not None:
        if variability is None:
            guards.append(guard_fail("recommendation", "variability_unavailable", "Selected variability could not be computed.", recoverable=True))
        elif variability > config.variability_threshold_percent:
            guards.append(
                guard_fail(
                    "recommendation",
                    "variability_threshold_exceeded",
                    f"Selected variability {variability:.6f}% exceeded {config.variability_threshold_percent:.6f}%.",
                    recoverable=True,
                )
            )
        else:
            guards.append(guard_pass("recommendation", "variability_threshold", "Selected variability is within the configured threshold."))

    if not any(result.status == "fail" for result in guards) and improvement_percent is not None:
        classification = "improved"
        status = "recommended"
    elif improvement_percent is not None and improvement_percent <= config.noise_band_percent:
        classification = "within_noise_band"
        status = "inconclusive"
    elif improvement_percent is not None and improvement_percent < 0:
        classification = "regressed"
        status = "inconclusive"

    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "metric": config.metric,
        "direction": config.direction,
        "baseline_median": baseline_median,
        "selected_median": selected_median,
        "improvement_percent": improvement_percent,
        "noise_band_percent": config.noise_band_percent,
        "selected_variability_percent": variability,
        "valid_confirmation_baseline_trials": len(baseline_confirmation),
        "valid_confirmation_selected_trials": len(selected_confirmation),
        "valid_search_baseline_trials": len(baseline_search),
        "valid_search_selected_trials": len(selected_search),
        "failed_trials": failed_trials,
        "contaminated_trial_count": len(contaminated_trials),
        "warmup_trial_count": len(warmup_trials),
        "paired_confirmation_count": len(paired),
        "guard_results": guard_dicts(guards),
    }


def _valid_trials(trials: list[Any], *, role: str, phase: str, metric: str) -> list[dict[str, Any]]:
    valid = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        if trial.get("role") != role or trial.get("phase") != phase:
            continue
        if trial.get("trial_status", "valid") != "valid" or trial.get("failed") is True:
            continue
        if _metric_value(trial, metric) is None:
            continue
        valid.append(trial)
    return valid


def _paired_confirmation_pairs(
    baseline_trials: list[dict[str, Any]],
    selected_trials: list[dict[str, Any]],
    metric: str,
) -> list[tuple[float, float]]:
    baseline_by_pair = {trial.get("pair_id"): trial for trial in baseline_trials if trial.get("pair_id")}
    pairs = []
    for selected in selected_trials:
        pair_id = selected.get("pair_id")
        baseline = baseline_by_pair.get(pair_id)
        selected_value = _metric_value(selected, metric)
        baseline_value = _metric_value(baseline or {}, metric)
        if selected_value is not None and baseline_value is not None:
            pairs.append((baseline_value, selected_value))
    return pairs


def _confirmation_direction_consistent(pairs: list[tuple[float, float]], *, direction: str) -> bool:
    if not pairs:
        return False
    if direction == "maximize":
        return all(selected > baseline for baseline, selected in pairs)
    return all(selected < baseline for baseline, selected in pairs)


def _metric_value(trial: dict[str, Any], metric: str) -> float | None:
    raw_metrics = trial.get("metrics", {})
    value = raw_metrics.get(metric) if isinstance(raw_metrics, dict) else trial.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _improvement_percent(selected: float | None, baseline: float | None, direction: str) -> float | None:
    if selected is None or baseline is None or baseline == 0:
        return None
    if direction == "maximize":
        return ((selected - baseline) / abs(baseline)) * 100.0
    return ((baseline - selected) / abs(baseline)) * 100.0


def _variability_percent(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    if mean_value == 0:
        return None
    return (pstdev(values) / abs(mean_value)) * 100.0
