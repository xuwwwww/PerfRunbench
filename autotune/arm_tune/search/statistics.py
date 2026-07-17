from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median, stdev
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import to_plain


PERCENTILE_MIN_SAMPLES = 5


@dataclass(frozen=True)
class MetricDistribution:
    metric_name: str
    valid_count: int
    failed_count: int
    median: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None
    iqr: float | None = None
    standard_deviation: float | None = None
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NoiseBandDecision:
    classification: str
    baseline_median: float | None
    candidate_median: float | None
    delta_percent: float | None
    noise_band_percent: float
    message: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def summarize_metric(metric_name: str, values: list[float], *, failed_count: int) -> MetricDistribution:
    clean = sorted(float(value) for value in values)
    warnings: list[str] = []
    if len(clean) < PERCENTILE_MIN_SAMPLES:
        warnings.append(
            f"p95 requires at least {PERCENTILE_MIN_SAMPLES} valid samples; observed {len(clean)}"
        )
    if not clean:
        return MetricDistribution(metric_name=metric_name, valid_count=0, failed_count=failed_count, warnings=warnings)
    return MetricDistribution(
        metric_name=metric_name,
        valid_count=len(clean),
        failed_count=failed_count,
        median=round(median(clean), 6),
        p95=_percentile(clean, 95) if len(clean) >= PERCENTILE_MIN_SAMPLES else None,
        min=round(clean[0], 6),
        max=round(clean[-1], 6),
        iqr=_iqr(clean) if len(clean) >= 4 else None,
        standard_deviation=round(stdev(clean), 6) if len(clean) >= 2 else None,
        warnings=warnings,
    )


def classify_noise_band(
    *,
    baseline: MetricDistribution,
    candidate: MetricDistribution,
    noise_band_percent: float,
) -> NoiseBandDecision:
    if baseline.median is None or candidate.median is None or baseline.median <= 0:
        return NoiseBandDecision(
            classification="not_comparable",
            baseline_median=baseline.median,
            candidate_median=candidate.median,
            delta_percent=None,
            noise_band_percent=noise_band_percent,
            message="missing comparable median metrics",
        )
    delta = ((candidate.median - baseline.median) / baseline.median) * 100.0
    if delta > noise_band_percent:
        classification = "improved"
        message = "candidate median is above the configured noise band"
    elif delta < -noise_band_percent:
        classification = "regressed"
        message = "candidate median is below the configured noise band"
    else:
        classification = "inconclusive"
        message = "candidate median is within the configured noise band"
    return NoiseBandDecision(
        classification=classification,
        baseline_median=baseline.median,
        candidate_median=candidate.median,
        delta_percent=round(delta, 6),
        noise_band_percent=noise_band_percent,
        message=message,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return round(values[0], 6)
    rank = (len(values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 6)


def _iqr(values: list[float]) -> float:
    lower_half = values[: len(values) // 2]
    upper_half = values[(len(values) + 1) // 2 :]
    return round(float(median(upper_half) - median(lower_half)), 6)
