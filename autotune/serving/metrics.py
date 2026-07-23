from __future__ import annotations

from statistics import median

from autotune.serving.schemas import ServingMetricSummary, ServingRequestEvidence


PERCENTILE_MIN_SAMPLES = 5


def summarize_requests(requests: list[ServingRequestEvidence]) -> dict[str, ServingMetricSummary]:
    failed = sum(1 for item in requests if item.error_category or item.http_status is None or item.http_status >= 400)
    ttft = [item.client_observed_timings["ttft_ms"] for item in requests if "ttft_ms" in item.client_observed_timings and not item.error_category]
    e2e = [item.client_observed_timings["e2e_latency_ms"] for item in requests if "e2e_latency_ms" in item.client_observed_timings and not item.error_category]
    output_tokens = [float(item.output_tokens) for item in requests if item.output_tokens is not None and not item.error_category]
    input_tokens = [float(item.input_tokens) for item in requests if item.input_tokens is not None and not item.error_category]
    wall_seconds = _wall_seconds(requests)
    completed = len(requests) - failed
    return {
        "completed_requests_per_second": _single_rate("completed_requests_per_second", completed, wall_seconds, failed),
        "aggregate_output_tokens_per_second": _single_rate("aggregate_output_tokens_per_second", sum(output_tokens), wall_seconds, failed),
        "aggregate_total_tokens_per_second": _single_rate(
            "aggregate_total_tokens_per_second",
            sum(input_tokens) + sum(output_tokens),
            wall_seconds,
            failed,
        ),
        "input_prefill_tokens_per_second": _single_rate("input_prefill_tokens_per_second", sum(input_tokens), wall_seconds, failed),
        "ttft_ms": summarize_values("ttft_ms", ttft, failed_count=failed),
        "e2e_latency_ms": summarize_values("e2e_latency_ms", e2e, failed_count=failed),
        "output_tokens": summarize_values("output_tokens", output_tokens, failed_count=failed),
    }


def summarize_values(metric_name: str, values: list[float], *, failed_count: int) -> ServingMetricSummary:
    clean = sorted(float(value) for value in values)
    warnings: list[str] = []
    if len(clean) < PERCENTILE_MIN_SAMPLES:
        warnings.append(f"p95/p99 require at least {PERCENTILE_MIN_SAMPLES} valid samples; observed {len(clean)}")
    if not clean:
        return ServingMetricSummary(metric_name=metric_name, valid_count=0, failed_count=failed_count, warnings=warnings)
    return ServingMetricSummary(
        metric_name=metric_name,
        valid_count=len(clean),
        failed_count=failed_count,
        p50=round(float(median(clean)), 6),
        p95=_percentile(clean, 95) if len(clean) >= PERCENTILE_MIN_SAMPLES else None,
        p99=_percentile(clean, 99) if len(clean) >= PERCENTILE_MIN_SAMPLES else None,
        min=round(clean[0], 6),
        max=round(clean[-1], 6),
        warnings=warnings,
    )


def _single_rate(metric_name: str, numerator: float, wall_seconds: float | None, failed_count: int) -> ServingMetricSummary:
    if wall_seconds is None or wall_seconds <= 0:
        return ServingMetricSummary(
            metric_name=metric_name,
            valid_count=0,
            failed_count=failed_count,
            warnings=["rate unavailable because wall time is not available"],
        )
    return ServingMetricSummary(
        metric_name=metric_name,
        valid_count=1,
        failed_count=failed_count,
        p50=round(float(numerator) / wall_seconds, 6),
        min=round(float(numerator) / wall_seconds, 6),
        max=round(float(numerator) / wall_seconds, 6),
    )


def _wall_seconds(requests: list[ServingRequestEvidence]) -> float | None:
    completed = [item for item in requests if item.completed_timestamp is not None]
    if not completed:
        return None
    start = min(item.submitted_timestamp for item in completed)
    end = max(float(item.completed_timestamp) for item in completed)
    return max(0.0, end - start)


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return round(values[0], 6)
    rank = (len(values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 6)
