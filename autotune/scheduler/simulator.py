from __future__ import annotations

from statistics import mean, quantiles

from autotune.scheduler.request import CompletedRequest


def summarize(completed: list[CompletedRequest]) -> dict:
    if not completed:
        return {
            "average_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "throughput": 0.0,
            "deadline_miss_rate": 0.0,
        }
    latencies = [item.latency for item in completed]
    total_time = max(item.finish_time for item in completed) - min(item.request.arrival_time for item in completed)
    percentile_values = quantiles(latencies, n=100) if len(latencies) >= 2 else latencies * 100
    return {
        "average_latency": round(mean(latencies), 3),
        "p95_latency": round(percentile_values[94], 3),
        "p99_latency": round(percentile_values[98], 3),
        "throughput": round(len(completed) / total_time, 3) if total_time > 0 else 0.0,
        "deadline_miss_rate": round(sum(item.missed_deadline for item in completed) / len(completed), 3),
    }

