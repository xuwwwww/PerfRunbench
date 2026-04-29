from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecommendationRequest:
    objective: str = "throughput"
    latency_budget_ms: float | None = None
    memory_budget_mb: float | None = None
    require_safe: bool = True


def recommend_config(records: list[dict[str, Any]], request: RecommendationRequest) -> tuple[dict[str, Any], list[str]]:
    if not records:
        raise ValueError("records cannot be empty")
    candidates = [record for record in records if _is_candidate_safe(record, request)]
    if not candidates:
        candidates = records
    best = sorted(candidates, key=lambda record: _score(record, request.objective))[0]
    return best, build_reasoning(best, records, request, used_fallback=candidates is records)


def build_reasoning(
    best: dict[str, Any],
    records: list[dict[str, Any]],
    request: RecommendationRequest,
    used_fallback: bool = False,
) -> list[str]:
    safe_count = sum(_is_candidate_safe(record, request) for record in records)
    reasons = []
    if used_fallback:
        reasons.append("No record satisfied every safety constraint; recommendation is the best fallback.")
    else:
        reasons.append(f"Selected from {safe_count} safe records out of {len(records)} measured records.")
    if request.objective == "throughput":
        reasons.append("Objective optimized highest measured throughput among eligible records.")
    elif request.objective == "latency":
        reasons.append("Objective optimized lowest measured p95 latency among eligible records.")
    elif request.objective == "memory":
        reasons.append("Objective optimized lowest measured peak RSS among eligible records.")
    if request.latency_budget_ms is not None:
        reasons.append(
            f"p95 latency {best.get('latency_p95_ms')} ms compared with budget {request.latency_budget_ms} ms."
        )
    effective_budget = _effective_memory_budget(best, request)
    if effective_budget is not None:
        reasons.append(f"peak RSS {best.get('peak_rss_mb', best.get('memory_mb'))} MB compared with budget {effective_budget} MB.")
    if best.get("cpu_affinity_applied"):
        reasons.append(f"CPU affinity applied to cores {best.get('affinity_cores')}.")
    return reasons


def _is_candidate_safe(record: dict[str, Any], request: RecommendationRequest) -> bool:
    if request.require_safe:
        if record.get("memory_budget_exceeded") is True:
            return False
        if record.get("cpu_quota_exceeded") is True:
            return False
    if request.latency_budget_ms is not None:
        latency = record.get("latency_p95_ms", record.get("latency_ms"))
        if latency is not None and float(latency) > request.latency_budget_ms:
            return False
    memory_budget = _effective_memory_budget(record, request)
    if memory_budget is not None:
        memory = record.get("peak_rss_mb", record.get("memory_mb"))
        if memory is not None and float(memory) > memory_budget:
            return False
    return True


def _effective_memory_budget(record: dict[str, Any], request: RecommendationRequest) -> float | None:
    budgets = []
    if request.memory_budget_mb is not None:
        budgets.append(request.memory_budget_mb)
    if record.get("effective_memory_budget_mb") is not None:
        budgets.append(float(record["effective_memory_budget_mb"]))
    elif record.get("memory_budget_mb") is not None:
        budgets.append(float(record["memory_budget_mb"]))
    return min(budgets) if budgets else None


def _score(record: dict[str, Any], objective: str) -> float:
    if objective == "throughput":
        return -float(record.get("throughput", 0.0))
    if objective == "memory":
        return float(record.get("peak_rss_mb", record.get("memory_mb", float("inf"))))
    if objective == "latency":
        return float(record.get("latency_p95_ms", record.get("latency_ms", float("inf"))))
    raise ValueError(f"Unsupported objective: {objective}")

