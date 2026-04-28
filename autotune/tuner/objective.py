from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    name: str
    latency_budget_ms: float | None = None
    memory_budget_mb: float | None = None

    @classmethod
    def from_config(cls, config: dict) -> "Objective":
        objective = config.get("objective", {})
        return cls(
            name=objective.get("name", "latency"),
            latency_budget_ms=objective.get("latency_budget_ms"),
            memory_budget_mb=objective.get("memory_budget_mb"),
        )


def satisfies_constraints(record: dict, objective: Objective) -> bool:
    if objective.latency_budget_ms is not None and record["latency_ms"] > objective.latency_budget_ms:
        return False
    if objective.memory_budget_mb is not None and record["memory_mb"] > objective.memory_budget_mb:
        return False
    return True


def score(record: dict, objective: Objective) -> float:
    if objective.name == "throughput":
        return -float(record["throughput"])
    if objective.name == "memory":
        return float(record["memory_mb"])
    return float(record["latency_ms"])

