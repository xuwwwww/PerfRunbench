from __future__ import annotations


def train_baseline(records: list[dict]) -> dict:
    if not records:
        raise ValueError("training records cannot be empty")
    return {
        "latency_ms_mean": sum(record["latency_ms"] for record in records) / len(records),
        "memory_mb_mean": sum(record["memory_mb"] for record in records) / len(records),
    }

