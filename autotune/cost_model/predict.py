from __future__ import annotations

from autotune.profiler.benchmark_runner import synthetic_profile
from autotune.tuner.search_space import InferenceConfig


def synthetic_predict(config: InferenceConfig, model_config: dict | None = None) -> dict:
    record = synthetic_profile(config, model_config)
    record["latency_ms"] = round(record["latency_ms"] * 1.04, 3)
    record["memory_mb"] = round(record["memory_mb"] * 0.98, 3)
    return record

