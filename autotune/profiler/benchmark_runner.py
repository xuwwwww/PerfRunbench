from __future__ import annotations

import json
from statistics import quantiles

from autotune.tuner.search_space import InferenceConfig
from autotune.utils.config import ensure_parent


def synthetic_profile(config: InferenceConfig, model_config: dict | None = None) -> dict:
    params = (model_config or {}).get("parameter_count", 10_000_000)
    model_factor = params / 10_000_000
    backend_factor = 0.82 if config.backend == "onnxruntime" else 1.0
    precision_factor = 0.72 if config.precision == "int8" else 1.0
    graph_factor = {
        "disable": 1.0,
        "basic": 0.94,
        "extended": 0.89,
        "all": 0.86,
    }.get(config.graph_optimization, 1.0)
    thread_factor = max(0.52, 1.0 / (config.thread_count ** 0.35))
    batch_efficiency = 0.78 + 0.22 / max(config.batch_size, 1)
    latency_ms = 7.5 * model_factor * backend_factor * precision_factor * graph_factor
    latency_ms *= thread_factor * config.batch_size * batch_efficiency
    throughput = (config.batch_size * 1000.0) / latency_ms
    memory_mb = 180 + model_factor * 85 + config.batch_size * 18
    if config.precision == "int8":
        memory_mb *= 0.72
    samples = [latency_ms * (0.96 + i * 0.004) for i in range(20)]
    p50, p95, p99 = quantiles(samples, n=100)[49], quantiles(samples, n=100)[94], quantiles(samples, n=100)[98]
    return {
        **config.__dict__,
        "latency_ms": round(latency_ms, 3),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_p99_ms": round(p99, 3),
        "throughput": round(throughput, 3),
        "memory_mb": round(memory_mb, 3),
    }


def write_records(records: list[dict], output: str) -> None:
    output_path = ensure_parent(output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

