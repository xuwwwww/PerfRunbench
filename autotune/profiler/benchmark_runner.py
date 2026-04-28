from __future__ import annotations

import csv
import json
import os
from statistics import quantiles
from pathlib import Path

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
        "mode": "synthetic",
        "model": (model_config or {}).get("name", "unknown"),
        **config.__dict__,
        "latency_ms": round(latency_ms, 3),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_p99_ms": round(p99, 3),
        "throughput": round(throughput, 3),
        "memory_mb": round(memory_mb, 3),
    }


def real_profile(
    config: InferenceConfig,
    model_config: dict,
    profiler_config: dict | None = None,
    model_cache: dict | None = None,
) -> dict:
    if config.precision != "fp32":
        raise ValueError("real profiling currently supports fp32 only")

    from autotune.backends.onnxruntime_backend import ONNXRuntimeBackend, summarize_onnx_latencies
    from autotune.backends.pytorch_backend import PyTorchBackend, summarize_latencies
    from autotune.models.model_loader import load_torchvision_model
    from autotune.models.onnx_exporter import export_to_onnx

    profiler = profiler_config or {}
    warmup = int(profiler.get("warmup", 5))
    repeat = int(profiler.get("repeat", 20))
    model_cache = model_cache if model_cache is not None else {}

    model_name = model_config.get("name", "resnet18")
    input_shape = [int(item) for item in model_config.get("input_shape", [1, 3, 224, 224])]
    input_shape[0] = 1
    model = model_cache.get("torch_model")
    if model is None:
        model = load_torchvision_model(model_name)
        model_cache["torch_model"] = model

    memory_before = _current_rss_mb()
    if config.backend == "pytorch":
        backend = PyTorchBackend(model, input_shape, config.thread_count)
        latencies = backend.run(config.batch_size, warmup, repeat)
        summary = summarize_latencies(latencies)
    elif config.backend == "onnxruntime":
        onnx_path = _onnx_path(model_name, profiler)
        if not onnx_path.exists():
            export_to_onnx(model, input_shape, onnx_path)
        backend = ONNXRuntimeBackend(str(onnx_path), input_shape, config.thread_count, config.graph_optimization)
        latencies = backend.run(config.batch_size, warmup, repeat)
        summary = summarize_onnx_latencies(latencies)
    else:
        raise ValueError(f"Unsupported real backend: {config.backend}")
    memory_after = _current_rss_mb()
    latency_ms = summary["latency_ms"]
    throughput = (config.batch_size * 1000.0) / latency_ms if latency_ms else 0.0
    return {
        "mode": "real",
        "model": model_name,
        **config.__dict__,
        **summary,
        "throughput": round(throughput, 3),
        "memory_mb": round(max(memory_before, memory_after), 3),
    }


def filter_real_configs(configs: list[InferenceConfig]) -> list[InferenceConfig]:
    return [config for config in configs if config.precision == "fp32"]


def write_records(records: list[dict], output: str, csv_output: str | None = None) -> None:
    output_path = ensure_parent(output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    if csv_output:
        write_csv_records(records, csv_output)


def write_csv_records(records: list[dict], output: str) -> None:
    if not records:
        return
    output_path = ensure_parent(output)
    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _onnx_path(model_name: str, profiler: dict) -> Path:
    onnx_dir = Path(profiler.get("onnx_dir", "artifacts/onnx"))
    return onnx_dir / f"{model_name}.onnx"


def _current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ModuleNotFoundError:
        return 0.0
