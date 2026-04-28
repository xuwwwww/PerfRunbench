from __future__ import annotations

import time
from statistics import quantiles


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, model, input_shape: list[int], thread_count: int) -> None:
        import torch

        self.torch = torch
        torch.set_num_threads(thread_count)
        self.model = model
        self.input_shape = input_shape

    def run(self, batch_size: int, warmup: int, repeat: int) -> list[float]:
        shape = [batch_size, *self.input_shape[1:]]
        inputs = self.torch.randn(*shape)
        with self.torch.no_grad():
            for _ in range(warmup):
                self.model(inputs)
            latencies: list[float] = []
            for _ in range(repeat):
                start = time.perf_counter()
                self.model(inputs)
                latencies.append((time.perf_counter() - start) * 1000.0)
        return latencies


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"latency_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}
    if len(latencies) == 1:
        p50 = p95 = p99 = latencies[0]
    else:
        percentile_values = quantiles(latencies, n=100)
        p50, p95, p99 = percentile_values[49], percentile_values[94], percentile_values[98]
    return {
        "latency_ms": round(sum(latencies) / len(latencies), 3),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_p99_ms": round(p99, 3),
    }
