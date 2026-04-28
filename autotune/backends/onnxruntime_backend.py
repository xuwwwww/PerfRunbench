from __future__ import annotations

import time

import numpy as np

from autotune.backends.pytorch_backend import summarize_latencies


class ONNXRuntimeBackend:
    name = "onnxruntime"

    def __init__(self, model_path: str, input_shape: list[int], thread_count: int, graph_optimization: str) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = thread_count
        options.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }.get(graph_optimization, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = input_shape

    def run(self, batch_size: int, warmup: int, repeat: int) -> list[float]:
        shape = [batch_size, *self.input_shape[1:]]
        inputs = np.random.randn(*shape).astype("float32")
        feeds = {self.input_name: inputs}
        for _ in range(warmup):
            self.session.run(None, feeds)
        latencies: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            self.session.run(None, feeds)
            latencies.append((time.perf_counter() - start) * 1000.0)
        return latencies


def summarize_onnx_latencies(latencies: list[float]) -> dict[str, float]:
    return summarize_latencies(latencies)
