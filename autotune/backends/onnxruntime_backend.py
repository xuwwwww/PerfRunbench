from __future__ import annotations


class ONNXRuntimeBackend:
    name = "onnxruntime"

    def run(self) -> None:
        raise NotImplementedError("ONNX Runtime benchmarking will be added after ONNX export is wired in.")

