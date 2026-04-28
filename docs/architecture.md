# Architecture

AutoTuneAI-Serve is organized around six modules:

- Model loader and ONNX exporter.
- Runtime backends for PyTorch and ONNX Runtime.
- Profiler for latency, throughput, memory, and hardware metadata.
- Cost model for predicting latency and memory under unseen configurations.
- Auto-tuner for objective and constraint-aware configuration selection.
- Scheduler simulator for inference request batching policies.

