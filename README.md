# AutoTuneAI-Serve

Cost-model-guided inference optimization and scheduling framework for resource-constrained AI systems.

This repository is being built as a reproducible AI systems project. The first version includes a working CLI, configuration search, synthetic profiling data, real PyTorch CPU benchmarking, ONNX export, ONNX Runtime CPU benchmarking, and inference request scheduling.

## Goals

- Profile inference configurations across backend, batch size, precision, threads, and graph optimization.
- Train or plug in lightweight cost models for latency, throughput, and memory.
- Compare exhaustive, random, and cost-model-guided search.
- Simulate FCFS, static batching, dynamic batching, and deadline-aware dynamic batching.
- Generate reproducible experiment outputs for a technical report and project webpage.

## Quick Start

```bash
conda env create -f environment.yml
conda activate autotuneai
python scripts/run_benchmark.py --config configs/resnet18.yaml
python scripts/run_autotune.py --config configs/resnet18.yaml --search exhaustive
python scripts/run_scheduler.py --workload burst --scheduler deadline_aware
python -m unittest discover -s tests
```

If the environment already exists:

```bash
conda activate autotuneai
python -m pip install -r requirements.txt
python scripts/run_all_experiments.py
```

## Real Benchmark Smoke Tests

Run a single PyTorch CPU configuration:

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --output results/raw/resnet18_pytorch_smoke.json \
  --csv-output results/raw/resnet18_pytorch_smoke.csv
```

Run a single ONNX Runtime CPU configuration:

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends onnxruntime \
  --max-configs 1 \
  --output results/raw/resnet18_onnx_smoke.json \
  --csv-output results/raw/resnet18_onnx_smoke.csv
```

The first ONNX Runtime run exports `artifacts/onnx/resnet18.onnx`. Generated ONNX files and result files are ignored by git.

## Repository Layout

```text
autotune/
  backends/       Runtime backend adapters.
  models/         Model loading and export boundaries.
  profiler/       Benchmark and hardware metadata collection.
  tuner/          Search spaces, objectives, and tuning strategies.
  cost_model/     Dataset, training, prediction, and evaluation helpers.
  scheduler/      Request workload simulation and batching policies.
  report/         Markdown and figure generation helpers.
  utils/          Shared config, logging, and timing utilities.
configs/          Experiment configurations.
scripts/          CLI entry points.
results/          Generated experiment artifacts.
docs/             Architecture, methodology, and report drafts.
tests/            Unit tests for core logic.
```

## Current Status

The current implementation supports synthetic experiments for fast development and real CPU benchmark smoke tests for ResNet18. Real profiling currently supports `fp32`; INT8 quantization is planned as a later experiment stage.
