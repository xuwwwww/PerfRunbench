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

Core install for resource guard, system inspector, source-safe tuning, and training wrapper:

```bash
conda env create -f environment.yml
conda activate autotuneai
python scripts/inspect_system.py
python scripts/inspect_executors.py --probe-systemd --check-sudo-cache
python scripts/tune_system.py
python scripts/run_with_budget.py -- python examples/dummy_train.py
python -m unittest discover -s tests
```

In WSL or non-interactive shells, prefer the explicit conda path:

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/inspect_system.py
```

Replace `/home/louis/miniforge3/bin/conda` with the user's own conda executable path on other machines.

Executor capability detection reports the best available resource backend on the current machine:

```bash
python scripts/inspect_executors.py --probe-systemd --check-sudo-cache
```

The current implemented executors are `local` and `systemd`. Docker, Windows Job Object, and macOS-specific execution are detected as planned backends so the tool can grow beyond Linux without changing the user-facing workflow.

Runtime system tuning can be previewed without changing the machine:

```bash
python scripts/tune_system.py
```

On Linux/WSL, applying writes before/after/diff snapshots under `.autotuneai/runs/<run_id>/`:

```bash
sudo -v
python scripts/tune_system.py --apply --sudo
python scripts/restore_run.py --run-id <run_id> --sudo
```

Benchmark install for PyTorch / ONNX Runtime experiments:

```bash
python -m pip install -r requirements-benchmark.txt
python scripts/run_benchmark.py --config configs/resnet18.yaml --mode real --backends pytorch --max-configs 1
```

AutoTuneAI can wrap a training command from another environment:

```bash
python scripts/run_with_budget.py \
  --executor auto \
  --memory-budget-gb 22 \
  -- /path/to/user/env/bin/python train.py
```

When `--executor auto` selects systemd and the machine requires sudo, it stops with a clear message unless you opt in:

```bash
sudo -v
python scripts/run_with_budget.py \
  --executor auto \
  --allow-sudo-auto \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py
```

For hard memory/CPU limits on Linux systems with systemd:

```bash
sudo -v

python scripts/check_system_executor.py \
  --sudo \
  --check-sudo-cache \
  --probe \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py

python scripts/run_with_budget.py \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py
```

Use `sudo -v` plus `--sudo`; do not run AutoTuneAI from a `sudo su` root shell. The root shell may not have the user's conda initialization and can point at the wrong conda path. AutoTuneAI should stay in the user environment while systemd/cgroup limits are requested through sudo.

Systemd runs use a named scope such as `autotuneai-<run_id>.scope`. When available, AutoTuneAI samples the scope cgroup files (`memory.current`, `memory.peak`, `cpu.stat`) and writes cgroup-level memory/CPU fields to `resource_timeline.json` and `resource_summary.json`.

User guide:

```text
USER_GUIDE.md
```

If the environment already exists:

```bash
conda activate autotuneai
python -m pip install -r requirements-core.txt
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

## Resource Budgets

Benchmark configs can reserve RAM and CPU capacity for the rest of the system:

```yaml
resource_budget:
  memory_budget_gb: 22
  reserve_memory_gb: 1.7
  reserve_cores: 1
  cpu_quota_percent: 90
  enforce: true
```

For real benchmarks, the CLI applies the budget by:

- Filtering out configurations whose `thread_count` exceeds the allowed CPU threads.
- Applying process CPU affinity so one or more cores can remain available for other work.
- Sampling process RSS and CPU usage during the benchmark.
- Recording `peak_rss_mb`, `effective_memory_budget_mb`, `available_memory_before_mb`, `available_memory_after_mb`, and CPU utilization fields.

You can override the config from the command line:

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90
```

In WSL, the Linux VM may expose less RAM than Windows reports. The benchmark records both `memory_budget_mb` and `effective_memory_budget_mb`; the effective budget is capped by the RAM visible inside WSL minus reserved memory.

## Repository Layout

```text
autotune/
  backends/       Runtime backend adapters.
  models/         Model loading and export boundaries.
  profiler/       Benchmark and hardware metadata collection.
  resource/       Resource budgets, CPU affinity, and runtime monitoring.
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
