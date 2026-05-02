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
python -m pip install -e .
autotuneai inspect
autotuneai executors --probe-systemd --probe-docker --check-sudo-cache
autotuneai tune-system
autotuneai run -- python examples/dummy_train.py
python -m unittest discover -s tests
```

In WSL or non-interactive shells, prefer the explicit conda path:

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/inspect_system.py
```

Replace `/home/louis/miniforge3/bin/conda` with the user's own conda executable path on other machines.

Executor capability detection reports the best available resource backend on the current machine:

```bash
autotuneai executors --probe-systemd --probe-docker --check-sudo-cache
```

The current implemented executors are `local`, `systemd`, and `docker`. Windows Job Object and macOS-specific native execution are detected as planned backends so the tool can grow beyond Linux without changing the user-facing workflow.

Runtime system tuning can be previewed without changing the machine:

```bash
autotuneai tune-system
autotuneai tune-system --profile linux-memory-conservative
autotuneai tune-system --profile linux-throughput
autotuneai tune-system --profile linux-low-latency
```

On Linux/WSL, applying writes before/after/diff snapshots under `.autotuneai/runs/<run_id>/`:

```bash
sudo -v
autotuneai tune-system --apply --sudo
autotuneai restore --run-id <run_id> --sudo
```

You can also let a workload automatically apply the recommended runtime tuning profile before it starts and restore the previous values afterward:

```bash
sudo -v
autotuneai run \
  --auto-tune-system \
  --system-tuning-sudo \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  -- /path/to/user/env/bin/python train.py
```

Runs that apply runtime tuning write `system_tuning_before.json`, `system_tuning_after.json`, and `system_tuning_diff.json` under `.autotuneai/runs/<run_id>/`.

Available runtime tuning profiles:

```text
linux-training-safe
  General conservative training profile.

linux-memory-conservative
  More aggressive memory headroom profile for RAM-constrained training.

linux-throughput
  Throughput-oriented profile for dataset/checkpoint-heavy runs.

linux-low-latency
  Lower dirty-page and THP settings for smoother latency.
```

Resource guard smoke test with CPU and memory load:

```bash
autotuneai run \
  --executor local \
  --memory-budget-gb -1 \
  --reserve-cores 1 \
  --cpu-quota-percent 50 \
  --sample-interval-seconds 0.1 \
  -- python scripts/stress_workload.py --workers 4 --duration-seconds 10 --memory-mb 512
```

`--memory-budget-gb -1` means "cap workload memory at visible RAM minus 1GB"; positive values remain absolute GB targets.

Analyze a run after testing:

```bash
autotuneai analyze --run-id <run_id>
autotuneai report --run-id <run_id>
```

The analyzer reports selected executor, affinity cores, expected CPU cap, observed peak CPU, effective memory budget, observed memory headroom, and cgroup stats when available.

Benchmark install for PyTorch / ONNX Runtime experiments:

```bash
python -m pip install -r requirements-benchmark.txt
python scripts/run_benchmark.py --config configs/resnet18.yaml --mode real --backends pytorch --max-configs 1
```

AutoTuneAI can wrap a training command from another environment:

```bash
autotuneai run \
  --executor auto \
  --memory-budget-gb 22 \
  -- /path/to/user/env/bin/python train.py
```

When `--executor auto` selects systemd and the machine requires sudo, it stops with a clear message unless you opt in:

```bash
sudo -v
autotuneai run \
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

autotuneai run \
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

Memory budget calibration is available when you need to measure how a machine behaves with negative reserve-to-full budgets:

```bash
autotuneai calibrate-memory \
  --budget-gb -5 -3 1 \
  --workload-memory-mb 2048 \
  --duration-seconds 10
```

Generic training config tuning can edit any single integer key and restore the file after each trial:

```bash
autotuneai tune-batch \
  --file configs/train.yaml \
  --key num_workers \
  --values 0 2 4 8 \
  -- python train.py --config configs/train.yaml
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
