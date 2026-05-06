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
autotuneai report-comparison --input results/reports/tuning_comparison.json
python scripts/run_tests.py --fast
```

Benchmark/GPU install for server-side performance optimization:

```bash
conda env create -f environment-benchmark.yml
conda activate autotuneai-benchmark
python -m pip install -e .

python - <<'PY'
import google.protobuf
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python scripts/run_tests.py --fast
```

`environment.yml` is intentionally small and does not install PyTorch. Use `environment-benchmark.yml` or `python -m pip install -e ".[benchmark]"` when running PyTorch, ONNX, or GPU pressure benchmarks. If the server needs a specific CUDA wheel, install the matching `torch`/`torchvision` build from the official PyTorch selector before running GPU benchmarks.

Fresh Linux server bootstrap:

```bash
git clone -b feature/performance-optimizer https://github.com/xuwwwww/PerfRunbench.git
cd PerfRunbench

# If conda/mamba already exists:
bash scripts/bootstrap_server_env.sh

# If conda is missing on the server:
bash scripts/bootstrap_server_env.sh --install-miniforge

# Optional: replace torch/torchvision with a specific CUDA wheel index.
bash scripts/bootstrap_server_env.sh --update --pytorch-index-url https://download.pytorch.org/whl/cu121
```

The bootstrap script does not require `conda activate`; it uses `conda run`, installs the editable package, checks `torch`, `google.protobuf`, CUDA visibility, `nvidia-smi`, executor capabilities, and the fast test suite.

If the server environment already exists but a command fails, run read-only diagnostics first:

```bash
bash scripts/diagnose_server_env.sh --env-name autotuneai-benchmark
```

This isolates common server failures: missing conda env, missing editable install, `ModuleNotFoundError: torch`, `ModuleNotFoundError: google`, CUDA not visible to PyTorch, missing `nvidia-smi`, or systemd/sudo executor issues.

Built-in repo demo workflows:

```bash
autotuneai demo
autotuneai demo --scenario tune-batch
autotuneai demo --scenario compare-tuning --executor systemd --sudo --system-tuning-sudo --memory-budget-gb -3
```

Real training workloads bundled in the repo:

```bash
autotuneai run -- python examples/iris_train.py --config examples/iris_train_config.yaml
autotuneai run --memory-budget-gb 1.5 --hard-kill -- python examples/stress_train.py --config examples/stress_train_config.yaml
autotuneai run --memory-budget-gb -3 --hard-kill -- python examples/heavy_training_pressure.py --config examples/heavy_training_pressure_config.yaml
autotuneai run --runtime-profile runtime-pytorch-max-performance -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_config.yaml
autotuneai tune-training \
  --file examples/iris_train_config.yaml \
  --knob batch_size=8,16,32 \
  --knob gradient_accumulation_steps=1,2,4 \
  --knob preload_copies=4,8,12 \
  -- python examples/iris_train.py --config examples/iris_train_config.yaml
sudo -v
autotuneai compare-tuning \
  --workload-profile memory \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --memory-budget-gb -3 \
  --repeat 3 \
  -- python examples/stress_train.py --config examples/stress_train_60s_config.yaml
```

In WSL or non-interactive shells, prefer the explicit path to the conda executable on that machine:

```bash
/path/to/conda run -n autotuneai python scripts/inspect_system.py
```

Replace `/path/to/conda` with `which conda`, `which micromamba`, or the absolute path used by the host environment.

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
autotuneai tune-system --profile linux-performance
autotuneai tune-system --profile linux-low-latency
autotuneai tune-system --profile windows-throughput
autotuneai tune-system --profile windows-performance
autotuneai tune-system --recommend-all
autotuneai tune-runtime --profile runtime-pytorch-max-performance
autotuneai tune-gpu --profile nvidia-performance
```

On Linux/WSL, applying writes before/after/diff snapshots under `.autotuneai/runs/<run_id>/`:

```bash
sudo -v
autotuneai tune-system --apply --sudo
autotuneai restore --run-id <run_id> --sudo
autotuneai restore --latest --sudo
autotuneai restore --active --sudo
```

On Windows, runtime system tuning currently uses reversible `powercfg` active power scheme changes:

```powershell
autotuneai tune-system --profile windows-throughput --apply
autotuneai restore --run-id <run_id>
```

You can also let a workload automatically apply the recommended runtime tuning profile before it starts and restore the previous values afterward:

```bash
sudo -v
autotuneai run \
  --auto-tune-system \
  --runtime-profile runtime-pytorch-max-performance \
  --system-tuning-sudo \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  -- /path/to/user/env/bin/python train.py
```

Runs that apply runtime tuning write `system_tuning_before.json`, `system_tuning_after.json`, and `system_tuning_diff.json` under `.autotuneai/runs/<run_id>/`.
Runs that apply process-level runtime environment tuning write `runtime_env_tuning.json` under the run directory. These changes are child-process environment variables, so they do not need machine-level restore.

Available runtime tuning profiles:

```text
linux-training-safe
  General conservative training profile.

linux-memory-conservative
  More aggressive memory headroom profile for RAM-constrained training.

linux-throughput
  Throughput-oriented profile for dataset/checkpoint-heavy runs.

linux-performance
  Aggressive throughput-first Linux/WSL profile with larger dirty-page windows and forced THP.

linux-low-latency
  Lower dirty-page and THP settings for smoother latency.

linux-cpu-conservative
  CPU quota / reserved-core profile that reduces kernel background flush bursts.

windows-training-safe
  General Windows training profile using a temporary high performance power scheme.

windows-memory-conservative
  Windows memory-budget profile; memory guard stays in AutoTuneAI, while CPU frequency scaling noise is reduced.

windows-throughput
  Throughput-oriented Windows profile using a temporary high performance power scheme.

windows-performance
  Aggressive throughput-first Windows profile using the reversible high performance power scheme path.

windows-low-latency
  Latency-oriented Windows profile using a temporary high performance power scheme.

windows-cpu-conservative
  CPU/thermal conservative Windows profile using a temporary balanced power scheme.

runtime-cpu-performance
  Set OpenMP/BLAS thread counts and CPU binding hints for CPU-heavy PyTorch/NumPy/BLAS workloads.

runtime-pytorch-gpu-performance
  Limit CPU helper thread oversubscription and enable PyTorch CUDA throughput-oriented environment variables.

runtime-pytorch-max-performance
  Combine CPU thread tuning and PyTorch CUDA throughput variables for aggressive benchmark runs.
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
`autotuneai run`, `compare-tuning`, `compare-budgets`, and `compare-profiles` now also emit sibling `.html` reports automatically, so you do not need a separate conversion step after each benchmark.

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
  --file examples/train_config.yaml \
  --key dataloader_workers \
  --values 0 2 4 8 \
  -- python examples/dummy_train.py
```

Compare an untuned run against a tuned run with the same command:

For a repo-local smoke test, use the bundled dummy workload instead of `train.py`:

```bash
sudo -v
autotuneai compare-tuning \
  --workload-profile memory \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --memory-budget-gb -3 \
  --sample-interval-seconds 0.1 \
  -- python examples/dummy_train.py
```

On Windows, use the local executor and a Windows runtime profile:

```powershell
autotuneai compare-tuning `
  --workload-profile throughput `
  --executor local `
  --sample-interval-seconds 0.1 `
  --repeat 3 `
  -- python examples/stress_train.py --config examples/stress_train_60s_config.yaml
```

For memory-pressure comparisons, use `examples/stress_train_memory_pressure_config.yaml` with a negative memory budget such as `--memory-budget-gb -3`. Comparison reports use `benchmark_duration_seconds`, which subtracts runtime tuning apply/restore time; workload quality metrics such as accuracy/loss/dice are intentionally excluded from `tuning_comparison.json`.

For a more generic high-pressure benchmark that is not tied to Iris classification semantics, use:

```bash
autotuneai run \
  --memory-budget-gb -3 \
  --hard-kill \
  -- python examples/heavy_training_pressure.py --config examples/heavy_training_pressure_config.yaml
```

`examples/heavy_training_pressure.py` is a synthetic 60-second CPU and memory pressure workload. It is useful for validating the wrapper, cgroup guard, monitoring, restore behavior, and rough Linux runtime effects. It is not a substitute for a real PyTorch/CUDA benchmark because it does not exercise DataLoader, CUDA kernels, cuDNN/cuBLAS, or GPU memory allocation.
For GPU pressure, use `examples/gpu_training_pressure.py`. It refuses to run when CUDA is unavailable, allocates GPU memory, runs CUDA matrix multiplications, and emits GPU metrics such as estimated TFLOPS, peak allocated GPU memory, and sampled step latency percentiles (`step_time_p50_seconds`, `step_time_p95_seconds`, `step_time_p99_seconds`). Use `examples/gpu_training_pressure_sweep_config.yaml` for fast recommendation sweeps and `examples/gpu_training_pressure_config.yaml` for longer confirmation runs.

If a run is interrupted after runtime tuning was applied, AutoTuneAI records `.autotuneai/active_tuning_state.json`. Use `autotuneai restore --active` to revert to the pre-run system state without manually finding the run id.

Visual reports are available for both single runs and tuning comparisons:

```bash
autotuneai report --run-id <run_id>
autotuneai report-comparison --input results/reports/tuning_comparison.json
autotuneai report-comparison --input results/reports/tuning_comparison.json --output results/reports/tuning_comparison_report.html
autotuneai compare-profiles --repeat 3 -- python examples/heavy_training_pressure.py --config examples/heavy_training_pressure_config.yaml
autotuneai compare-budgets --memory-budget-gb -3 --profile linux-low-latency --repeat 3 -- python examples/heavy_training_pressure.py --config examples/heavy_training_pressure_config.yaml
```

NVIDIA runtime tuning is available when `nvidia-smi` is on PATH:

```bash
autotuneai tune-gpu
sudo -v
autotuneai tune-gpu --apply --sudo --profile nvidia-performance
autotuneai tune-gpu --apply --sudo --profile nvidia-guard
autotuneai restore --run-id <run_id> --gpu-sudo
```

GPU profiles are reversible runtime controls:

- `nvidia-performance` chases throughput with persistence mode, max power limit, and max supported application clocks when the driver allows it.
- `nvidia-balanced` caps power around 80% of the min/max range and chooses mid-high supported application clocks for lower heat with less throughput loss.
- `nvidia-guard` is the resource-guard profile: it attempts the minimum reported power limit and lowest supported application clocks. This is the GPU-side counterpart to CPU/memory guard limits.

Linux `linux-performance` also attempts reversible cpufreq tuning when the kernel exposes it: `scaling_governor=performance`, energy performance preference, and an aggressive minimum frequency during the run. AutoTuneAI does not perform voltage increases or BIOS/firmware overclocking as a generic runtime action; it only uses OS/driver controls that can be snapshotted and restored.

For a single training run, GPU tuning can be applied and restored as part of the run lifecycle:

```bash
sudo -v
autotuneai run \
  --auto-tune-system \
  --auto-tune-gpu \
  --runtime-profile runtime-pytorch-max-performance \
  --system-tuning-sudo \
  --gpu-tuning-sudo \
  --executor systemd \
  --sudo \
  --memory-budget-gb -3 \
  -- python train.py
```

Aggressive A/B comparison for a formal benchmark:

```bash
sudo -v
autotuneai compare-tuning \
  --profile linux-throughput \
  --runtime-profile runtime-pytorch-max-performance \
  --gpu-profile nvidia-performance \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --gpu-tuning-sudo \
  --memory-budget-gb -3 \
  --sample-interval-seconds 0.1 \
  --repeat 3 \
  --cooldown-seconds 8 \
  -- /path/to/benchmark/env/bin/python train_or_benchmark.py --config config.yaml
```

Performance-only recommendation search for servers:

```bash
sudo -v
autotuneai optimize-performance \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --gpu-tuning-sudo \
  --target gpu \
  --monitor-mode minimal \
  --time-budget-hours 0.5 \
  --max-candidates 8 \
  --repeat 2 \
  --warmup-runs 1 \
  --cooldown-seconds 2 \
  -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_sweep_config.yaml

autotuneai launch-performance \
  --apply-recommendation \
  --recommendation .autotuneai/recommendations/latest_performance_gpu.json \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --gpu-tuning-sudo \
  -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_config.yaml
```

`optimize-performance` is for raw speed. It does not apply memory budgets, CPU quotas, or reserved-core guard limits. The default `--monitor-mode minimal` launches candidates without AutoTuneAI's per-sample resource monitor, so ranking is driven by workload metrics such as `samples_per_second` or `gpu_tflops_estimate`; sampled `step_time_p95_seconds` is used as a stability tie-breaker when throughput is close. Recommendation JSON and HTML include a `decision` section that compares the recommended profile against baseline, applies a default +/-2% noise band, and explains whether the result is a meaningful speedup, baseline-best result, or too close to trust without a longer repeat. Performance sweeps use paired baseline controls by default, so candidates are ranked by speed relative to a nearby baseline run instead of raw cold-start throughput. Use `--target gpu`, `--target cpu`, or `--target memory` when `--max-candidates` is small and you want the early candidate set to focus on one bottleneck. Targeted sweeps use target-specific defaults, for example `results/reports/performance_recommendation_gpu.json` and `.autotuneai/recommendations/latest_performance_gpu.json`; `--target auto` keeps the legacy `performance_recommendation.json` and `latest.json` paths.

The sweep command above is intentionally short: roughly 8 seconds of measured GPU work per candidate plus 1 second warmup. If a short sweep finds a non-baseline winner, confirm the cached profile on the longer config or on the real training command with `launch-performance`.

`launch-performance --apply-recommendation` applies the cached `system_profile`, `runtime_profile`, and `gpu_profile`, starts the real workload without resource monitoring, waits for it to finish, and restores the original system/GPU settings even on nonzero exit or Ctrl+C. If baseline is still the fastest candidate, the cached recommendation can still be launched this way to avoid measurement overhead during the real run.

Guarded recommendation search for local machines:

```bash
sudo -v
autotuneai optimize \
  --executor systemd \
  --sudo \
  --system-tuning-sudo \
  --gpu-tuning-sudo \
  --target gpu \
  --memory-budget-gb -3 \
  --sample-interval-seconds 0.1 \
  --repeat 1 \
  --warmup-runs 1 \
  --cooldown-seconds 8 \
  -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_config.yaml

autotuneai run --apply-recommendation -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_config.yaml
```

`optimize` evaluates curated candidates across guard mode, system profile, runtime environment profile, and NVIDIA GPU profile. In guarded mode, NVIDIA candidates include `nvidia-balanced` and `nvidia-guard`, so GPU power/clocks can be capped and restored with the same lifecycle as CPU/memory guard. Use it when the goal includes resource guard behavior. Targeted guarded sweeps use paths such as `results/reports/auto_recommendation_gpu.json` and `.autotuneai/recommendations/latest_guarded_gpu.json`; `--target auto` keeps the legacy `auto_recommendation.json` and `latest.json` paths.

Open `results/reports/auto_recommendation.html` to see the current baseline, the recommended configuration, the decision/noise-band interpretation, per-candidate throughput, GPU throughput, step p95/p99 latency, CPU peaks, and deltas.

For daily development, run the fast suite first:

```bash
python scripts/run_tests.py --fast
```

Before release or after executor/lifecycle changes, run the full suite:

```bash
python scripts/run_tests.py --all
python -m unittest discover -s tests
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
