# PerfRunbench Quick Start

PerfRunbench helps users inspect a workload environment, run reproducible
benchmarks, search bounded runtime/build configurations, confirm changes with
statistics, and safely reuse recommendations.

PerfRunbench is not published to PyPI in this repository state. Install from a
source checkout or a locally built wheel.

## Install From Source

```bash
python -m pip install .
autotuneai --help
```

For development:

```bash
python -m pip install -e .
python scripts/run_tests.py --fast
```

## Fresh Training Workflow

Create a public placeholder config:

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
```

Edit only the fields that describe your workload:

- `workload.cwd`
- `workload.identity_files`
- `objective.metric`
- `metrics`
- `training.slice`
- `training.knobs`
- `tuning`

Check readiness without running the workload:

```bash
autotuneai doctor --config autotune.yaml -- python train.py
```

Run automatically:

```bash
autotuneai run --auto --config autotune.yaml -- python train.py
```

## What Happens On First Run

`run --auto` computes a workload identity and environment fingerprint. If the
RecommendationCache has no eligible match and the retune policy permits tuning,
PerfRunbench runs bounded candidates, performs paired confirmation, writes a
Performance Passport, writes an applyable RecommendationBundle, and launches
the full workload through process-scoped apply.

If tuning is inconclusive, the command fails explicitly. It does not convert
failed candidates into zero metrics and does not apply stale or non-applyable
recommendations.

## Reuse Commands

Tune and cache without launching the full workload afterward:

```bash
autotuneai tune --config autotune.yaml -- python train.py
```

Reuse the best exact or soft-drift cache match without retuning:

```bash
autotuneai run --best --config autotune.yaml -- python train.py
```

List and inspect cache entries:

```bash
autotuneai cache list
autotuneai cache show <cache-id>
autotuneai cache invalidate <cache-id>
autotuneai cache gc
```

## Process-Scoped Apply

PerfRunbench recommendations apply to the launched process by default. The
selected configuration may set argv entries and environment variables. It does
not persist machine-wide settings unless an explicit system-scope workflow is
used.

Advanced recommendation operations:

```bash
autotuneai recommendation show --bundle <recommendation-bundle.json>
autotuneai recommendation validate --bundle <recommendation-bundle.json>
autotuneai recommendation apply --bundle <recommendation-bundle.json> --scope process --dry-run
autotuneai recommendation apply --bundle <recommendation-bundle.json> --scope process --lease-minutes 30 --canary-runs 1
autotuneai recommendation rollback --bundle <recommendation-bundle.json>
```

## Bounded Training Slices

The `pytorch-training` adapter requires bounded slices for tuning. A slice may
be bound through an allowlisted CLI argument, environment variable, or JSON/YAML
config overlay. The full-run launch spec omits the slice control so the user's
long workload is not accidentally limited by tuning settings.

Supported default knob bindings include:

- `micro_batch_size`
- `gradient_accumulation`
- `dataloader_workers`
- `cpu_threads`
- `interop_threads`
- `intraop_threads`
- `amp_mode`
- `compile_mode`

Effective batch size is preserved by default. Candidates that change training
semantics are not eligible for default recommendation.

## Common Corrections

- Missing executable: pass the command after `--`, for example
  `-- python train.py`.
- Missing identity file: create the declared file or update
  `workload.identity_files`.
- Missing metric: make the workload emit the configured metric source.
- Missing bounded slice: configure `training.slice` before tuning.
- Missing cache: use `run --auto` to permit first-run tuning, or run
  `autotuneai tune` first.
- Hard drift: retune on the changed machine, CPU mask, executable, config, or
  runtime identity.

## ArmTune Path

ArmTune is PerfRunbench's Arm AI inference benchmarking and autotuning
subsystem:

```text
inspect Arm target
-> inspect workload runtime
-> benchmark portable baseline
-> build Arm profiles
-> search runtime candidates
-> confirm selected candidate
-> generate Performance Passport
-> validate and guarded-apply recommendation
```

Public ArmTune examples under `configs/armtune/` contain placeholders only.
Keep hostnames, SSH identity files, models, and private target configs outside
the repository.
