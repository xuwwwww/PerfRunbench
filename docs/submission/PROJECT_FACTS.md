# PerfRunbench Competition Submission Facts

This file is the source of truth for the competition submission package.
Use these facts consistently across Devpost copy, screenshots, demo narration,
charts, and release notes.

## Project Identity

- Project title: `PerfRunbench`
- Version: `0.2.0`
- Product definition: PerfRunbench is a guarded performance autotuning system.
- Repository URL: `https://github.com/xuwwwww/PerfRunbench`
- GitHub Release URL: `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.2.0`
- Related project: `https://github.com/xuwwwww/BootTraceBench` is external
  experimental firmware-log work and is not part of the default tuning path.

ArmTune is the Arm inference subsystem inside PerfRunbench. It should not be
presented as a separate product competing with PerfRunbench.

## One-Line Description

One-command, guarded performance autotuning for Arm inference and PyTorch
training workloads.

## Product Narrative

A user wraps an existing workload with one command. PerfRunbench runs bounded
tuning slices, evaluates declared configuration candidates, statistically
confirms the best observed eligible configuration, caches the recommendation,
and safely reuses it with environment validation, canary execution,
process-scoped apply, and rollback.

## Core Commands

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

First run:

```text
cache miss
-> bounded tuning
-> fresh confirmation
-> recommendation
-> canary
-> full workload
```

Later runs:

```text
cache lookup
-> environment validation
-> canary when required
-> reuse verified configuration
-> full workload
```

## Supported Platforms Validated For v0.2.0

- Linux Arm64 C7g / Neoverse-V1 class target.
- Linux Arm64 T4g / Neoverse-N1 class low-resource target.
- Windows x86_64 controller and local CPU target.
- GitHub Actions release validation matrix:
  - `windows-latest`
  - `ubuntu-latest`
  - `ubuntu-24.04-arm`

Do not compare throughput numbers across C7g, T4g, and Windows. These are
different targets and validation protocols.

## Supported Adapters In This Submission

- `pytorch-training`: bounded synthetic CPU training slices and guarded cache
  reuse.
- `llama.cpp`: ArmTune inference build-profile and runtime tuning subsystem.
- generic command workloads through existing PerfRunbench command wrappers.

## Core Differentiators

- One-command wrapping of existing workloads.
- Architecture-aware tuning.
- Bounded search instead of Cartesian explosion.
- Preservation of training semantics.
- Statistical confirmation.
- Environment-bound recommendation cache.
- Safe process-scoped apply.
- Canary, lock, lease, and rollback.
- Auditable Performance Passport.
- Windows x86_64 and Linux Arm64 validation.

## Measured Results To Use

Only use these measured results. Keep each protocol separate.

### C7g llama.cpp Build-Profile Comparison

| Profile | Decode median |
| --- | ---: |
| `portable-baseline` | `10.825 tokens/s` |
| `arm-native` | `17.530 tokens/s` |
| `arm-optimized` | `17.380 tokens/s` |

### C7g llama.cpp Runtime Search

| Item | Decode median |
| --- | ---: |
| baseline | `13.630 tokens/s` |
| selected | `17.480 tokens/s` |
| observed relative improvement | `28.246515%` |

### C7g llama-server TTFT

| Profile | TTFT median |
| --- | ---: |
| `portable-baseline` | `97.260 ms` |
| `arm-native` | `62.826 ms` |

### C7g Synthetic PyTorch Training

| Item | Median |
| --- | ---: |
| baseline | `77859.454508 samples/s` |
| selected | `203215.038822 samples/s` |
| observed relative improvement | `161.002392%` |

### T4g Synthetic PyTorch Training

| Item | Median |
| --- | ---: |
| baseline | `13796.607073 samples/s` |
| selected | `28037.469976 samples/s` |
| observed relative improvement | `103.220037%` |

### Windows x86 Synthetic PyTorch Paired Confirmation

| Item | Median |
| --- | ---: |
| baseline | `41636.929876 samples/s` |
| selected | `83556.804262 samples/s` |

## Evidence Documents

- `docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`
- `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`
- `docs/results/PHASE_14B_REAL_ARM_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15A_T4G_LOW_RESOURCE_VALIDATION.md`
- `docs/results/PHASE_15B_T4G_REAL_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15C_WINDOWS_X86_PYTORCH_VALIDATION.md`
- `docs/results/VALIDATION_MATRIX.md`
- `docs/releases/v0.2.0.md`

## Approved Result Wording

Use this wording for recommendations:

```text
Best observed eligible configuration under the tested target, workload,
runtime, and measurement protocol.
```

For Arm inference:

```text
Best observed eligible configuration under the tested Arm target, model,
inference workload, and measurement protocol.
```

## Required Claim Boundaries

Do not:

- add improvements across protocols;
- average improvements across machines;
- compare samples/s with tokens/s;
- rank C7g, T4g, and Windows by throughput;
- claim that all PyTorch workloads improve by the measured synthetic amounts;
- describe synthetic results as production training performance;
- present fixture or CI smoke output as production Arm performance;
- imply global optimality.

## Release Artifacts

GitHub Release assets:

- `perfrunbench-0.2.0-py3-none-any.whl`
- `perfrunbench-0.2.0.tar.gz`

Verified SHA-256 digests:

```text
perfrunbench-0.2.0-py3-none-any.whl
d1f4b01bc4251c13700f36deeae5deef982c778c33b2c8408a2883dc22e18e21

perfrunbench-0.2.0.tar.gz
da4e14021ef8daa8f214c575827fe9a07050685a89bb54f23746892ad7d0ee37
```
