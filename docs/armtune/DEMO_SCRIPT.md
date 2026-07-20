# PerfRunbench ArmTune Demo Script

This demo shows the default release workflow. BootTrace quality artifacts may
be mentioned as optional experimental evidence, but they do not block or drive
the default build/runtime tuning recommendation.

## Pre-Demo Requirements

- PerfRunbench is installed in an isolated Python environment.
- Public placeholder configs are visible under `configs/armtune/`.
- Any real target config, SSH identity, model path, and raw private run
  artifacts remain outside git.
- The presenter uses measured Phase 9 and Phase 10 documentation for result
  claims, not live fixture numbers.

## Storyboard

### 0:00-0:25 - Product Identity

Show `README.md`.

Say:

```text
PerfRunbench is the reproducible benchmarking and tuning toolkit. ArmTune is
its Arm AI inference subsystem. The workflow is inspect, benchmark, build,
search, confirm, passport, and guarded apply.
```

### 0:25-0:55 - Arm Target Inspection

Show:

```bash
autotuneai arm doctor --target-config configs/armtune/target_ssh.example.json --require-production
autotuneai arm inspect-host --target-config configs/armtune/target_ssh.example.json
```

Highlight:

- Controller may be x86.
- Production benchmark execution must validate as Linux Arm64.
- Host capability evidence is separate from workload execution evidence.

### 0:55-1:25 - Reproducible Baseline

Show:

```bash
autotuneai arm benchmark --config configs/armtune/llama_bench.example.json
```

Highlight:

- Raw stdout and stderr are preserved.
- Parser failures remain failures, not zero metrics.
- Model, binary, target, and protocol digests are compatibility boundaries.

### 1:25-1:55 - Build Profiles

Show:

```bash
autotuneai arm build-plan --config configs/armtune/build_profiles.example.json
autotuneai arm build-profile --config configs/armtune/build_profiles.example.json
```

Highlight:

- Profiles are `portable-baseline`, `arm-native`, and `arm-optimized`.
- `arm-optimized` needs build/runtime evidence before optimized-path claims.
- The measured Phase 9 fixed `llama-bench` profile comparison was:

```text
portable-baseline decode median: 10.825 tok/s
arm-native decode median: 17.530 tok/s
arm-optimized decode median: 17.380 tok/s
```

### 1:55-2:20 - Runtime Search

Show:

```bash
autotuneai arm optimize \
  --config configs/armtune/llama_bench_optimize.example.json \
  --max-candidates 12 \
  --affinity-modes scheduler-default,compact,spread,reserve-one-core
```

Highlight:

```text
runtime-search baseline decode: 13.630 tok/s
selected decode: 17.480 tok/s
observed runtime-search improvement: 28.246515%
```

This is a separate `llama-bench` staged-search protocol and must not be
combined with build-profile or server latency deltas.

### 2:20-2:40 - Secondary Server Timing

Show the Phase 9 evidence summary:

```text
portable llama-server TTFT median: 97.260 ms
arm-native llama-server TTFT median: 62.826 ms
```

Highlight request completion, nonempty output, streaming parse, TTFT, E2E
latency, cleanup, and port release. BootTrace semantic correctness is not part
of default candidate eligibility.

### 2:40-3:10 - Passport And Guarded Apply

Show:

```bash
autotuneai arm passport --comparison-result .autotuneai/runs/<run_id>/armtune/profile_comparison/profile_comparison_result.json
autotuneai arm verify-passport --passport .autotuneai/runs/<run_id>/armtune/passport/performance_passport.json
autotuneai recommendation show --bundle .autotuneai/recommendations/<bundle_id>.json
autotuneai recommendation validate --bundle .autotuneai/recommendations/<bundle_id>.json
autotuneai recommendation apply --bundle .autotuneai/recommendations/<bundle_id>.json --scope process --dry-run
autotuneai recommendation rollback --bundle .autotuneai/recommendations/<bundle_id>.json
```

Highlight:

```text
Performance Passport: eligible_preliminary / verification passed
guarded apply validation: passed
```

## Demo Integrity Checklist

- Do not show placeholder configs as private target details.
- Do not combine improvements from different protocols.
- Do not claim global optimality.
- Do not claim KleidiAI from a requested flag alone.
- Do not present BootTrace as the default performance gate.
- Do not reveal endpoint details, identity-file paths, tokens, model weights,
  or private artifact locations.
