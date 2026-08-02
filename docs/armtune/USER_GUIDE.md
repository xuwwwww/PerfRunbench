# PerfRunbench And ArmTune Operator Guide

PerfRunbench has one guided product entrance. Users choose a workload adapter,
not an Arm or x86 edition. ArmTune activates when the selected target and
inference workflow need Arm-specific inspection, search, and evidence.

Start on any controller with:

```bash
autotuneai guide
```

The default Arm inference experience remains intentionally smaller than the
complete research command surface:

```text
initialize a private project
-> validate a Linux Arm64 target
-> inspect CPU and topology
-> establish a fixed baseline
-> run bounded search
-> confirm outside the noise band
-> review the Performance Passport
-> apply only after validation and canary checks
```

The controller may run on Windows, Linux, macOS, x86-64, or Arm64. A production
Arm performance result must come from a real Linux Arm64 target.

## 1. Create A Starter Project

```bash
autotuneai init \
  --adapter llama-bench \
  --output .autotuneai/perfrunbench-project \
  --target ssh \
  --preset quick
```

Generated files:

| file | purpose |
| --- | --- |
| `armtune-project.json` | Versioned project manifest and preset selection. |
| `target.json` | Local or SSH target definition. |
| `benchmark.json` | Fixed baseline protocol. |
| `optimize.json` | Bounded candidate search and confirmation protocol. |

The generated files contain angle-bracket placeholders and
`result_origin: unverified`. They do not contain credentials and they do not
claim measured performance. Keep the edited copies outside public source
control because host, identity-file, binary, and model paths are private
deployment details.

## 2. Understand The Cost Preset

ArmTune presets bound candidate count and repetition cost. They never weaken
identity, parser, cleanup, or provenance guards.

| preset | maximum candidates | search repetitions | confirmation repetitions | noise band | intended use |
| --- | ---: | ---: | ---: | ---: | --- |
| `quick` | 4 | 1 | 3 | 5% | Validate the complete workflow at low cost. |
| `standard` | 8 | 3 | 5 | 5% | Compare a small set with repeated measurements. |
| `deep` | 12 | 5 | 10 | 2% | Revisit an already stable and qualified target. |

Candidate limits count unique configurations. Warm-up, paired baseline, and
confirmation executions are additional measurements.

## 3. Check Configuration Readiness

```bash
autotuneai guide --project .autotuneai/perfrunbench-project
```

Possible readiness states:

| state | meaning |
| --- | --- |
| `not_initialized` | One or more managed starter files are missing. |
| `invalid_configuration` | A managed file is not valid JSON. |
| `needs_configuration` | Placeholders remain; target access is not ready. |
| `ready_for_doctor` | No placeholders remain; run target diagnostics next. |

`guide` is read-only. It does not connect to SSH, launch a benchmark, build a
runtime, or apply a recommendation.

## 4. Validate And Inspect The Target

```bash
autotuneai doctor \
  --target-config .autotuneai/perfrunbench-project/target.json \
  --require-production \
  --summary

autotuneai inference inspect-host \
  --target-config .autotuneai/perfrunbench-project/target.json \
  --summary
```

Expected production boundary:

```text
system: Linux
architecture: aarch64
production_benchmark_ready: true
workload_execution_evidence: not_collected
```

The last line is intentional. Host capabilities do not prove that an optimized
workload path executed.

## 5. Establish A Fixed Baseline

```bash
autotuneai inference benchmark \
  --config .autotuneai/perfrunbench-project/benchmark.json
```

The fixed baseline records target identity, model and executable digests,
protocol settings, raw stdout/stderr, parser status, and metric distributions.
A parser failure remains a failure; it is never converted to a zero metric.

## 6. Run Bounded Optimization

```bash
autotuneai inference optimize \
  --config .autotuneai/perfrunbench-project/optimize.json
```

The generated search uses deterministic order, warm-up exclusion, paired
baseline insertion, bounded affinity modes, confirmation runs, cooldown, and a
noise band. The result may be `recommended` or `inconclusive`. An inconclusive
result is valid when no tested candidate clears the configured evidence guard.

`autotuneai arm ...` remains a backward-compatible alias for this inference
command surface. New operator instructions use `autotuneai inference ...` so
the namespace describes the workload instead of the controller architecture.

## 7. Understand The Workload Before Reading A Chart

```bash
autotuneai explain
autotuneai explain prefill
autotuneai explain decode-heavy --json
```

The short version:

- `prefill` reads the prompt before the first output token.
- `decode` generates output tokens one by one.
- `interactive-short` prioritizes responsive first-token and end-to-end
  latency.
- `decode-heavy` stresses sustained generation throughput.
- `prefill-heavy` stresses long-prompt ingestion and queueing.

The same candidate may be eligible for one workload and rejected for another.
See `GLOSSARY.md` for metric and guard definitions.

## 8. Review Evidence And Apply Safely

The advanced workflow can compare build profiles, generate and verify a
Performance Passport, and create a guarded recommendation:

```bash
autotuneai arm compare-builds --config <private-profile-comparison.json>
autotuneai arm passport --comparison-result <profile_comparison_result.json>
autotuneai arm verify-passport --passport <performance_passport.json>
autotuneai recommendation validate --bundle <recommendation_bundle.json>
autotuneai recommendation apply --bundle <recommendation_bundle.json> --scope process --dry-run
```

Use the explicit build and apply commands only after the baseline workflow is
understood. System- and kernel-scoped tuning remains advanced opt-in and is not
part of the guided default.

## Result Language

Use:

> Best observed eligible configuration under the tested Arm target, model,
> workload, runtime, concurrency, topology, and measurement protocol.

Do not claim global optimality or combine percentages from different
measurement protocols.
