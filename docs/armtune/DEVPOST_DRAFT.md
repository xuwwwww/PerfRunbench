# PerfRunbench ArmTune Devpost Draft

## Project Name

PerfRunbench ArmTune

## Tagline

Reproducible Arm AI inference benchmarking, autotuning, statistical
confirmation, and audited recommendation apply.

## Track

Cloud AI

## Overview

PerfRunbench is a reproducible AI systems benchmarking and tuning toolkit.
ArmTune is its Arm AI inference subsystem. It helps a developer inspect a real
Linux Arm64 target, benchmark a fixed AI inference workload, compare portable
and Arm-specific build profiles, run bounded runtime search, confirm whether a
candidate beats noise, generate a Performance Passport, and validate a guarded
recommendation before applying it.

`llama.cpp` is the first workload adapter. The product identity is the evidence
pipeline: platform inspection, reproducible benchmarking, build/runtime
autotuning, statistical comparison, Performance Passport, and recommendation
safety.

The public UX starts with three discoverable commands:

```bash
autotuneai guide
autotuneai init --adapter llama-bench --target ssh --output .autotuneai/perfrunbench-project --preset quick
autotuneai guide --project .autotuneai/perfrunbench-project
autotuneai explain decode-heavy
```

The generated project is versioned, bounded, strict-SSH by default, and marked
unverified until the user supplies private target and workload details.

## Problem

AI inference can run on Arm servers, but performance claims are often hard to
audit. A useful optimization tool must answer:

- What Arm target was measured?
- Which binary, model, build flags, and runtime protocol were used?
- Were raw outputs and artifacts preserved?
- Was the observed difference larger than the configured noise band?
- Is the recommendation safe to reuse on the current environment?

PerfRunbench ArmTune is built around these evidence boundaries instead of
around a single benchmark number.

## Implemented Pipeline

- Controller and target workflow for local or SSH Linux Arm64 targets.
- Arm host identity, feature, topology, and production-readiness inspection.
- Versioned SSH worker protocol using system OpenSSH and strict host-key
  checking by default.
- `llama-bench` baseline and bounded runtime search.
- Declarative build-profile evidence for `portable-baseline`, `arm-native`,
  and `arm-optimized`.
- KleidiAI evidence states that distinguish unavailable, requested but
  unverified, runtime unverified, build failed, and enabled with evidence.
- Repeated measurements, paired baseline insertion, confirmation runs,
  cooldown, and noise-band classification.
- `llama-server` secondary runtime measurements for request completion,
  nonempty output, streaming parse, TTFT, E2E latency, cleanup, and port
  release.
- Performance Passport JSON/Markdown generation and verification.
- RecommendationBundle show, validate, dry-run apply, process canary, lease,
  rollback, and status commands.
- Optional experimental firmware-log capability workload preserved separately
  from the default performance-tuning eligibility path.

## Real Results

The measured Phase 9 result used one user-provisioned Linux Arm64 C7g /
Neoverse-V1 target, Qwen2.5-7B Q4_K_M, and a pinned llama.cpp source commit.
The numbers below are preliminary and protocol-specific.

`llama-bench` build-profile comparison:

| profile | decode median |
| --- | ---: |
| `portable-baseline` | `10.825 tok/s` |
| `arm-native` | `17.530 tok/s` |
| `arm-optimized` | `17.380 tok/s` |

`llama-bench` bounded runtime search:

| item | decode median |
| --- | ---: |
| search baseline | `13.630 tok/s` |
| selected candidate | `17.480 tok/s` |
| observed runtime-search improvement | `28.246515%` |

`llama-server` secondary latency evidence:

| profile | TTFT median |
| --- | ---: |
| `portable-baseline` | `97.260 ms` |
| `arm-native` | `62.826 ms` |

The Phase 9 Performance Passport was `eligible_preliminary` and verification
passed. Phase 10 guarded apply validation also passed for the generated
RecommendationBundle.

These results must not be combined across protocols. The supported claim is:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

## How To Read The Workloads

- `interactive-short`: short prompt and short-to-medium answer; responsiveness,
  p95 TTFT, E2E latency, and output throughput matter.
- `decode-heavy`: short prompt and long answer; sustained token generation is
  the dominant workload phase.
- `prefill-heavy`: long prompt and short answer; prompt ingestion, memory
  traffic, queueing, and TTFT dominate.
- `TTFT`: time from request submission to the first generated token.
- `p95`: the value at or below which 95 percent of measured samples fall.

This distinction is operational, not cosmetic. In the same C7g size-scaling
protocol, four-slot continuous batching was eligible for interactive and
decode traffic but regressed prefill-heavy screening on all three sizes. ArmTune
preserved the negative evidence and did not create a universal recommendation.

## What Is New During The Competition Window

ArmTune adds Arm target validation, SSH worker execution, host inspection,
llama.cpp workload adapters, build-profile evidence, bounded runtime search,
server timing, Performance Passport generation, Passport verification, and
guarded recommendation apply on top of the pre-existing PerfRunbench execution
and reporting infrastructure.

## Demo Flow

1. Start with `autotuneai guide`, create a workload-specific project with
   `autotuneai init`, and inspect its readiness with the same guided entrance.
2. Explain the traffic shape with `autotuneai explain`.
3. Validate the Arm target with the unified `autotuneai doctor` and inspect it with
   `autotuneai inference inspect-host`.
4. Run a placeholder `llama-bench` baseline.
5. Generate build-profile evidence for portable and Arm profiles.
6. Run bounded runtime search and show the statistical guard result.
7. Generate and verify the Performance Passport.
8. Show, validate, dry-run apply, and rollback a guarded RecommendationBundle.

## Claim Boundaries

- The real result is preliminary and single-target.
- It does not prove global optimality or cross-instance portability.
- `llama-bench` throughput and `llama-server` latency improvements are
  separate protocols.
- External semantic workloads are not the default tuning gate.
- Model weights, private target configs, endpoint details, SSH keys, and
  private run artifacts are not committed.

## Future Work

- More workload adapters beyond llama.cpp.
- Optional Arm Performix evidence when available.
- Multi-target regression comparisons.
- Stronger release automation around public artifact packaging.
