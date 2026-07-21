# ArmTune Execution Plan

## Current Status

PerfRunbench ArmTune v0.1.0 has been merged to `main` and preserved with the
tag:

```text
v0.1.0-armtune
```

Current development is Phase 13: Golden-path CLI and Recommendation Cache.
The focus is CLI usability for the generic PerfRunbench workflow:

```text
autotuneai run --auto --config autotune.yaml -- python train.py
```

PerfRunbench remains focused on reproducible benchmarking, autotuning,
statistical confirmation, Performance Passport evidence, and guarded
recommendation apply. BootTraceBench is external and experimental.

## Default Product Flow

```text
inspect Arm target
-> inspect workload runtime
-> benchmark portable baseline
-> build Arm profiles
-> search bounded runtime candidates
-> confirm the selected candidate
-> generate and verify a Performance Passport
-> validate guarded recommendation apply
```

`llama.cpp` remains the first workload adapter. ArmTune is not defined by a
single model, a single benchmark binary, or a workload-specific semantic suite.

## Completed Milestones

### Phase 0: Governance And Architecture

- Audited pre-challenge baseline:
  `bd7d4bc1f728482aa1d923d59e6a99a604be79f2`.
- Documented reuse map, architecture boundaries, benchmark integrity rules,
  and before/after tracking.

### Phase 1: Foundations

- Added `autotuneai arm doctor` and `autotuneai arm inspect-host`.
- Added local and SSH target abstractions.
- Added versioned worker protocol, Arm platform inspection, and schema
  validation.
- Kept the controller valid on x86_64 while requiring production benchmarks on
  real Linux Arm64 targets.

### Phase 2: llama-bench Baseline And Native Arm64 CI

- Added the fixed `llama-bench` benchmark path, defensive parser, provenance,
  and artifact validation.
- Validated fixture execution on a GitHub-hosted native Linux Arm64 runner.
- Preserved the boundary that fake workload values remain `test_fixture`, not
  measured Arm performance.

### Phase 3: Repeated Search And Server Runtime

- Added repeated `llama-bench` measurement, bounded staged runtime search,
  candidate caps, affinity allowlists, paired baseline insertion, confirmation
  runs, and noise-band classification.
- Added `llama-server` lifecycle, target-local loopback HTTP measurement,
  streaming TTFT, end-to-end latency, generated-output preservation, cleanup,
  and port-release evidence.

### Phase 9: First Real Arm Build And Runtime Tuning

- Used a user-provisioned C7g / Graviton3 Linux Arm64 target with Neoverse-V1.
- Used Qwen2.5-7B Q4_K_M and a pinned llama.cpp source revision.
- Built and inspected `portable-baseline`, `arm-native`, and `arm-optimized`.
- Ran fixed `llama-bench` profile comparison.
- Ran bounded runtime search and paired confirmation.
- Ran secondary `llama-server` timing.
- Generated and verified a Performance Passport.

Key measured values:

- `portable-baseline` decode median: `10.825 tok/s`
- `arm-native` decode median: `17.530 tok/s`
- `arm-optimized` decode median: `17.380 tok/s`
- runtime-search baseline decode median: `13.630 tok/s`
- selected runtime-search decode median: `17.480 tok/s`
- runtime-search observed improvement: `28.246515%`
- `portable-baseline` `llama-server` TTFT median: `97.260 ms`
- `arm-native` `llama-server` TTFT median: `62.826 ms`
- Passport: `eligible_preliminary`, verification passed

These protocol-specific results must not be combined into one aggregate
speedup.

### Phase 10: Guarded Recommendation Apply

- Added RecommendationBundle generation, validation, environment drift checks,
  dry-run apply, process-scoped canary, lease, status, and rollback.
- Real guarded apply validation passed for bundle
  `92b0eb69b189a06baae62f15`.
- The validation is apply-safety evidence, not a new benchmark result.

### Phase 11: Release Packaging

- Rewrote public README, Devpost draft, demo script, before/after summary,
  architecture, limitations, and release safety tests around the default
  ArmTune product flow.
- Added public placeholder-only configs and E2E release validation.

## Phase 12: BootTrace Repository Extraction

Status: completed and published.

Goal:

```text
PerfRunbench
  keeps generic benchmark, correctness guard, recommendation, and apply safety

BootTraceBench
  owns firmware-log cases, prompt contracts, semantic validators, and model
  capability experiment history
```

Phase 12 does not add new model tests, new prompts, new benchmark backends,
cloud automation, or performance tuning features.

Required outputs:

- `https://github.com/xuwwwww/BootTraceBench` as a standalone public
  repository.
- PerfRunbench branch `refactor/split-boottrace`, merged to `main` by PR #2.
- `docs/refactors/BOOTTRACE_EXTRACTION_PLAN.md`.
- `docs/refactors/BOOTTRACE_MIGRATION.md`.
- `docs/refactors/BOOTTRACE_EXTRACTION_VALIDATION.md`.
- BootTraceBench `docs/RELEASE_VALIDATION.md`.
- BootTraceBench commits:
  - `chore: initialize standalone BootTraceBench`
  - `feat: migrate BootTrace evaluation suite`
  - `docs: document BootTraceBench extraction and validation`
- PerfRunbench commits:
  - `refactor: remove BootTrace-specific workload`
  - `docs: document external BootTraceBench migration`
  - `test: validate PerfRunbench after BootTrace extraction`

## Phase 13: Golden-Path CLI And Recommendation Cache

Status: in progress.

Goals:

- Add the default public workflow:
  `autotuneai run --auto --config autotune.yaml -- <command>`.
- Add `autotuneai tune --config <config> -- <command>`.
- Add `autotuneai run --best --config <config> -- <command>`.
- Preserve existing advanced commands and the legacy `autotuneai run --`
  resource wrapper.
- Add a versioned command-tuning config schema with allowlisted metric
  sources.
- Add WorkloadIdentity fingerprints that hash executable, script, selected
  config files, and explicit identity files without hashing whole directories.
- Add RecommendationCache lookup, insert, list, show, invalidate, and gc.
- Support cache match states:
  `exact_match`, `soft_drift`, `hard_drift`, `expired`, `revoked`,
  `missing`, `corrupt`, and `ambiguous`.
- Implement `retune_policy` and `canary_policy` for `run --auto`.
- Keep recommendation apply process-scoped by default.

Non-goals:

- No AWS access.
- No model downloads.
- No real llama.cpp workload execution.
- No new BootTrace prompts, model qualification, or UEFI-specific work.
- No new benchmark backend or search algorithm.

## Current Claim Boundary

The allowed public recommendation wording remains:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

Do not claim global optimality, cross-instance portability, private workload
coverage, automatic cloud provisioning, or optimized-path usage without
recorded build and runtime evidence.

## Phase 19: CPU-Aware Native Build Autotuning

Status: in progress on `feature/cpu-aware-build-autotuning`.

Phase 19 makes CPU-specific compilation a first-class ArmTune dimension while
preserving historical Phase 9 evidence as a separate protocol.

Goals:

- add CPU target, backend, optimization, and portability fields to build
  profile evidence
- support `upstream-default`, `portable-reference`, `native-current-cpu`,
  `explicit-current-cpu`, `native-kleidiai`, and `native-kleidiai-lto`
  profile IDs
- require KleidiAI build and runtime backend evidence before
  `enabled_and_evidenced`
- keep decode and prefill comparison metrics separate
- add native-binary portability guards independent from workload identity
- validate the workflow on one low-resource T4g.small target without comparing
  the result to C7g evidence

Allowed Phase 19 wording:

> Best observed eligible configuration under the tested CPU, model, runtime
> source commit, compiler, and measurement protocol.

Non-goals:

- no change to `v0.2.0` release/tag
- no BootTrace prompt or model qualification work
- no new model downloads
- no C7g access
- no global Arm speedup claim
