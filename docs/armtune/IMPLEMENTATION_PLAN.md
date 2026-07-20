# ArmTune Implementation Plan

## Principles

- Preserve all existing PerfRunbench commands and behavior.
- Reuse PerfRunbench infrastructure for runs, monitoring, resource controls,
  comparison, and reporting.
- Keep Arm-specific logic inside `autotune.arm_tune`.
- Make every serialized top-level artifact versioned.
- Keep host capabilities separate from verified workload execution evidence.
- Mark unknown, unverified, fixture, synthetic, and measured data explicitly.
- Reject invalid production Arm benchmark setups rather than silently falling
  back to controller-side execution.

## Phase 1: Foundations

Planned scope:

- Typed schema dataclasses for target specs, command results, host info, CPU
  identity, CPU features, topology, and evidence records.
- Versioned configuration loader and validation.
- `LocalArmTarget` for real local Linux Arm64 inspection.
- `SshArmTarget` protocol skeleton using system OpenSSH.
- Internal worker command contract.
- Host inspection for architecture, OS, kernel, `/proc/cpuinfo`, and
  `lscpu --json` fixtures.
- CLI commands: `arm doctor`, `arm inspect-host`.
- Unit tests that require no Arm machine, network, root, real model, or
  Performix.

Review boundary:

- No llama.cpp execution.
- No production benchmark execution.
- No artifact download beyond target inspection outputs.

## Phase 2: llama.cpp Baseline

Planned scope:

- Build and runtime inspection for existing llama.cpp binaries.
- Parser support for `llama-bench`, `llama-cli`, and `llama-server` output.
- Baseline benchmark runner using fake executables in tests.
- Raw stdout/stderr artifact preservation.
- CLI commands: `arm inspect-build`, `arm benchmark`.

## Phase 3: Search and Quality

Planned scope:

- Arm-aware candidate generation for threads, batch, micro-batch, concurrency,
  and affinity modes supported by the runtime.
- Staged search and paired baselines.
- Metric distributions, validity checks, objective functions, and Pareto
  frontier.
- BootTrace synthetic/public prompt-suite fixtures.
- Deterministic quality validators.
- CLI command: `arm optimize`.

## Phase 4: Passport, Report, CI Gate

Planned scope:

- Versioned Arm Performance Passport.
- Passport reader, writer, comparison, and migration foundation.
- HTML report and Markdown summary.
- CI regression gate with stable exit codes.
- Reference demo workflow.
- CLI commands: `arm passport`, `arm compare`, `arm ci-check`, `arm demo`.

## Phase 5: Build Profiles and Performix

Planned scope:

- Portable Arm64, native Arm64, and verified Arm-optimized build profiles.
- Safe build workflow and build-profile comparison.
- Optional Arm Performix integration.
- Baseline and recommended profiling evidence.

## Phase 6: Real Arm Validation and Submission

This phase requires a real Linux Arm64 target supplied or approved by the user.

Planned scope:

- Real target validation.
- Real llama.cpp baseline.
- Search and confirmation runs.
- Quality validation.
- Optional Performix run if available.
- Result audit, public README update, Devpost draft, demo script, and
  three-minute demo outline.

No measured Arm claims should be published before this phase is executed.
