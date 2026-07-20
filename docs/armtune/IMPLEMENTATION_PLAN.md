# ArmTune Implementation Plan

## Principles

- Preserve existing PerfRunbench commands and behavior.
- Keep Arm-specific orchestration inside `autotune.arm_tune`.
- Reuse PerfRunbench run storage, process execution, resource controls,
  comparison, reporting, recommendation, and rollback infrastructure.
- Version every serialized top-level artifact.
- Keep host capabilities separate from verified workload execution evidence.
- Mark unknown, unverified, fixture, synthetic, and measured data explicitly.
- Require real Linux Arm64 provenance before reporting production Arm results.
- Avoid `shell=True` and arbitrary remote shell strings.
- Do not store credentials, private target configs, identity-file contents, or
  model weights in public artifacts.

## Implemented Product Surface

ArmTune now supports:

- target inspection through local and SSH targets
- versioned controller/worker protocol
- `llama-bench` inspection, execution, parsing, provenance, and artifacts
- `llama-server` lifecycle, loopback HTTP timing, stream parsing, cleanup, and
  artifacts
- Arm-aware build-profile evidence and comparison
- bounded staged runtime search
- paired baseline insertion, confirmation runs, and noise-band classification
- Performance Passport generation and verification
- RecommendationBundle generation, validation, process-scoped dry-run apply,
  canary, lease, status, and rollback

Default flow:

```text
inspect Arm target
-> inspect workload runtime
-> benchmark portable baseline
-> build Arm profiles
-> search runtime candidates
-> confirm the selected candidate
-> generate and verify a Performance Passport
-> validate guarded recommendation apply
```

## Current Refactor Scope

Phase 12 removes firmware-log model capability evaluation from PerfRunbench and
migrates it to BootTraceBench. PerfRunbench keeps only workload-neutral
correctness and guard infrastructure.

The migration must not:

- change Phase 9 measured results
- change Phase 10 guarded apply evidence
- weaken recommendation safety
- add new prompts, models, benchmark backends, or search algorithms
- create a dependency from PerfRunbench to BootTraceBench
- require BootTraceBench to import PerfRunbench internals

## Future Work

Possible post-release work should be evaluated separately:

- generic external command workload adapter polish
- additional non-firmware correctness contracts
- Performix evidence integration
- broader Arm instance coverage
- more workload adapters after the llama.cpp vertical slice remains stable
- Android or edge proof-of-concepts

None of these are required for the Phase 12 split.
