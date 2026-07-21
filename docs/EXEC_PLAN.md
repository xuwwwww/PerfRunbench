# PerfRunbench Execution Plan

PerfRunbench's current default workflow is:

```text
inspect environment
-> inspect workload runtime
-> establish a baseline
-> run bounded candidate search
-> confirm with paired measurements
-> generate a Performance Passport
-> validate and safely apply an eligible RecommendationBundle
```

## Completed Validation

- Phase 14B validated the `pytorch-training` adapter on a real Linux Arm64 C7g
  target.
- Phase 15A validated low-resource behavior on a constrained T4g.small target.
- Phase 15B validated real PyTorch tuning on that T4g target without comparing
  numbers to C7g.
- Phase 15C validated the golden-path CLI and `pytorch-training` adapter on
  Windows x86_64, including cache miss, cache reuse, cross-platform hard-drift
  refusal, process-scoped apply, and cleanup.

## Phase 15C Boundary

Phase 15C used a local Windows x86_64 controller only. AWS was not accessed or
controlled. The measured numbers are local synthetic-workload validation data
and are not Arm performance claims.

The only product change made during Phase 15C was a focused correctness fix for
hex CPU-affinity mask canonicalization. No model, dataset, benchmark backend,
cloud automation, or system-scoped tuning feature was added.

## Completed Release Scope

Phase 16 through Phase 18 completed the v0.2 release validation, GitHub
Release packaging, competition submission materials, and final demo rehearsal.
The v0.2.0 tag and competition submission claims must not be modified
retrospectively.

## Phase 19: CPU-Aware Native Build Autotuning On Arm

Phase 19 makes CPU-specific native compilation a first-class tuning dimension
for Arm inference workloads:

```text
inspect Arm target
-> lock a pinned llama.cpp source commit
-> inspect compiler and CPU feature evidence
-> create allowlisted build candidates
-> verify build and backend evidence
-> compare builds with a fixed benchmark protocol
-> run bounded runtime search on the selected build
-> confirm the complete configuration
-> generate and verify a Performance Passport
-> guard native-binary recommendation reuse
```

Boundaries:

- Use a single existing low-resource T4g.small target for bounded validation.
- Do not compare T4g numbers directly with C7g evidence.
- Do not access the stopped paid C7g target.
- Do not download new models or datasets.
- Do not add BootTrace research or model qualification logic.
- Do not claim global Arm speedup from one T4g.small session.

Expected commits:

```text
feat: add CPU-aware native build profiles
feat: guard native binary portability and backend evidence
docs: record T4g CPU-aware build tuning
```

Current Phase 19 real T4g outcome:

```text
insufficient_evidence
```

The target produced build artifacts for `upstream-default` and
`native-current-cpu`, but no suitable existing small GGUF model was available.
The run therefore stopped before benchmark comparison, runtime search,
Performance Passport, or recommendation apply. This is a guard outcome, not a
performance result.

## Phase 20: C7g Build Backend And Runtime Tuning

Phase 20 validates the CPU-aware build-profile implementation on a real
C7g / Neoverse-V1 target with the existing Qwen2.5-7B Q4_K_M model and pinned
`llama.cpp` source `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.

Completed:

- add the explicit `explicit-neoverse-v1` build profile
- build `upstream-default`, `explicit-neoverse-v1`, `native-kleidiai`, and
  `native-kleidiai-lto`
- preserve `portable-reference` as `build_failed`
- reject KleidiAI-requested profiles from eligibility when runtime backend
  evidence is not proven
- run fixed build-profile comparison
- run a six-candidate bounded runtime search
- run secondary `llama-server` TTFT/E2E measurement without BootTrace quality
  eligibility
- generate and verify a Performance Passport
- validate 77 artifact manifests on-target and again after download

Current Phase 20 real C7g outcome:

```text
build_profile_recommendation: inconclusive
runtime_search_recommendation: inconclusive
performance_improvement_verified: false
phase20_c7g_build_backend_runtime_tuning: passed
```

The best evidence-valid build profile was `explicit-neoverse-v1`, but its
decode median (`17.78 tok/s`) was only `0.395257%` above `upstream-default`
(`17.71 tok/s`), inside the configured 5% noise band. Runtime search selected
the baseline configuration with a confirmed decode median of `17.72 tok/s`.
