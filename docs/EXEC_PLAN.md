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
