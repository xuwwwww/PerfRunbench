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

## Next Steps

Keep feature expansion stopped unless a release-blocking correctness issue is
found. Submission work should focus on documentation, release notes, PR review,
and CI verification.
