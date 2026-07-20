# BootTrace Migration

## Summary

After the `v0.1.0-armtune` release, firmware-log model capability evaluation
was split out of PerfRunbench into a standalone local repository:

```text
<local-boottracebench-repo>
```

PerfRunbench current main keeps Arm platform inspection, reproducible
benchmarking, build/runtime autotuning, statistical confirmation, Performance
Passport evidence, RecommendationBundle safety, and generic correctness hooks.

## What Moved

BootTraceBench owns workload-specific material:

- firmware-log case suites
- development and evaluation suites
- prompt-contract rendering
- canonical expected answers
- firmware-log semantic validators
- single-fence normalization diagnostics for that workload
- fixture responses
- historical sanitized model-capability experiment summaries

## What Stayed

PerfRunbench keeps reusable infrastructure:

- target inspection
- SSH worker protocol
- process execution and cleanup
- `llama-bench` and `llama-server` benchmark adapters
- generic quality/result schemas
- optional workload-neutral correctness guards
- artifact manifests and digest validation
- build-profile evidence
- runtime search
- Performance Passport
- RecommendationBundle validation, dry-run apply, canary, lease, status, and
  rollback

## Compatibility

Historical v0.1.0 ArmTune artifacts may contain optional quality fields from
earlier experiments. Current PerfRunbench should parse or display those fields
when practical, but it does not upgrade historical workload-specific semantic
evidence into default recommendation eligibility.

The prior integrated state is preserved by:

```text
v0.1.0-armtune
```

Current main does not ship the removed firmware-log suites or CLI calibration
command.

## Future Integration

PerfRunbench can later execute BootTraceBench as an external command workload
or consume its JSON result artifacts. The repositories must remain decoupled:

- PerfRunbench must not import BootTraceBench.
- BootTraceBench must not import PerfRunbench internals.
- Shared data should cross the boundary through versioned JSON artifacts.

## Recommendation Eligibility

Default ArmTune candidate eligibility is performance/provenance based:

- measured Linux Arm64 provenance
- matching target/model/protocol digests
- valid build evidence
- successful execution and parsing
- cleanup passed
- sufficient samples
- improvement outside the configured noise band

External semantic workloads may add separate evidence, but they are not part
of the default performance recommendation path unless a future adapter
explicitly defines that policy.
