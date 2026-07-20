# ArmTune Documentation Index

ArmTune is the Arm AI inference benchmarking and autotuning subsystem inside
PerfRunbench. These documents define the architecture, benchmark integrity
rules, evidence records, release packaging, and phase history.

Measured Arm results, examples, fixtures, synthetic outputs, and planned work
must remain clearly labeled. Public examples are placeholders and must not
include private target configuration, credentials, model weights, or endpoint
details.

## Documents

- `EXEC_PLAN.md`: Phase audit record, repository findings, command log, and
  phase-by-phase execution plan.
- `IMPLEMENTATION_PLAN.md`: Reviewable implementation phases and expected code
  surfaces.
- `ARCHITECTURE.md`: Controller/target design, release architecture, and
  package boundaries.
- `REUSE_MAP.md`: Exact existing PerfRunbench symbols planned for ArmTune
  reuse, including required modifications, risks, and tests.
- `ARM_SPECIFICITY.md`: Why ArmTune is Arm-specific and what evidence is
  required before making Arm optimization claims.
- `REMOTE_TARGETS.md`: Local Arm64 Linux and SSH target strategy.
- `BENCHMARK_METHODOLOGY.md`: Measurement rules, comparability checks, and
  result labeling.
- `PERFORMANCE_PASSPORT_SPEC.md`: Planned schema content for the Arm
  Performance Passport artifact.
- `BEFORE_AFTER.md`: Pre-challenge baseline, reused infrastructure, and
  challenge-period change tracking.
- `RISK_REGISTER.md`: Known risks, mitigations, and ownership.
- `LIMITATIONS.md`: Current unsupported features, release limitations, and
  integrity constraints.
- `RESEARCH_NOTES.md`: Source-tracked research notes for competition,
  llama.cpp, KleidiAI, and Arm Performix decisions.
- `RESULTS_TEMPLATE.md`: Placeholder format for future measured, synthetic,
  fixture, and unverified results.
- `DEVPOST_DRAFT.md`: Competition submission narrative draft.
- `DEMO_SCRIPT.md`: Demo outline and required evidence checklist.
- `results/PHASE_9_FIRST_REAL_ARM_TUNING.md`: First preliminary measured Arm
  build/runtime tuning result.
- `results/PHASE_10_GUARDED_APPLY_VALIDATION.md`: Sanitized guarded
  recommendation apply validation.

## Public Placeholder Examples

```text
configs/armtune/target_ssh.example.json
configs/armtune/llama_bench.example.json
configs/armtune/build_profiles.example.json
configs/armtune/llama_bench_optimize.example.json
configs/armtune/recommendation_apply.example.md
```

## Release Boundary

Phase 12 split BootTrace-specific model-capability research into a standalone
BootTraceBench project. Current PerfRunbench documentation covers Arm platform
inspection, benchmarking, build/runtime tuning, statistical confirmation,
Performance Passport generation, and guarded recommendation apply.
