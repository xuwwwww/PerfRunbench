# ArmTune Documentation Index

ArmTune is the Arm-first AI inference autotuning subsystem planned inside
PerfRunbench. These documents define the current Phase 0 audit, target
architecture, benchmark integrity rules, and phased implementation plan.

No document in this directory contains measured Arm performance results yet.
Any examples, templates, or future-result placeholders are labeled as such.

## Documents

- `EXEC_PLAN.md`: Phase 0 audit record, repository findings, command log, and
  phase-by-phase execution plan.
- `IMPLEMENTATION_PLAN.md`: Reviewable implementation phases and expected code
  surfaces.
- `ARCHITECTURE.md`: Controller/target design and package boundary proposal.
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
- `LIMITATIONS.md`: Current unsupported features and integrity constraints.
- `RESEARCH_NOTES.md`: Source-tracked research notes for competition,
  llama.cpp, KleidiAI, and Arm Performix decisions.
- `RESULTS_TEMPLATE.md`: Placeholder format for future measured, synthetic,
  fixture, and unverified results.
- `DEVPOST_DRAFT.md`: Early English submission narrative skeleton.
- `DEMO_SCRIPT.md`: Three-minute demo outline and required evidence checklist.

## Phase Boundary

Phase 0 creates planning and audit artifacts only. Product code, SSH execution,
llama.cpp adapters, benchmarking, search, passport generation, and Performix
integration start in later phases after explicit approval.
