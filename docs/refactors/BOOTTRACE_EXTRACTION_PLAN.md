# BootTrace Extraction Plan

## Scope

This plan records the Phase 12 extraction boundary after the
`v0.1.0-armtune` release tag. The goal is to remove BootTrace-specific model
capability evaluation from PerfRunbench while preserving generic benchmark,
correctness, recommendation, artifact, and guarded-apply infrastructure.

PerfRunbench after this refactor must keep the default ArmTune flow:

```text
inspect
-> benchmark
-> tune
-> confirm
-> Performance Passport
-> recommendation
-> guarded apply
```

BootTrace-specific prompts, case suites, validators, model-qualification
history, and prompt-contract research move to the standalone BootTraceBench
repository.

## Inventory Method

The repository was searched case-insensitively for:

```text
BootTrace
boottrace
firmware trace
firmware log reasoning
UEFI reasoning
prompt contract
held-out capability
model qualification
canonical BootTrace answer
BootTrace quality calibration
```

Inspected areas included Python modules, configuration files, schemas, prompt
templates, validators, tests, fixtures, documentation, evidence summaries, CLI
help text, and package imports/exports.

## Classification Policy

Classification values:

```text
MOVE_TO_BOOTTRACEBENCH
KEEP_GENERIC_IN_PERFRUNBENCH
KEEP_HISTORICAL_REFERENCE
REMOVE_AFTER_EXTRACTION
REVIEW_REQUIRED
```

## Python Modules

| Path | Classification | Rationale |
| --- | --- | --- |
| `autotune/arm_tune/quality/boottrace.py` | MOVE_TO_BOOTTRACEBENCH | BootTrace taxonomy, prompt contracts, calibration-suite loading, prompt rendering, and BootTrace schema digests. |
| `autotune/arm_tune/quality/calibration.py` | MOVE_TO_BOOTTRACEBENCH | Fixed `quality-calibrate` flow imports BootTrace prompt contracts and model suitability semantics. |
| `autotune/arm_tune/quality/diagnostics.py` | MOVE_TO_BOOTTRACEBENCH | Failure categories such as `wrong_root_cause`, `invented_evidence`, and `model_capability_failure` are tied to BootTrace semantic evaluation. |
| `autotune/arm_tune/quality/normalization.py` | MOVE_TO_BOOTTRACEBENCH | Current policy is versioned as `boottrace-normalization-v1` and was created for BootTrace fenced-output diagnostics. |
| `autotune/arm_tune/quality/schemas.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic quality/correctness dataclasses are workload-neutral. Remove only BootTrace-dependent users. |
| `autotune/arm_tune/quality/suite.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic JSON quality-suite loader and prompt-suite snapshot helper. |
| `autotune/arm_tune/quality/validators.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Validator types are generic, but imports and failure categorization must be made BootTrace-free. |
| `autotune/arm_tune/quality/aggregation.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic aggregation should remain after removing BootTrace-specific histogram dependency. |
| `autotune/arm_tune/server_search/runner.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Server search is generic ArmTune runtime tuning. Remove default BootTrace references and keep any optional quality path workload-neutral. |
| `autotune/arm_tune/config.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Config parser is generic. Remove BootTrace normalization dependency and BootTrace-specific quality defaults. |
| `autotune/arm_tune/cli.py` | KEEP_GENERIC_IN_PERFRUNBENCH | CLI remains. Remove `quality-calibrate` and BootTrace-oriented help/summary text. |
| `autotune/arm_tune/benchmark/llama_server.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Server lifecycle, stream parsing, TTFT/E2E, cleanup, and `quality_gate_status: not_evaluated` are generic runtime evidence. |
| `autotune/arm_tune/passport/*` | KEEP_GENERIC_IN_PERFRUNBENCH | Passport eligibility and historical optional `ineligible_quality` states remain parser-compatible. |
| `autotune/recommendation/*` | KEEP_GENERIC_IN_PERFRUNBENCH | RecommendationBundle, guard stages, dry-run, canary, lease, and rollback are generic safety infrastructure. |

## Configurations and Prompt Suites

| Path | Classification | Rationale |
| --- | --- | --- |
| `configs/armtune/quality/boottrace_suite.json` | MOVE_TO_BOOTTRACEBENCH | Held-out BootTrace capability suite with canonical answers and validators. |
| `configs/armtune/quality/boottrace_calibration_suite.json` | MOVE_TO_BOOTTRACEBENCH | Development prompt-contract calibration suite. |
| `configs/armtune/quality/boottrace_tuning_suite.json` | MOVE_TO_BOOTTRACEBENCH | BootTrace deterministic tuning/capability suite. |
| `configs/armtune/quality/boottrace_case_audit.json` | MOVE_TO_BOOTTRACEBENCH | BootTrace canonical-answer audit and taxonomy rationale. |
| `configs/armtune/quality_calibration.example.json` | MOVE_TO_BOOTTRACEBENCH | Example points at BootTrace suites and prompt contracts. |
| `configs/armtune/llama_server.example.json` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic llama-server benchmark example. |
| `configs/armtune/llama_server_optimize.example.json` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic server runtime optimization example. |
| `configs/armtune/prompt_suite.example.json` | KEEP_GENERIC_IN_PERFRUNBENCH | Generic prompt-suite fixture, not BootTrace-specific. |
| `configs/armtune/recommendation_apply.example.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Public guarded-apply example. |

## Tests and Fixtures

| Path | Classification | Rationale |
| --- | --- | --- |
| `tests/test_arm_tune_quality.py` | MOVE_TO_BOOTTRACEBENCH | Mostly BootTrace suite, taxonomy, prompt contract, normalization, and semantic validator tests. Replace in PerfRunbench with BootTrace-free generic validator tests if needed. |
| `tests/test_arm_tune_quality_calibration.py` | MOVE_TO_BOOTTRACEBENCH | Tests BootTrace quality-calibration command and model suitability behavior. |
| `tests/fixtures/arm_tune/boottrace/*` | MOVE_TO_BOOTTRACEBENCH | BootTrace canonical and invalid response fixtures. |
| `tests/test_arm_tune_server_optimize.py` | REVIEW_REQUIRED | Contains both generic server optimize tests and BootTrace quality-gate tests. Keep generic default optimize coverage and move BootTrace quality cases. |
| `tests/fixtures/arm_tune/fake_llama_server.py` | REVIEW_REQUIRED | Generic fake server is still needed, but BootTrace response modes inside it should move or be removed. |
| `tests/test_arm_tune_release_packaging.py` | REVIEW_REQUIRED | Update safety assertions so public release no longer treats BootTrace as in-repo optional workload. |
| `tests/test_recommendation_safety.py` | KEEP_GENERIC_IN_PERFRUNBENCH | Recommendation safety tests are generic. Adjust historical optional fields only if needed. |

## Documentation

| Path | Classification | Rationale |
| --- | --- | --- |
| `README.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Rewrite BootTrace section into a short related/historical note with no in-repo workflow. |
| `docs/armtune/ARCHITECTURE.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Remove integrated BootTrace architecture details; document external workload integration boundary. |
| `docs/armtune/EXEC_PLAN.md` | KEEP_HISTORICAL_REFERENCE | Keep v0.1 history in tag; current main should replace detailed BootTrace phases with migration notes. |
| `docs/armtune/BENCHMARK_METHODOLOGY.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Remove BootTrace as a default quality gate; keep benchmarking/statistical methodology. |
| `docs/armtune/BEFORE_AFTER.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Keep Phase 9/10 performance evidence; remove BootTrace from primary flow. |
| `docs/armtune/DEVPOST_DRAFT.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Keep competition text focused on Arm benchmarking/tuning; move BootTrace to appendix-style historical note if mentioned. |
| `docs/armtune/DEMO_SCRIPT.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Demo must not depend on BootTrace. |
| `docs/armtune/LIMITATIONS.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Retain BootTrace only as moved experimental limitation/history. |
| `docs/armtune/RESEARCH_NOTES.md` | KEEP_HISTORICAL_REFERENCE | Move BootTrace model-qualification details to BootTraceBench experiment history; keep generic Arm research. |
| `docs/armtune/results/PHASE_4_REAL_QUALITY_AWARE_TUNING.md` | MOVE_TO_BOOTTRACEBENCH | Historical BootTrace quality-aware evidence. |
| `docs/armtune/results/PHASE_7_BOOTTRACE_CONTRACT_CALIBRATION.md` | MOVE_TO_BOOTTRACEBENCH | BootTrace prompt-contract calibration history. |
| `docs/armtune/results/PHASE_7_REAL_CONTRACT_SELECTION_AND_MODEL_LADDER.md` | MOVE_TO_BOOTTRACEBENCH | BootTrace model qualification/model ladder history. |
| `docs/armtune/results/PHASE_8_MODEL_QUALIFICATION_PREFLIGHT.md` | MOVE_TO_BOOTTRACEBENCH | BootTrace model qualification preflight history. |
| `docs/armtune/results/PHASE_8_GEMMA12B_QUALIFICATION.md` | MOVE_TO_BOOTTRACEBENCH | BootTrace Gemma qualification history. |
| `docs/armtune/results/PHASE_8_GEMMA12B_NORMALIZATION_DIAGNOSTIC.md` | MOVE_TO_BOOTTRACEBENCH | BootTrace fenced-output diagnostic history. |
| `docs/armtune/results/phase8_gemma12b_normalization_diagnostic.json` | MOVE_TO_BOOTTRACEBENCH | BootTrace diagnostic sidecar. |
| `docs/armtune/results/PHASE_3B1_REAL_SERVER_VALIDATION.md` | REVIEW_REQUIRED | Keep generic llama-server lifecycle evidence; move BootTrace-specific references if present. |
| `docs/armtune/results/PHASE_5_COMPETITION_EVIDENCE.md` | REVIEW_REQUIRED | Keep build/passport architecture evidence; move BootTrace qualification details. |
| `docs/armtune/results/PHASE_6_QWEN25_PROFILE_COMPARISON.md` | REVIEW_REQUIRED | Keep performance-profile details if not quality-dependent; move BootTrace qualification details. |
| `docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Primary Arm build/runtime tuning evidence. |
| `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Guarded apply validation evidence. |
| `docs/recommendation_safety.md` | KEEP_GENERIC_IN_PERFRUNBENCH | Remove BootTrace wording; keep generic correctness guard and apply safety model. |
| `docs/product_vision_zh.md` | REVIEW_REQUIRED | Mentions BIOS/UEFI tuning boundaries but is not a BootTrace workload. Leave unless a later docs audit finds it affects public release scope. |

## Removal Targets in PerfRunbench

After migration is complete, remove or rewrite these integrated BootTrace
entry points from PerfRunbench:

- `autotuneai arm quality-calibrate`
- BootTrace prompt-contract selection fields from public examples.
- BootTrace suite paths from public examples and tests.
- BootTrace response modes inside the fake llama-server fixture.
- BootTrace-specific release-packaging assertions.
- Detailed BootTrace Phase 7/8 evidence documents from current main.

## Compatibility Plan

- The `v0.1.0-armtune` tag preserves the integrated BootTrace state.
- Current PerfRunbench should continue to verify generic artifact manifests,
  Performance Passports, and RecommendationBundles.
- Historical optional quality fields such as `quality_gate_status`,
  `ineligible_quality`, and nested unknown quality payloads must remain safe to
  parse or ignore.
- Removed BootTrace semantic validation is not upgraded into default
  recommendation eligibility.

## Dependency Plan

- BootTraceBench must not import PerfRunbench through relative paths or
  editable checkouts.
- PerfRunbench must not import BootTraceBench.
- Shared data exchange, if needed later, should use stable JSON files or the
  generic command workload path.
- BootTraceBench dependencies should remain standard-library only for the
  initial extraction.
