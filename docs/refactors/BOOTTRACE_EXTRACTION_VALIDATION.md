# BootTrace Extraction Validation

## Scope

This document records the Phase 12 validation for the PerfRunbench side of the
BootTrace extraction.

Target branch:

```text
refactor/split-boottrace
```

Source baseline:

```text
origin/main after v0.1.0-armtune
```

## Extraction Result

Removed from current PerfRunbench:

- firmware-log case suite JSON files
- prompt-contract calibration command
- workload-specific semantic validators
- workload-specific calibration runner
- firmware-log fixture responses
- integrated model-capability result documents

Retained in PerfRunbench:

- generic correctness schemas and aggregation
- generic quality-gate compatibility fields
- optional `quality_gate` config for generic server optimize checks
- artifact digest validation
- benchmark adapters
- build profiles
- runtime search
- Performance Passport
- guarded RecommendationBundle apply

## Safety Audit Criteria

The public audit checks tracked files for:

- public AWS target IP addresses
- private target addresses
- PEM paths
- credentials or tokens
- local private artifact directories
- model weight files
- private absolute paths
- ignored evidence copied into public docs
- stale default CLI flags
- cross-repository imports

Loopback fixture addresses such as `127.0.0.1` are allowed.

## Expected Validation Commands

```powershell
py -3.11 -m unittest discover -s tests -p "test_arm_tune*.py" -v
py -3.11 -m unittest discover -s tests -p "test_*recommendation*.py" -v
py -3.11 scripts\run_tests.py --fast
py -3.11 -m compileall autotune
git diff --check
```

## Claim Boundary

This extraction does not change Phase 9 measured results, Phase 10 guarded
apply evidence, or the `v0.1.0-armtune` release tag. It changes the current
development boundary so firmware-log capability evaluation lives outside
PerfRunbench's default benchmark and recommendation workflow.
