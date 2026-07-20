# CLI Reference

## New-User Commands

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

`init` creates a public placeholder config. It refuses to overwrite an existing
file unless `--overwrite` is provided. `--force` remains a backward-compatible
alias.

`doctor` checks readiness without running the workload or modifying the config.
It returns exit code `0` when no check has status `fail`. Warnings remain
visible in terminal and JSON output.

```bash
autotuneai doctor --json --config autotune.yaml -- python train.py
```

Doctor check statuses:

- `pass`
- `warning`
- `fail`
- `not_available`

## Run Modes

Plain monitored run:

```bash
autotuneai run -- python train.py
```

Tune and cache only:

```bash
autotuneai tune --config autotune.yaml -- python train.py
```

Automatically tune on cache miss or hard drift according to policy:

```bash
autotuneai run --auto --config autotune.yaml -- python train.py
```

Reuse an eligible cache match without retuning:

```bash
autotuneai run --best --config autotune.yaml -- python train.py
```

`--best` fails with an actionable missing-cache error when no eligible cache
entry exists.

## Recommendation Cache

```bash
autotuneai cache list
autotuneai cache show <cache-id>
autotuneai cache invalidate <cache-id>
autotuneai cache gc
```

Cache states:

- `exact_match`
- `soft_drift`
- `hard_drift`
- `expired`
- `revoked`
- `missing`
- `corrupt`
- `ambiguous`

## Recommendation Safety

```bash
autotuneai recommendation show --bundle <bundle.json>
autotuneai recommendation validate --bundle <bundle.json>
autotuneai recommendation apply --bundle <bundle.json> --scope process --dry-run
autotuneai recommendation apply --bundle <bundle.json> --scope process --lease-minutes 30 --canary-runs 1
autotuneai recommendation status --bundle <bundle.json>
autotuneai recommendation rollback --bundle <bundle.json>
```

Process-scoped apply is the default safety model. It validates the bundle,
checks environment drift, runs a canary when required, records a lease, and
rolls back on failure.

## ArmTune Commands

Inspect an Arm target:

```bash
autotuneai arm doctor --target-config configs/armtune/target_ssh.example.json --require-production
autotuneai arm inspect-host --target-config configs/armtune/target_ssh.example.json
```

Run a `llama-bench` baseline:

```bash
autotuneai arm benchmark --config configs/armtune/llama_bench.example.json
```

Build and compare profiles:

```bash
autotuneai arm build-plan --config configs/armtune/build_profiles.example.json
autotuneai arm build-profile --config configs/armtune/build_profiles.example.json
autotuneai arm profile-compare --config configs/armtune/profile_comparison.example.json
```

Run bounded runtime search:

```bash
autotuneai arm optimize --config configs/armtune/llama_bench_optimize.example.json --max-candidates 12
```

Generate and verify a Performance Passport:

```bash
autotuneai arm passport --comparison-result <profile-comparison-result.json>
autotuneai arm verify-passport --passport <performance-passport.json>
```

## Advanced Legacy Commands

The repository also retains lower-level resource and tuning commands such as
`inspect`, `executors`, `compare-tuning`, `compare-profiles`,
`compare-budgets`, `optimize`, `optimize-performance`, `tune-system`,
`tune-runtime`, and `restore`.

Use these commands only when you need explicit low-level resource controls
instead of the golden-path `init -> doctor -> run --auto` workflow.
