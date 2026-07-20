# Training Slice Protocol

The PyTorch training adapter tunes only bounded, representative slices. It must
not run an unbounded full training job for every candidate.

## Required Slice Controls

A tune or cache-miss `run --auto` requires a bounded slice control:

- `cli_argument`
- `env_var`
- `json_config_path`
- `yaml_config_path`

If a config has no usable bounded slice control, tuning fails with an
actionable configuration error. A normal `autotuneai run -- <command>` remains
available, but it does not perform candidate search.

## Step Accounting

The slice config records:

- `warmup_steps`
- `measured_steps`
- `minimum_completed_steps`
- `max_wall_time_seconds`
- whether the slice control is omitted from the full run

Warm-up work is excluded from the result metric by the workload's emitted
metrics. PerfRunbench records the slice protocol digest in the tuning result,
Performance Passport, and RecommendationBundle metadata.

## Search Discipline

Candidate generation is staged and bounded. The adapter does not build a full
Cartesian product.

Default stages:

```text
input_pipeline -> batch_config -> cpu_runtime -> experimental
```

Typical presets keep the search small:

- quick: up to 8 unique candidates
- standard: up to 16 unique candidates

Search order is deterministic for the same config and seed. Stage status is
recorded as completed, not applicable, truncated, or skipped due to candidate
limit.

## Confirmation

Recommendations require:

- warm-up exclusion
- repeated measured slices
- paired baseline insertion
- fresh confirmation runs
- median comparison
- configured noise-band classification
- enough valid samples

Possible result states include `recommended`, `inconclusive`,
`no_eligible_candidate`, `insufficient_evidence`, and `failed`.
