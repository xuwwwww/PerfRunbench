# Auto Retune Policy

`autotuneai run --auto` evaluates cache state before launching the workload.

Default policy:

```text
retune_policy: on_hard_drift_or_missing
canary_policy: on_new_or_drift
```

## Retune Policy

- `never`: never retune automatically.
- `on_hard_drift_or_missing`: retune on `hard_drift`, `expired`, `missing`,
  or `corrupt`.
- `always`: retune before every automatic run.

`ambiguous` and `revoked` require explicit operator action.

## Canary Policy

- `never`: skip canary when the bundle and environment preflight pass.
- `on_new_or_drift`: canary newly tuned, soft-drift, hard-drift, expired, or
  missing paths after retune.
- `always`: canary before every automatic run.

`run --best` always runs a canary because it is an explicit guarded apply
request.

## Boundaries

Automatic retuning is limited to process-scoped command launch configuration.
It does not provision cloud resources, modify global system state, download
models, or add benchmark backends.

For `pytorch-training`, cache-miss retuning also requires a bounded slice
control. If no slice control is configured, `run --auto` fails before candidate
search instead of running an unbounded full training job for every candidate.

