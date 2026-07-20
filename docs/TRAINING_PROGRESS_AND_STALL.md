# Training Progress And Stall Guard

The training adapter requires structured progress and result records. This
keeps tuning evidence separate from arbitrary training logs.

## Helper API

Workloads may use:

```python
from autotune.training_metrics import emit_step, emit_result
```

The helper emits deterministic JSON lines with these prefixes:

```text
AUTOTUNEAI_TRAIN_STEP
AUTOTUNEAI_TRAIN_RESULT
```

The helper has no PyTorch dependency and does not include private data by
default.

## Metric Contract

Conventional fields include:

- `completed_steps`
- `samples_per_second`
- `step_time_ms`
- `final_loss`
- `peak_memory_mb`
- `examples_processed`
- `tokens_per_second`
- `tokens_processed`

The configured objective metric must be present, numeric, and finite. If
`final_loss` is present it must also be finite. Workloads must report completed
steps and nonzero work evidence.

## Failure Classes

Candidate failures are preserved with a failure class:

- `out_of_memory`
- `killed_by_system`
- `timeout`
- `stalled`
- `nonzero_exit`
- `invalid_metric`
- `nonfinite_loss`
- `incomplete_steps`
- `configuration_error`
- `interrupted`
- `cleanup_failed`

Known PyTorch, CUDA, and allocator messages are mapped to `out_of_memory` or
`killed_by_system` where possible. Missing progress or missing final result is
not converted into a zero metric.

## Timeout And Stall Boundary

The adapter applies the slice `max_wall_time_seconds` to bounded candidate
runs. A candidate that exceeds that wall-time limit is terminated and recorded
as `timeout`.

The adapter also validates preserved stdout/result artifacts after each bounded
candidate run. A candidate with no final result record, stale progress, or
fewer than the required completed steps is not eligible.

The serialized progress schema already records startup and no-progress timeout
settings. A finer real-time progress watcher can extend this protocol without
changing the top-level result schema.
