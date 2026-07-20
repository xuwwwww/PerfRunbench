# Effective Batch Guard

Training performance can change when effective batch size changes. The
`pytorch-training` adapter treats this as a workload-semantic boundary, not as
a normal runtime tweak.

Effective batch size is:

```text
micro_batch_size * gradient_accumulation * world_size
```

`world_size` is fixed for a tuning run.

## Default Policy

The default policy is:

```yaml
training:
  semantics:
    effective_batch_policy: preserve
    semantic_change_allowed: false
```

Under this policy, batch-stage candidates are generated only when the effective
batch size remains equal to the baseline. For example, `micro_batch_size: 8`
can be paired with `gradient_accumulation: 1` when the baseline is
`4 * 2 * 1`.

## Explicit Semantic Changes

Semantic changes require:

```yaml
effective_batch_policy: allow_change
semantic_change_allowed: true
```

Candidates with changed effective batch size are recorded as
`changed_explicitly_allowed`, but they are not eligible for the default
recommendation path. PerfRunbench does not automatically scale learning rate
and does not mix preserved and changed effective-batch candidates in one
default recommendation.

## Dependency Guards

The adapter rejects or records guard evidence for:

- non-positive batch, accumulation, or thread counts
- negative dataloader workers
- prefetch or persistent workers with zero dataloader workers
- thread counts above visible CPU count
- AMP or compile modes without declared support
- CPU `pin_memory` benefit as unverified until measured

Rejected candidates remain in artifacts as failed evidence; their metrics are
not replaced with zeros.
