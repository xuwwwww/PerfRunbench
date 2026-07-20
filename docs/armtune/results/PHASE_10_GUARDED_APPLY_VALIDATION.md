# Phase 10 Guarded Recommendation Apply Validation

## Scope

This document records the sanitized real Phase 10 validation of PerfRunbench
RecommendationBundle safety using the previously generated Phase 9 ArmTune
Performance Passport. It validates recommendation show, validate, dry-run
apply, process-scoped canary, lease state, rollback, artifact revalidation, and
cleanup.

This evidence does not control AWS lifecycle, does not apply permanent
system-scoped tuning, does not rerun the Phase 9 profile comparison, and does
not create a new performance claim.

## RecommendationBundle

```text
bundle_id: 92b0eb69b189a06baae62f15
source: Phase 9 eligible preliminary Performance Passport
apply_scope: process
result_origin: measured
source_passport_verification: passed
```

The public evidence intentionally omits the target endpoint, SSH identity-file
path, private target configuration, private controller paths, model contents,
and raw private run directories.

## Environment Validation

```text
target_os: Linux
target_architecture: aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
environment_validation: pass
retune_required: false
```

The environment validation confirms that the bundle did not require retuning
for the observed target and runtime identity used in this validation.

## Dry-Run Apply

```text
dry_run_status: dry_run_passed
guard_status: pass
retune_required: false
```

The dry run validated the bundle, source Passport digest, expected environment
shape, selected launch specification, correctness contract, and guard
configuration without starting a canary process.

## Canary Result

```text
canary_status: passed
canary_runs: 1
apply_scope: process
prefill_tokens_per_second: 52.1
decode_tokens_per_second: 17.56
decode_ms_per_token: 56.94760820045558
```

The canary result is an apply-safety validation sample, not a new profile
comparison or a replacement for the Phase 9 repeated benchmark distributions.

## Lease And Rollback

```text
lease_state_after_apply: active
lock_state_after_apply: active
rollback_state: rolled_back
system_restore_run_id: not_applicable_process_scope
```

The validation exercised the process-scoped lease and rollback path. No
permanent system-scoped tuning was promoted.

## Artifact Revalidation

```text
remote_manifest_revalidation_checked: 26
remote_manifest_revalidation_failed: 0
downloaded_manifest_revalidation_checked: 26
downloaded_manifest_revalidation_failed: 0
```

Artifact manifests were independently revalidated after generation and after
download. Digest validation passed in both locations.

## Cleanup Evidence

```text
residual_process_count: 0
residual_listener_count: 0
cleanup_status: passed
```

No residual benchmark worker, server process, or benchmark listener was
observed after rollback and cleanup checks.

## Limitations

- This validates guarded recommendation application, not a new optimization
  result.
- The canary sample is not statistically comparable to the Phase 9 repeated
  profile comparison.
- Only process-scoped apply was validated; permanent system-scoped apply
  remains out of scope.
- The result is bound to the Phase 9 Passport, target class, model, runtime,
  and protocol digests used by the bundle.
- Private target configuration, model contents, SSH identity material, and
  endpoint details are intentionally excluded from public documentation.
