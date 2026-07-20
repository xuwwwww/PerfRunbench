# Phase 15A T4g Low-Resource Arm Validation

## Scope

This document records a sanitized Phase 15A validation on a manually started
low-resource Arm target. The purpose is guard behavior, portability, cache
safety, and constrained-target operation. It is not a performance claim and
must not be compared directly with the Phase 9 or Phase 14B C7g evidence.

No AWS lifecycle operation was performed by PerfRunbench during this phase. The
paid C7g target was not accessed.

## Target

```text
target_os: Linux
target_architecture: aarch64
cpu_model: Neoverse-N1
logical_cpus: 2
allowed_cpus: 2
kernel_release: 6.17.0-1017-aws
memory_total_gib: 1.794
memory_available_gib_preflight: 1.425
swap_total_gib: 0.000
root_filesystem_free_gib_preflight: 25.611
environment_fingerprint_digest: 6035ac6cea86d3a201718018636ec1f3860585fbdf53f6ee360677d73e4a4601
workload_identity_digest: d0b10ff7ed99e36dded22d59d3e60c4fad72d0e27f9e2ee1b6ba859a427dc5c0
```

CPU credit balance was not queried because this validation did not use AWS API
or CloudWatch access. The target was treated as a burstable instance with
unmeasured CPU-credit state.

## Source Checkout

The remote validation used a clean checkout of latest `origin/main`:

```text
remote_head: 26ade1f259152256101de2798d1f83898e252e50
remote_status_short: clean
```

The clean `origin/main` checkout used for this validation did not contain the
Phase 14B PyTorch training adapter or public synthetic PyTorch example:

```text
has_training_adapter: false
has_pytorch_example: false
has_pytorch_doc: false
```

Because no source entry point was available from the required clean main
checkout, real PyTorch was not installed or executed. Phase 15A continued with
the dependency-free training fixture to validate guard behavior safely on the
same low-resource Arm target.

## C7g Recommendation Reuse Refusal

The Phase 14B C7g RecommendationBundle was tested only through validation and
dry-run apply. No workload was launched through this refusal path.

```text
bundle_id: 555238aaec70532e69c538d6
bundle_status: eligible_preliminary
validation_status: fail
apply_dry_run_status: dry_run_failed
environment_drift_status: hard_mismatch
retune_required: true
workload_launch_attempted: false
```

Hard mismatch fields:

```text
logical_cpu_count
allowed_cpu_mask
executable_sha256
workload_config_sha256
```

Soft mismatch fields:

```text
kernel_release
available_memory_mb
hostname
```

This proves the C7g recommendation could not be reused on the T4g target. The
clean `run --best` path was also exercised before tuning and failed with a
missing cache entry:

```text
pre_tune_run_best_status: failed
pre_tune_run_best_cache_state: missing
pre_tune_run_best_reason: no usable eligible cached recommendation
```

## Dependency-Free Fixture

The low-resource fixture preserved the training semantics needed for this
guard validation:

```text
world_size: 1
effective_batch_size: 32
baseline: batch 8 / accumulation 4
alternatives: batch 16 / accumulation 2, batch 32 / accumulation 1
workers_allowed: 0, 1, 2
omp_threads_allowed: 1, 2
amp: disabled
torch_compile: disabled
model_downloads: none
dataset_downloads: none
```

Candidate generation was bounded and did not use a full Cartesian product.

```text
candidate_count_including_baseline: 5
search_order:
  baseline
  intentional_failed_candidate
  batch16_accum2_workers1_omp1
  batch16_accum2_workers0_omp2
  batch32_accum1_workers0_omp2
rejected_before_execution: 3
full_cartesian_product_used: false
```

The rejected candidates were excluded before execution because they violated
the CPU-pressure, worker-count, or effective-batch guard.

## Tuning Result

The dependency-free fixture completed and produced an eligible cache entry:

```text
dependency_free_fixture_status: recommended
statistical_status: recommended
statistical_classification: improved
selected_candidate_id: batch32_accum1_workers0_omp2
trial_count: 17
valid_trial_count: 15
failed_trial_count: 2
noise_band_percent: 5.0
warmup_steps_per_trial: 5
measured_steps_per_trial: 20
search_repetitions: 2
confirmation_repetitions: 3
```

The fixture metric values are guard evidence only:

```text
baseline_median_fixture_samples_per_second: 96.2
selected_median_fixture_samples_per_second: 130.2
fixture_improvement_percent: 35.343035
```

These fixture values are not Arm performance results and are not comparable to
C7g llama.cpp or PyTorch evidence.

The intentional failed candidate remained failed and was not converted to a
zero-valued metric:

```text
failed_candidate_ids: intentional_failed_candidate
failed_candidate_metric_values: null, null
failed_candidates_not_zeroed: true
```

## Cache And Apply Safety

Cache insertion occurred only after eligible statistical evidence was present:

```text
cache_inserted_only_for_eligible: true
cache_lookup_state_after_tune: soft_drift
cache_lookup_hard_mismatch_count: 0
```

The observed cache reuse state was `soft_drift` because available memory and
other soft fields can change on the constrained target. A memory-only drift
diagnostic confirmed that available-memory changes remain soft and do not
produce a hard CPU mismatch:

```text
soft_memory_drift_diagnostic_status: soft_mismatch
soft_memory_drift_hard_mismatch_count: 0
```

`run --best` was exercised twice after tuning. Both runs reused the cached
recommendation and did not retune:

```text
run_best_first_status: completed
run_best_first_cache_state: soft_drift
run_best_first_newly_tuned: false
run_best_second_status: completed
run_best_second_cache_state: soft_drift
run_best_second_newly_tuned: false
```

Process-scoped apply, canary validation, and rollback were also exercised:

```text
apply_status: applied
apply_canary_status: passed
rollback_state: rolled_back
```

## Budget Guard

A small dependency-free memory guard test deliberately exceeded a process-level
memory budget without creating machine-wide memory pressure:

```text
memory_budget_guard_run_status: failed
memory_budget_guard_return_code: -15
memory_budget_guard_memory_exceeded: true
memory_budget_guard_peak_rss_mb: 11.751
```

This validates clean budget termination behavior on the constrained Arm target.
It was not an OOM test.

## Artifact Validation

Artifact manifests were validated on the remote target and again after download:

```text
remote_artifact_validation_status: passed
remote_artifact_validation_checked_artifacts: 3
remote_evidence_archive_validation_status: passed
remote_evidence_archive_checked_files: 321
downloaded_evidence_archive_validation_status: passed
downloaded_evidence_archive_checked_files: 321
downloaded_evidence_archive_sha256: 51525f6c094375c62e1119a25a4f113364504b22ae09536831e78e2d22957045
```

## Cleanup

```text
residual_process_check: none
active_worker_check: none
active_listener_check: none
rollback_state: rolled_back
cleanup_status: passed
```

## Limitations

- This phase validates low-resource behavior and safety on a T4g-class target;
  it is not a performance benchmark.
- Real PyTorch was skipped because the required clean `origin/main` checkout
  did not contain the Phase 14B PyTorch training adapter or public synthetic
  PyTorch example.
- The dependency-free fixture validates guard semantics, cache behavior,
  process cleanup, and artifact integrity only.
- CPU credit balance was not measured because AWS API or CloudWatch access was
  not used.
- The target has no swap and approximately 2 GiB memory, so future real
  PyTorch validation must continue to enforce conservative disk and memory
  preflight checks.
- Private endpoint, SSH identity, private target config, raw private paths, and
  hostnames are intentionally omitted from public documentation.

## Result

```text
phase15a_t4g_low_resource_validation: passed
ready_to_stop_t4g: true
```
