# Phase 14B Real Arm PyTorch Training Validation

## Scope

This document records the sanitized Phase 14B validation of PerfRunbench's
`pytorch-training` adapter on a real Linux Arm64 target. The run validates:

- guest root filesystem expansion after the user enlarged the root EBS volume
- isolated CPU-only PyTorch environment setup
- bounded training slice tuning
- RecommendationCache insertion and reuse
- process-scoped canary, lease, and rollback
- hard-drift rejection
- Performance Passport generation
- remote and downloaded artifact digest validation

This evidence does not control AWS lifecycle, does not download models or
datasets, does not install CUDA wheels, does not apply system-scoped tuning,
and does not claim global training performance.

Private endpoint, SSH identity, private controller paths, raw home-directory
paths, and target configuration paths are intentionally omitted.

## Target And Repository State

```text
target_class: AWS C7g / Neoverse-V1
target_os: Linux
target_architecture: aarch64
logical_cpus: 8
controller_to_target_access: system OpenSSH with StrictHostKeyChecking=yes
remote_worktree_status: clean
remote_head: 3e4396303cf69acd3ddda4392ea7c7c2a65b517a
```

The remote HEAD includes the Phase 14B validation fixes for:

- generated YAML parsing without PyYAML
- equivalent CPU mask comparison, such as `0-7` and `0,1,2,3,4,5,6,7`
- preserving virtualenv executable paths instead of resolving them to the
  underlying system Python interpreter

## Guest Filesystem Expansion

The root EBS block device was already enlarged by the user before guest-side
filesystem work started. The guest expansion used actual `findmnt`, `lsblk`,
`df`, and `fdisk` evidence before choosing the root disk, root partition, and
filesystem-specific resize command.

```text
root_volume_size_before: 64.0 GiB
root_volume_size_after: 64.0 GiB
root_partition_size_before: 28.999 GiB
root_partition_size_after: 62.999 GiB
root_filesystem_type: ext4
root_free_space_before: 7.997 GiB
root_free_space_after: 40.931 GiB
filesystem_expansion_status: passed
```

The executed guest-side operation was limited to the verified root partition
and `ext4` filesystem. No format, partition deletion, filesystem conversion,
mount configuration change, model-directory change, or non-root filesystem
expansion was performed.

## Runtime Environment

```text
python: 3.12.3
torch: 2.13.0+cpu
cuda_available: false
official_wheel_index: https://download.pytorch.org/whl/cpu
dataset_downloaded: false
model_downloaded: false
system_scoped_tuning: false
```

The final validation command used the isolated virtualenv Python executable in
the stored launch specs. Earlier diagnostic attempts correctly failed because
the launch spec resolved the virtualenv symlink to the system Python
interpreter, which did not have `torch` installed.

## Workload And Search Protocol

The workload was the committed synthetic PyTorch training example:

```text
adapter: pytorch-training
world_size: 1
baseline_micro_batch_size: 32
baseline_gradient_accumulation: 4
baseline_effective_batch_size: 128
selected_micro_batch_size: 128
selected_gradient_accumulation: 1
selected_effective_batch_size: 128
slice_flag: --max-steps
warmup_steps: 10
measured_steps: 60
full_run_steps: 200
result_origin: measured
```

Unsupported features were not forced:

```text
dataloader_workers: not used
dataloader_prefetch: not used
dataloader_persistent_workers: not used
amp: disabled
torch_compile: disabled
gpu: not used
```

Candidate coverage:

```text
candidate_count: 7
trial_count: 32
warmup_trial_count: 1
valid_search_baseline_trials: 3
valid_search_selected_trials: 18
valid_confirmation_baseline_trials: 5
valid_confirmation_selected_trials: 5
failed_trials: 0
contaminated_trials: 0
paired_confirmation_count: 5
noise_band_percent: 3.0
```

Stage coverage:

```text
input_pipeline: not_applicable, candidate_count=0
batch_config: completed, candidate_count=3
cpu_runtime: completed, candidate_count=3
```

## Result

The selected candidate was:

```text
selected_candidate_id: micro_batch_size_128_gradient_accumulation_1
selected_candidate_recommendation_eligible: true
```

Bounded tuning and confirmation produced:

```text
objective_metric: samples_per_second
baseline_confirmation_median: 77859.454508
selected_confirmation_median: 203215.038822
observed_improvement_percent: 161.002392
classification: improved
statistical_status: recommended
passport_claim_eligibility_state: eligible_preliminary
```

Allowed recommendation wording:

```text
Best observed eligible configuration under the tested training workload,
slice protocol, and measurement protocol.
```

Do not compare this synthetic training result to the Phase 9 llama.cpp
inference results. They use different workloads, metrics, and protocols.

## Golden-Path Validation

```text
run_auto_cache_miss_status: completed
run_auto_cache_miss_newly_tuned: true
run_auto_cache_miss_canary: passed
run_auto_cache_reuse_status: completed
run_auto_cache_reuse_newly_tuned: false
run_auto_cache_reuse_canary: passed
run_best_status: completed
run_best_canary: passed
cache_state_observed: soft_drift
```

The cache state was `soft_drift` because available memory is a soft
fingerprint field and naturally changed during validation. Hard fingerprint
fields matched.

Full workload launches completed `200` steps and emitted one structured result
record in each run:

```text
run_auto_cache_miss_samples_per_second: 213654.266969
run_auto_cache_reuse_samples_per_second: 215273.197791
run_best_samples_per_second: 215613.995765
```

These full-run samples validate that the cached selected configuration can be
launched after guarded apply. They are not a separate statistical comparison.

## Recommendation Safety

```text
recommendation_bundle_id: 555238aaec70532e69c538d6
bundle_validate_status: pass
dry_run_apply_status: dry_run_passed
dry_run_drift_status: soft_mismatch
dry_run_retune_required: false
apply_scope: process
canary_runs: 1
canary_status: passed
lease_state_after_run: rolled_back
lock_state_after_run: rolled_back
hard_drift_test_status: dry_run_failed
hard_drift_test_retune_required: true
hard_drift_fail_code: stale_environment
```

The hard-drift test used a modified private copy of the bundle with a mismatched
hard architecture fingerprint. It was rejected before apply.

## Artifact Validation

Remote validation:

```text
tuning_artifact_manifest_status: passed
checked_tuning_artifacts: 3
bundle_validation_status: pass
```

Downloaded validation:

```text
downloaded_archive_file_count: 418
downloaded_archive_manifest_status: passed
downloaded_tuning_artifact_manifest_status: passed
downloaded_archive_sha256: 03d24263a1f26de6c5e5770c5fc0c47a3d2056213b0c846488b6ea738b690db7
```

## Cleanup Evidence

```text
residual_train_synthetic_processes: 0
residual_autotune_processes: 0
residual_phase14_driver_processes: 0
residual_phase14_listener_processes: 0
active_lease: false
```

Only standard system listeners such as SSH and local resolver sockets were
observed.

## Limitations

- The workload is synthetic and public; it validates adapter mechanics rather
  than end-user model training quality.
- The selected candidate preserves effective batch size but changes
  micro-batch and gradient accumulation shape, so results are bound to this
  workload and protocol.
- `available_memory_mb` caused cache reuse to report `soft_drift`, not
  `exact_match`; this is acceptable but should be explained in demos.
- The committed synthetic workload hard-codes `torch.manual_seed(0)` and does
  not expose a seed CLI flag.
- PyTorch was installed only in an isolated user virtualenv with CPU wheels.
- The evidence is single-target, single-runtime, and preliminary.

