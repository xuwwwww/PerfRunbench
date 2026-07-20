# Phase 15B T4g Real PyTorch Validation

## Scope

This document records a sanitized Phase 15B validation on a manually started
AWS T4g.small Arm target. The purpose is low-resource portability and workflow
evidence for PerfRunbench's `pytorch-training` adapter. It is not a general
performance claim and must not be compared directly with C7g results.

No AWS lifecycle operation was performed by PerfRunbench during this phase. No
models, datasets, CUDA wheels, `torchvision`, `torchaudio`, or notebook
packages were downloaded.

## Git Integration

```text
integration_commit: 97380574c4271da3ae160b2a3cf95a62133ef674
remote_branch_used_for_phase15b: main
origin_main_updated: true
source_fix_commit: 9738057 fix: support low-resource PyTorch synthetic knobs
controller_remote_commit_match: true
```

The integration branch merged the PyTorch training adapter source, Phase 14B
evidence, and Phase 15A documentation into `origin/main`. During Phase 15B
preflight, the public PyTorch synthetic workload was found to lack the
`--num-workers` and `--seed` flags required by the generated config and real
low-resource validation protocol. The minimal fix added those flags, real
`torch.utils.data.DataLoader` use, and OMP thread evidence to the public
synthetic workload, with a focused regression test.

## Target

```text
target_os: Linux
target_architecture: aarch64
cpu_model: Neoverse-N1
logical_cpus: 2
allowed_cpu_mask: 0,1
kernel_release: 6.17.0-1017-aws
memory_total_mib: 1836
memory_available_mib_before_torch: 1496
memory_available_mib_after_torch_import: 1333
memory_available_mib_after_run: 1434
swap_total_mib: 0
root_filesystem: ext4
root_available_mib_before_torch: 26166
root_available_mib_after_run: 25230
```

CPU credit balance was not queried because this validation did not use AWS API
or CloudWatch access. The target was treated as a burstable low-resource Arm
host with unmeasured CPU-credit state.

## Runtime

```text
python_version: 3.12.3
torch_version: 2.13.0+cpu
torch_cuda_available: false
torch_num_threads_default: 2
torch_num_interop_threads_default: 2
install_source: official PyTorch CPU wheel index
```

The isolated environment used a dedicated virtual environment and installed
PerfRunbench from the exact remote checkout. A non-fatal PyTorch warning was
observed because `numpy` was not installed; `numpy` was not required for the
committed synthetic training workload.

## Source Capability Preflight

```text
remote_head: 97380574c4271da3ae160b2a3cf95a62133ef674
remote_status_short: clean
supports_pytorch_training_adapter: true
supports_init_adapter_pytorch_training: true
supports_public_pytorch_synthetic_example: true
supports_seed_flag: true
supports_num_workers_flag: true
uses_torch_dataloader: true
uses_omp_threads: true
```

## Smoke

The smoke test used the committed public PyTorch synthetic workload with CPU
only, synthetic data, `seed=20260720`, batch 8, accumulation 4, workers 0,
OMP threads 1, and five steps.

```text
smoke_exit_code: 0
progress_records_observed: true
result_record_observed: true
finite_loss_observed: true
residual_worker_observed: false
```

## Bounded Tuning Protocol

```text
world_size: 1
effective_batch_size: 32
amp: disabled
torch_compile: disabled
pin_memory: false
process_scoped_apply_only: true
warmup_steps: 5
measured_steps: 20
minimum_completed_steps: 20
search_runs_per_candidate: 2
fresh_confirmation_repetitions: 3
per_slice_timeout_seconds: 60
total_tuning_budget_minutes: 20
cooldown_seconds: 2
noise_band_percent: 5.0
```

Candidate generation was bounded and did not run a Cartesian product.

| Candidate | Stage | Micro batch | Accumulation | Workers | OMP threads | Effective batch | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | baseline | 8 | 4 | 0 | 1 | 32 | yes |
| `micro_batch_size_16_gradient_accumulation_2` | batch_config | 16 | 2 | 0 | 1 | 32 | yes |
| `micro_batch_size_32_gradient_accumulation_1` | batch_config | 32 | 1 | 0 | 1 | 32 | yes |
| `dataloader_workers_1` | input_pipeline | 8 | 4 | 1 | 1 | 32 | yes |
| `cpu_threads_2` | cpu_runtime | 8 | 4 | 0 | 2 | 32 | yes |

## Result

```text
recommendation_bundle_id: 1bce3e25a12db1db3eaccefb
bundle_status: eligible_preliminary
status: recommended
classification: improved
selected_candidate_id: micro_batch_size_32_gradient_accumulation_1
candidate_count_total_including_baseline: 5
unique_nonbaseline_candidate_count: 4
failed_trial_count: 0
paired_confirmation_count: 3
valid_confirmation_baseline_trials: 3
valid_confirmation_selected_trials: 3
valid_search_baseline_trials: 2
valid_search_selected_trials: 8
remote_manifest_validation_status: passed
local_artifact_revalidation_status: passed
```

Paired confirmation summary:

| Metric | Baseline | Selected |
| --- | ---: | ---: |
| median `samples_per_second` | 13796.607073 | 28037.469976 |
| median `step_time_ms` | 2.319411 | 1.141330 |
| median `final_loss` | 1.335323 | 1.330552 |
| sample variability percent | 2.612990 | 14.290842 |

```text
observed_relative_difference_percent: 103.220037
noise_band_percent: 5.0
noise_classification: outside_noise_band
```

This is a low-resource synthetic PyTorch validation result on one T4g.small
target. It validates that PerfRunbench can tune and safely reuse an eligible
configuration under the tested workload and measurement protocol. It does not
claim a general PyTorch speedup.

Allowed wording:

```text
Best observed eligible configuration under the tested training workload,
slice protocol, and measurement protocol.
```

## Golden Path

The full user workflow was exercised with the same command and config:

```text
pre_tune_run_best_status: failed
pre_tune_run_best_cache_state: missing
pre_tune_run_best_retuned: false

run_auto_cache_miss_status: completed
run_auto_cache_miss_newly_tuned: true
run_auto_cache_miss_cache_state: soft_drift
run_auto_cache_miss_canary_status: passed
run_auto_cache_miss_full_run_completed_steps: 50

run_auto_cache_reuse_status: completed
run_auto_cache_reuse_newly_tuned: false
run_auto_cache_reuse_cache_state: soft_drift
run_auto_cache_reuse_canary_status: passed
run_auto_cache_reuse_full_run_completed_steps: 50

run_best_cache_reuse_status: completed
run_best_cache_reuse_newly_tuned: false
run_best_cache_reuse_cache_state: soft_drift
run_best_cache_reuse_canary_status: passed
run_best_cache_reuse_full_run_completed_steps: 50
```

The cache state was `soft_drift` because low-resource available memory changed
between lookup and apply. It was not a hard CPU mismatch. `run --best` did not
retune and required the existing eligible cached recommendation.

## Drift And Safety

The Phase 14B C7g RecommendationBundle was tested on the T4g target only
through validation and dry-run apply. No C7g workload was launched.

```text
c7g_bundle_id: 555238aaec70532e69c538d6
t4g_bundle_id: 1bce3e25a12db1db3eaccefb
compare_status: hard_mismatch
dry_run_status: dry_run_failed
retune_required: true
workload_launched: false
```

Hard mismatch fields:

```text
logical_cpu_count
allowed_cpu_mask
runtime_version
workload_config_sha256
```

Soft mismatch fields:

```text
kernel_release
available_memory_mb
hostname
```

Apply safety status:

```text
apply_scope: process
lock_state_after_workflow: rolled_back
lease_state_after_workflow: rolled_back
no_active_lock: true
no_active_lease: true
residual_worker_processes: none observed
```

## Artifact Validation

```text
remote_manifest_validation_status: passed
remote_manifest_issues: 0
downloaded_archive_sha256_matches_remote: true
local_artifact_manifest_validation_status: passed
local_artifact_manifest_issues: 0
checked_artifact_count: 3
```

## Tests

Pre-publication integration tests before the Phase 15B source push:

```text
py -3.11 -m unittest tests.test_pytorch_training_adapter -v
result: 10 tests OK

py -3.11 -m unittest tests.test_golden_path_cli tests.test_recommendation_safety -v
result: 16 tests OK

py -3.11 -m unittest discover -s tests -p "test_arm_tune*.py" -v
result: 150 tests OK

py -3.11 scripts\run_tests.py --fast
result: 311 tests OK

py -3.11 -m compileall autotune
result: passed

git diff --check
result: passed
```

Focused tests after the public PyTorch synthetic workload fix:

```text
py -3.11 -m unittest tests.test_pytorch_training_adapter -v
result: 11 tests OK

py -3.11 -m unittest tests.test_golden_path_cli tests.test_recommendation_safety -v
result: 16 tests OK

py -3.11 -m compileall autotune examples\pytorch_synthetic_training
result: passed

git diff --check
result: passed
```

Remote T4g focused test:

```text
python -m unittest tests.test_pytorch_training_adapter -v
result: 11 tests OK
```

Final post-documentation tests are recorded in the completion report for the
Phase 15B evidence commit.

## Limitations

- This is a single T4g.small validation on one burstable Arm target.
- CPU credit state was not queried, so the measured throughput is portability
  and workflow evidence rather than a stable performance claim.
- The workload uses synthetic data and a small public model fragment inside the
  example script, not a real user dataset.
- The selected configuration changes micro-batch and accumulation while
  preserving effective batch size. It does not validate training quality beyond
  finite loss and successful bounded execution.
- The selected confirmation variability was higher than the baseline. No
  variability threshold was configured for this low-resource validation.
- C7g and T4g results must not be compared directly.
