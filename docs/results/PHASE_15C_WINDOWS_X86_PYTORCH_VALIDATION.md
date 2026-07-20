# Phase 15C Windows x86 PyTorch Validation

## Scope

This document records a sanitized Phase 15C validation of PerfRunbench's
golden-path CLI and `pytorch-training` adapter on a local Windows x86_64
controller. AWS was not accessed or controlled during this phase.

This is portability, guard, and local workflow validation. It is not an Arm
performance claim and must not be compared directly with C7g or T4g results.

## Source

```text
branch: validation/phase15c-windows-x86
origin_main_commit: 97380574c4271da3ae160b2a3cf95a62133ef674
phase15b_evidence_commit_present: 40aee3c0b8e78ad9b9f6611df1d97df7f08c0457
validation_source_commit: 5bf2a0d217283e8da2ed4954a9be65a4be4ed954
fix_commit: 5bf2a0d fix: normalize hex CPU affinity masks
```

The validation first exposed a real false-hard-drift defect: a Windows-style
hex CPU affinity mask such as `0xff` was not canonicalized as equivalent to
`0,1,2,3,4,5,6,7`. The focused fix adds hex mask normalization and a regression
test. The real golden-path validation was rerun after that fix was committed.

## Target

```text
target_os: Windows 11 Home
windows_version: 10.0.26200
architecture: X64 / AMD64
cpu_model: 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz
logical_cpus: 8
process_affinity_mask: 0xff
total_memory_mib: 24267
available_memory_mib_before_torch: 5662
repo_drive_free_mib_before_torch: 34174
powershell_version: 5.1.26100.8875
python_version: 3.11.0
```

The preflight requirements were satisfied before installing PyTorch:

```text
free_disk_at_least_8_gib: true
available_memory_at_least_4_gib: true
clean_source_checkout: true
```

## Runtime

```text
torch_version: 2.13.0+cpu
torch_cuda_available: false
torch_num_threads_default: 4
torch_num_interop_threads_default: 4
platform_machine: AMD64
install_source: official PyTorch CPU wheel index
```

The environment was an isolated local CPU-only virtual environment. No CUDA
wheel, `torchvision`, `torchaudio`, model, dataset, notebook package, or AWS
credential was installed or used.

A non-fatal PyTorch warning was observed because `numpy` was not installed.
The public synthetic workload does not require NumPy.

## Windows CLI Validation

The generic CLI wrapper was exercised from a working directory containing
spaces. The command after `--` was preserved as argv and not concatenated into
a shell string.

```text
args_with_spaces_preserved: true
unicode_arg_preserved: true
quoted_value_preserved: true
flag_after_separator_preserved: true
cwd_with_spaces_preserved: true
success_exit_code_visible: true
child_nonzero_exit_code_visible: true
stdout_stderr_visible: true
timeout_returns_124: true
timeout_cleanup_no_residual_marker: true
```

## PyTorch Smoke

The smoke test used the committed public synthetic PyTorch workload with CPU
only, `seed=20260720`, micro batch 16, gradient accumulation 4, effective batch
64, workers 0, OMP threads 1, and five steps.

```text
smoke_exit_code: 0
completed_steps: 5
examples_processed: 320
torch_num_threads: 1
finite_loss_observed: true
final_loss: 1.3717694282531738
```

## Bounded Tuning Protocol

```text
world_size: 1
effective_batch_size: 64
amp: disabled
torch_compile: disabled
pin_memory: false
seed: 20260720
warmup_steps_per_slice: 5
measured_steps_per_slice: 30
minimum_completed_steps: 30
search_repetitions_per_candidate: 3
fresh_confirmation_repetitions: 5
per_slice_timeout_seconds: 90
cooldown_seconds: 2
noise_band_percent: 5.0
candidate_count_total_including_baseline: 6
```

Candidate generation was bounded and did not run a Cartesian product.

| Candidate | Stage | Micro batch | Accumulation | Workers | OMP threads | Effective batch | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | baseline | 16 | 4 | 0 | 1 | 64 | yes |
| `micro_batch_size_32_gradient_accumulation_2` | batch_config | 32 | 2 | 0 | 1 | 64 | yes |
| `cpu_threads_2` | cpu_runtime | 16 | 4 | 0 | 2 | 64 | yes |
| `cpu_threads_4` | cpu_runtime | 16 | 4 | 0 | 4 | 64 | yes |
| `dataloader_workers_2` | input_pipeline | 16 | 4 | 2 | 1 | 64 | yes |
| `micro_batch_size_64_gradient_accumulation_1` | batch_config | 64 | 1 | 0 | 1 | 64 | yes |

## Tuning Result

```text
run_id: 20260720_205338_114313
recommendation_bundle_id: c61f327d271288d01437d435
bundle_status: eligible_preliminary
status: recommended
classification: improved
selected_candidate_id: micro_batch_size_64_gradient_accumulation_1
failed_trial_count: 0
paired_confirmation_count: 5
valid_confirmation_baseline_trials: 5
valid_confirmation_selected_trials: 5
valid_search_baseline_trials: 3
valid_search_selected_trials: 15
artifact_manifest_revalidated: passed
```

Bounded slice confirmation:

| Metric | Baseline | Selected |
| --- | ---: | ---: |
| median `samples_per_second` | 37722.186201 | 70585.862645 |
| observed relative difference |  | 87.120286% |
| selected variability percent |  | 3.332346 |

This result is a Windows x86_64 synthetic PyTorch validation result. It should
be described only as:

```text
Best observed eligible configuration under the tested training workload,
slice protocol, and measurement protocol.
```

## Golden Path

The full user workflow was exercised with the same command and config.

```text
pre_tune_run_best_status: failed
pre_tune_run_best_cache_state: missing
pre_tune_run_best_retuned: false
pre_tune_run_best_workload_launched: false

run_auto_cache_miss_status: completed
run_auto_cache_miss_newly_tuned: true
run_auto_cache_miss_cache_state: exact_match
run_auto_cache_miss_canary_status: passed
run_auto_cache_miss_full_run_completed_steps: 100
run_auto_cache_miss_full_run_examples_processed: 6400

run_auto_cache_reuse_status: completed
run_auto_cache_reuse_newly_tuned: false
run_auto_cache_reuse_cache_state: exact_match
run_auto_cache_reuse_canary_status: skipped
run_auto_cache_reuse_full_run_completed_steps: 100
run_auto_cache_reuse_full_run_examples_processed: 6400

run_best_cache_reuse_status: completed
run_best_cache_reuse_newly_tuned: false
run_best_cache_reuse_cache_state: exact_match
run_best_cache_reuse_canary_status: passed
run_best_cache_reuse_full_run_completed_steps: 100
run_best_cache_reuse_full_run_examples_processed: 6400
```

The second `run --auto` and the post-cache `run --best` did not create a new
tuning result. `run --best` required the existing eligible cached
RecommendationBundle.

## Full-Workload Paired Confirmation

Because the selected candidate was eligible, a separate full-workload paired
confirmation was run with five baseline trials and five selected trials. The
order was deterministic and paired. The seed, world size, learning rate,
effective batch size, and workload identity were unchanged.

| Metric | Baseline | Selected |
| --- | ---: | ---: |
| trial count | 5 | 5 |
| median `samples_per_second` | 41636.929876 | 83556.804262 |
| min `samples_per_second` | 40074.990324 | 73981.083964 |
| max `samples_per_second` | 43252.478230 | 87875.513360 |
| variability percent | 7.631417 | 16.628723 |

```text
observed_relative_difference_percent: 100.679552
noise_band_percent: 5.0
noise_classification: improved
```

The selected full-workload distribution was more variable than the bounded
slice distribution. This does not invalidate the guard result, but it limits
claim strength to this local workload and protocol.

## Drift And Safety

Arm RecommendationBundles were validated on Windows only through comparison
and dry-run apply. No Arm recommendation launched a Windows workload.

```text
c7g_bundle_id: 555238aaec70532e69c538d6
c7g_on_windows_status: hard_mismatch
c7g_dry_run_status: dry_run_failed

t4g_bundle_id: 1bce3e25a12db1db3eaccefb
t4g_on_windows_status: hard_mismatch
t4g_dry_run_status: dry_run_failed

windows_bundle_id: c61f327d271288d01437d435
windows_dry_run_status: dry_run_passed
soft_memory_drift_retune_required: false
```

Hard mismatch fields for the Arm bundles included architecture and logical CPU
count. Memory drift was validated as a soft mismatch that does not require
retuning by itself.

## Cleanup

```text
process_scope_apply: true
lock_state_after_workflow: rolled_back
lease_state_after_workflow: rolled_back
active_lease_or_lock_records: 0
residual_processes: 0
artifact_manifest_revalidation_status: passed
```

## Limitations

- The result is a synthetic CPU-only PyTorch workload on one Windows x86_64
  controller.
- The validation does not compare Windows with Arm and does not imply that the
  selected Windows configuration is portable to C7g or T4g.
- PyTorch emitted a non-fatal missing-NumPy warning; NumPy was intentionally not
  installed because the public workload does not require it.
- The full-workload selected candidate had higher variability than the bounded
  slice result, so the recommendation remains local, preliminary, and
  protocol-bound.
