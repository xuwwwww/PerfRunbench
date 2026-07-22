# PerfRunbench Validation Matrix

This matrix separates portability and guard evidence from performance claims.
Numbers from different targets or protocols must not be combined.

| Phase | Target | Workload | Purpose | Result boundary | Status |
| --- | --- | --- | --- | --- | --- |
| 14B | Linux Arm64 C7g / Neoverse-V1 | Public synthetic PyTorch training | Real Arm training-adapter workflow and guarded apply | Single Arm target, synthetic workload | passed |
| 15A | Linux Arm64 T4g.small / Neoverse-N1 class | Dependency-free fixture, PyTorch skipped when unsafe | Low-resource guard and portability evidence | No performance claim | passed |
| 15B | Linux Arm64 T4g.small / Neoverse-N1 class | Public synthetic PyTorch training | Real low-resource Arm PyTorch workflow | Do not compare with C7g | passed |
| 15C | Windows x86_64 controller | Public synthetic PyTorch training | Local CLI, cache, drift, apply, and portability validation | Do not compare with Arm | passed |
| 16 | Windows local clean venv plus GitHub Actions matrix | Dependency-free public fixtures | Release-candidate installed-wheel usability | Packaging and CLI usability, not performance | pending final CI |
| 19 | Linux Arm64 T4g.small / Neoverse-N1 class | No suitable existing model available; build evidence only | CPU-aware native build tuning guard validation | Build artifacts collected, but no benchmark or recommendation claim | insufficient evidence |
| 20 | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp | CPU-aware build/backend evidence, bounded runtime search, server timing, Passport, and artifact validation | Real measured workflow validation; build and runtime improvements stayed inside the 5% noise band | passed |
| 21A | Local fake Linux roots | Synthetic `/proc`, `/sys`, cgroup, IRQ, block, and boot fixtures | Host system and kernel profile framework, leases, rollback, and boot-plan safety | Planning and fake E2E only; no AWS and no real host mutation | passed |
| 21B | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp upstream-default binary | Reversible Linux host-system apply, measurement, rollback, cleanup, and artifact validation | Real measured guard validation; no positive candidate cleared the 5% noise band | passed |
| 21B2 | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp upstream-default binary | Remaining reversible system families: cpuset, IRQ/workqueue, mapping, THP exercise, storage cold-start, perf, and boot-readiness audit | Real measured guard validation; no positive same-protocol candidate cleared the 5% noise band | passed |
| 21C0 | Local fake Linux roots plus read-only Linux Arm64 C7g / Neoverse-V1 audit | Synthetic boot fixtures and read-only boot/kernel inventory | C7g 4K/64K boot-readiness model, one-time plans, recovery blockers, official 64K package metadata, and future protocol plan | Planning and read-only inventory only; no kernel install, no bootloader write, no reboot | passed |
| 21C1 | Linux Arm64 C7g / Neoverse-V1 | No benchmark workload | Same-kernel GRUB one-time boot rehearsal with serial-console recovery validation | Reboot safety evidence only; no kernel change and no performance claim | passed |
| 21C2A | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp llama-bench and llama-server | 4K one-time boot-parameter isolation, rollback, cleanup, matched baseline measurement, and artifact validation | Real measured guard validation; G1/G2 stayed inside the 5% noise band and G3 regressed | passed |
| 21C2B | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp llama-bench and llama-server | Official Ubuntu AWS ARM64 64K kernel transaction, G4/G5/G6/G7 one-time boots, matched measurements, package cleanup, and artifact validation | Real measured guard validation; 64K page-size and isolation differences stayed inside the 5% noise band except G7, which regressed | passed |

## Current Evidence Boundaries

- Arm RecommendationBundles do not apply on Windows x86_64 because architecture
  and CPU-count hard fingerprints differ.
- Windows RecommendationBundles are bound to the tested Windows x86_64
  environment and are not portable to Arm targets.
- Fixture, smoke, and synthetic-training measurements are reproducibility and
  guard evidence unless a document explicitly marks them as a target-specific
  measured result.
- `run --best` never retunes. It requires an existing eligible cache entry and
  refuses to launch when no usable recommendation is available.
- `run --auto` may retune on cache miss or hard drift according to policy; exact
  cache reuse does not create a new tuning result.
- CPU-specific binary reuse requires native-binary portability evidence in
  addition to workload and environment identity. Neoverse-N1 and Neoverse-V1
  CPU-model-bound binaries are not interchangeable by default.

## Phase 15C Summary

```text
source_commit: 5bf2a0d217283e8da2ed4954a9be65a4be4ed954
target: Windows x86_64 / 8 logical CPUs
torch_version: 2.13.0+cpu
candidate_count_total_including_baseline: 6
recommendation_bundle_id: c61f327d271288d01437d435
bounded_slice_status: recommended
artifact_manifest_revalidation_status: passed
cleanup_status: passed
```

Phase 15C also validated the CPU-affinity mask canonicalization fix for
Windows-style hex masks such as `0xff`.

## Phase 16 Release Candidate Scope

Phase 16 validates that a fresh user can install the built distribution,
initialize a config, check readiness with `autotuneai doctor`, run bounded
tuning through `run --auto`, reuse recommendations, and inspect the cache
without knowing internal Passport, RecommendationBundle, or cache paths.

Required CI platforms:

```text
windows-latest
ubuntu-latest
ubuntu-24.04-arm
```

The required CI matrix uses dependency-free fixtures and must not install
PyTorch, download models/datasets, access AWS, or turn fixture evidence into
measured performance evidence.

## Phase 20 C7g Build/Backend Summary

Phase 20 returned to a user-provisioned C7g / Neoverse-V1 target with the
existing Qwen2.5-7B Q4_K_M model and pinned `llama.cpp` source commit
`47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.

The validation completed build-profile comparison, bounded runtime search,
secondary `llama-server` measurement, Performance Passport generation and
verification, remote artifact revalidation, and downloaded artifact
revalidation.

Outcome:

```text
build_profile_recommendation: inconclusive
runtime_search_recommendation: inconclusive
selected_runtime_candidate: baseline
KleidiAI_status: requested profiles rejected unless runtime backend evidence exists
Performance_Passport: verification passed, claim_eligibility_state ineligible_provenance
artifact_revalidation: 77/77 passed remotely and after download
performance_improvement_verified: false
```

## Phase 21A System/Kernel Framework Summary

Phase 21A adds a guarded local framework for future Linux system and kernel
autotuning:

```text
AWS_access: false
real_host_mutation: false
snapshot_model: HostStateSnapshot
profile_catalog: capability-driven safe-system
reversible_apply: fake-root only
rollback: reverse-order exact restoration
boot_profiles: planning only, no GRUB writes
protocol_separation: enforced
dirty_host_recommendation: ineligible
```

This phase does not provide a performance result. It prepares future C7g
validation for reversible system profiles and reboot-required kernel profiles.

## Phase 21B C7g Reversible System Summary

Phase 21B applied bounded reversible real-host profiles one at a time from the
verified original state, measured the fixed `llama-bench` steady-state decode
protocol, and rolled each profile back before the next candidate.

Outcome:

```text
candidate_count: 7
baseline_decode_median_tok_s: 17.44
best_positive_candidate: thp-always at 17.50 tok/s
best_positive_delta_percent: 0.344
noise_band_percent: 5.0
performance_improvement_verified: false
host_restored_to_original_state: true
remote_artifacts_checked: 712
local_artifacts_checked: 712
```

`affinity-cpus-1-7` was a measured regression at `-10.493%`, because it
reduced the workload CPU set and thread count from the eight-CPU baseline.

## Phase 21B2 C7g Remaining System Summary

Phase 21B2 completed the remaining reversible C7g families and preserved
protocol boundaries.

Outcome:

```text
candidate_count: 14
baseline_decode_median_tok_s: 16.26
best_positive_same_protocol_candidate: no-mmap at 16.75 tok/s
best_positive_same_protocol_delta_percent: 3.014
noise_band_percent: 5.0
performance_improvement_verified: false
host_restored_to_original_state: true
all_reversible_families_exhausted: true
remote_artifacts_checked: 327
local_artifacts_checked: 327
```

`thp-always-mmap` exercised THP pages but remained inside the guard at
`+1.630%`. `cpuset-1h7w`, `cpuset-2h6w`, and
`cpuset-1h7w-irq-workqueue` regressed sharply. Storage cold-start and
prefault evidence was labeled `protocol_baseline_missing` instead of being
ranked against steady-state decode throughput.

## Phase 21C0 Boot Readiness Summary

Phase 21C0 adds local read-only planning for future reboot-required kernel
profiles and validates the inventory path with a read-only C7g audit.

Outcome:

```text
phase21c0_local_boot_readiness: passed
phase21c0_remote_readonly_audit: passed
remote_commit: 0280f6d70b88dade575e4db27299ca3e20b2433f
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
bootloader: grub_uefi
fallback_kernel_verified: true
official_64k_candidate_found: true
official_64k_candidate_state: metadata_qualified_not_installed
ready_for_phase21c_real_boot_validation: false
kernel_or_boot_state_modified: false
real_phase21c_started: false
downloaded_artifact_validation: 599/599 passed
```

The fixture validation covers:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
G4 64k-default
G5 64k-nohz-rcu-irq
G6 64k-managed-irq-isolation
```

No performance claim exists for Phase 21C0.

The remote read-only audit required one focused inventory fix:

```text
0280f6d fix: discover AWS 64K kernel metadata from apt
```

## Phase 21C1 Same-Kernel Boot Rehearsal Summary

Phase 21C1 performed exactly one authorized guest OS reboot through a GRUB
one-time entry that resolved to the already running known-good kernel.

Outcome:

```text
phase21c1_one_time_boot_rehearsal: passed
recovery_snapshot_verified: true
serial_or_out_of_band_recovery_verified: true
serial_console_interactive_login_verified: true
serial_console_sudo_verified: true
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
post_boot_kernel_unchanged: true
post_boot_page_size_unchanged: true
kernel_command_line_unchanged: true
permanent_default_unchanged: true
host_healthy_after_reboot: true
downloaded_artifact_validation: 30/30 passed
ready_for_phase21c_real_boot_validation: true
performance_improvement_verified: false
```

GRUB consumed the one-time entry value but left an empty `next_entry=` key in
`grubenv`; the key was removed with `grub-editenv - unset next_entry` and the
final GRUB environment list matched the pre-arm state.

No 64K kernel was installed or booted, no permanent boot default was changed,
and no benchmark was run.

## Phase 21C2A 4K Boot Isolation Validation Summary

Phase 21C2A executed guarded one-time 4K boot-parameter experiments on the
existing C7g / Neoverse-V1 target:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
```

The validation used isolated temporary GRUB custom entries, explicit removal
verification, permanent-default refusal, mandatory return to G0 before another
profile, and independent artifact revalidation.

Comparison identities:

```text
E1 full-capacity baseline: G0 CPUs 0-7, threads 8
E2 matched causal baseline: G0 CPUs 2-7, threads 6
E3 experimental profile: G1/G2/G3 CPUs 2-7, threads 6
```

Only E2 versus E3 was used for causal boot-isolation classification. E1 remains
a secondary product-capacity tradeoff.

Outcome:

```text
G0_E2_decode_median: 13.52 tok/s
G1_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G2_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G3_E3_decode_median: 12.59 tok/s, -6.88%, regressed
remote_artifact_revalidation: 410/410 passed
downloaded_artifact_revalidation: 410/410 passed
original_4k_state_restored: true
permanent_default_unchanged: true
performance_improvement_verified: false
latency_or_noise_improvement_verified: false
```

The first G1 attempt exposed a temporary-entry kernel-path generation defect
before measurement. Recovery used Serial Console to select the original Ubuntu
entry, the defect was corrected, cleanup passed, and corrected G1/G2/G3
transactions restarted from verified G0. The failed pre-measurement boot is not
included in performance comparisons.

## Phase 21C2B0 64K Local Preparation Summary

Phase 21C2B0 prepares future official AWS ARM64 64K validation without remote
access or host mutation.

Outcome:

```text
phase21c2b0_64k_local_preparation: passed
g1_grub_path_regression_guarded: true
package_transaction_plan_ready: true
safe_g0_default_plan_ready: true
package_cleanup_plan_ready: true
ready_for_manual_c7g_start: true
kernel_or_boot_state_modified: false
aws_accessed: false
```

## Phase 21C2B C7g 64K Kernel Validation

Phase 21C2B performed the guarded real 4K-versus-64K C7g validation described
in `docs/results/PHASE_21C2B_C7G_64K_KERNEL_VALIDATION.md`.

```text
final_source_commit: f2175f69e37e2d826237d395a04b1c04db40e786
original_kernel: 6.17.0-1019-aws
candidate_64k_kernel: 6.17.0-1019-aws-64k
H1_4k_full_decode_median: 17.46 tok/s
H3_64k_full_decode_median: 17.55 tok/s
H2_4k_matched_decode_median: 13.63 tok/s
H4_64k_matched_decode_median: 13.64 tok/s
G5_H5_decode_median: 13.61 tok/s
G6_H5_decode_median: 13.67 tok/s
G7_H5_decode_median: 12.63 tok/s
page_size_performance_improvement_verified: false
64k_isolation_improvement_verified: false
original_4k_state_restored: true
package_state_restored: true
permanent_default_restored: true
ready_to_stop_c7g: true
```

Coverage:

```text
generated GRUB path parsing
guessed kernel path rejection
official 64K package transaction planning
apt simulation safety checks
resolved-default safety checks
G0 safety-pin planning
G4/G5/G6/G7 experiment planning
H1/H2/H3/H4/H5 comparison identities
exact package cleanup planning
plan-only CLI fake E2Es
```

This is not a 64K performance result. A future real phase must start from a
manually started C7g target and create fresh same-session 4K and 64K baselines.
