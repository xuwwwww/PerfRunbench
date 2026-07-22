# PerfRunbench Execution Plan

PerfRunbench's current default workflow is:

```text
inspect environment
-> inspect workload runtime
-> establish a baseline
-> run bounded candidate search
-> confirm with paired measurements
-> generate a Performance Passport
-> validate and safely apply an eligible RecommendationBundle
```

## Completed Validation

- Phase 14B validated the `pytorch-training` adapter on a real Linux Arm64 C7g
  target.
- Phase 15A validated low-resource behavior on a constrained T4g.small target.
- Phase 15B validated real PyTorch tuning on that T4g target without comparing
  numbers to C7g.
- Phase 15C validated the golden-path CLI and `pytorch-training` adapter on
  Windows x86_64, including cache miss, cache reuse, cross-platform hard-drift
  refusal, process-scoped apply, and cleanup.

## Phase 15C Boundary

Phase 15C used a local Windows x86_64 controller only. AWS was not accessed or
controlled. The measured numbers are local synthetic-workload validation data
and are not Arm performance claims.

The only product change made during Phase 15C was a focused correctness fix for
hex CPU-affinity mask canonicalization. No model, dataset, benchmark backend,
cloud automation, or system-scoped tuning feature was added.

## Completed Release Scope

Phase 16 through Phase 18 completed the v0.2 release validation, GitHub
Release packaging, competition submission materials, and final demo rehearsal.
The v0.2.0 tag and competition submission claims must not be modified
retrospectively.

## Phase 19: CPU-Aware Native Build Autotuning On Arm

Phase 19 makes CPU-specific native compilation a first-class tuning dimension
for Arm inference workloads:

```text
inspect Arm target
-> lock a pinned llama.cpp source commit
-> inspect compiler and CPU feature evidence
-> create allowlisted build candidates
-> verify build and backend evidence
-> compare builds with a fixed benchmark protocol
-> run bounded runtime search on the selected build
-> confirm the complete configuration
-> generate and verify a Performance Passport
-> guard native-binary recommendation reuse
```

Boundaries:

- Use a single existing low-resource T4g.small target for bounded validation.
- Do not compare T4g numbers directly with C7g evidence.
- Do not access the stopped paid C7g target.
- Do not download new models or datasets.
- Do not add BootTrace research or model qualification logic.
- Do not claim global Arm speedup from one T4g.small session.

Expected commits:

```text
feat: add CPU-aware native build profiles
feat: guard native binary portability and backend evidence
docs: record T4g CPU-aware build tuning
```

Current Phase 19 real T4g outcome:

```text
insufficient_evidence
```

The target produced build artifacts for `upstream-default` and
`native-current-cpu`, but no suitable existing small GGUF model was available.
The run therefore stopped before benchmark comparison, runtime search,
Performance Passport, or recommendation apply. This is a guard outcome, not a
performance result.

## Phase 20: C7g Build Backend And Runtime Tuning

Phase 20 validates the CPU-aware build-profile implementation on a real
C7g / Neoverse-V1 target with the existing Qwen2.5-7B Q4_K_M model and pinned
`llama.cpp` source `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.

Completed:

- add the explicit `explicit-neoverse-v1` build profile
- build `upstream-default`, `explicit-neoverse-v1`, `native-kleidiai`, and
  `native-kleidiai-lto`
- preserve `portable-reference` as `build_failed`
- reject KleidiAI-requested profiles from eligibility when runtime backend
  evidence is not proven
- run fixed build-profile comparison
- run a six-candidate bounded runtime search
- run secondary `llama-server` TTFT/E2E measurement without BootTrace quality
  eligibility
- generate and verify a Performance Passport
- validate 77 artifact manifests on-target and again after download

Current Phase 20 real C7g outcome:

```text
build_profile_recommendation: inconclusive
runtime_search_recommendation: inconclusive
performance_improvement_verified: false
phase20_c7g_build_backend_runtime_tuning: passed
```

The best evidence-valid build profile was `explicit-neoverse-v1`, but its
decode median (`17.78 tok/s`) was only `0.395257%` above `upstream-default`
(`17.71 tok/s`), inside the configured 5% noise band. Runtime search selected
the baseline configuration with a confirmed decode median of `17.72 tok/s`.

## Phase 21A: Host System And Kernel Profile Framework

Phase 21A is local implementation, fake-root testing, and documentation only.
AWS is not accessed and no remote host is modified.

Git integration record:

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase20_commit: 9bc71d034f1cb7e986f7bf87a222335191e2c712
phase21a_initial_commit: 9bc71d034f1cb7e986f7bf87a222335191e2c712
phase20_pr_status: not_created_auth_unavailable
```

Phase 20 branch `validation/phase20-c7g-build-backend` was pushed without
force. GitHub PR creation was attempted through `gh` and the GitHub connector;
`gh` lacked authentication and the connector returned `403 Resource not
accessible by integration`.

Phase 21A scope:

- versioned host state snapshots
- capability-driven safe profile catalog
- reversible fake-root leases and rollback
- boot-kernel profile planning without bootloader writes
- system benchmark protocol separation

## Phase 21B: C7g Reversible Linux System Validation

Phase 21B validated a bounded real-host subset of the Phase 21A framework on a
C7g / Neoverse-V1 target using the fixed Qwen2.5-7B Q4_K_M
`upstream-default` `llama-bench` protocol.

Completed:

- verified controller and target source match
- captured immutable original snapshots with a private recovery marker digest
- rehearsed a fixture watchdog rollback
- executed `baseline-current`, `thp-never`,
  `allocator-glibc-bounded-arenas`, `scheduler-batch`,
  `affinity-cpus-1-7`, `vm-system-conservative`, and `thp-always`
- rolled back every host-level profile and verified final normalized host
  state matched the original
- validated artifacts remotely and after download

Current Phase 21B outcome:

```text
phase21b_c7g_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
```

The best positive observed candidate was `thp-always` at `+0.344%` versus the
baseline, inside the configured 5% noise band. It is not a recommendation.

## Phase 21B2: Remaining C7g Reversible System Validation

Phase 21B2 completed the remaining safe reversible real-host families that
were intentionally left out of Phase 21B:

```text
cpuset partitions
IRQ/workqueue isolation
mmap/no-mmap and prefault protocols
THP exercised with mmap evidence
storage cold-start read-ahead and scheduler protocols
HugeTLB consumer qualification
jemalloc/mimalloc availability
cpuidle capability
perf-counter availability without security weakening
boot-validation readiness audit
```

Current Phase 21B2 outcome:

```text
phase21b2_remaining_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
all_reversible_families_exhausted: true
ready_for_phase21c_boot_kernel_validation: false
```

The strongest positive steady-state decode result was `no-mmap` at `+3.014%`,
inside the configured 5% noise band. `thp-always-mmap` exercised THP pages but
stayed at `+1.630%`. Reduced-CPU cpuset and IRQ/workqueue isolation regressed
sharply. Cold-start storage and prefault evidence was preserved as
protocol-specific data and was not ranked against the steady-state baseline.

One real-host correctness fix was required:

```text
88a2cee fix: normalize bracketed block scheduler mode
```

Phase 21C boot/kernel changes remain future work. Phase 21B2 performed only a
read-only boot-readiness audit.

## Phase 21C0: C7g Boot And Kernel Validation Readiness

Phase 21C0 prepares safe future 4K/64K kernel validation without accessing AWS
or modifying any real host.

Git consolidation:

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21b2_commit: 09e5809d05dda933c7c27b88b1222e067674090e
pr6_head_before: 27c73e69ada9d3d5ba501de2151111f30c5d0bb3
pr6_head_after: 09e5809d05dda933c7c27b88b1222e067674090e
phase21c0_initial_commit: 09e5809d05dda933c7c27b88b1222e067674090e
```

Implemented local-only readiness:

```text
BootInventory
BootRecoveryCapability
KernelCandidate
KernelConfigCapability
OneTimeBootPlan
PostBootHealthPolicy
BootRollbackPlan
BootExperimentResult
```

New read-only CLI:

```text
autotuneai system boot-doctor --json
autotuneai system kernel-inventory --json
autotuneai system boot-plan --profile <profile> --inventory <inventory.json>
autotuneai system boot-plan verify --plan <plan.json>
autotuneai system recovery-plan --inventory <inventory.json>
```

Phase 21C0 profile order:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
G4 64k-default
G5 64k-nohz-rcu-irq
G6 64k-managed-irq-isolation
```

The private read-only remote audit completed on a C7g / Neoverse-V1 target
after a focused metadata parser fix:

```text
0280f6d fix: discover AWS 64K kernel metadata from apt
phase21c0_remote_readonly_audit: passed
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
fallback_kernel_verified: true
official_64k_candidate_found: true
official_64k_candidate_state: metadata_qualified_not_installed
downloaded_artifact_validation: 599/599 passed
```

Phase 21C0 does not install kernels, write bootloader state, modify GRUB or
EFI, generate initramfs, reboot, or control AWS.

## Phase 21C1: C7g Same-Kernel One-Time Boot Rehearsal

Phase 21C1 converts the Phase 21C0 boot-readiness blockers into one bounded
same-kernel boot transaction. It is recovery and bootloader-safety evidence,
not performance evidence.

Completed:

```text
recovery_snapshot_or_ami_verified: true
serial_or_out_of_band_recovery_verified: true
serial_console_interactive_login_verified: true
serial_console_sudo_verified: true
serial_getty_persists_across_boot: true
exactly_one_guest_reboot_performed: true
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
selected_one_time_entry: current known-good non-recovery 4K kernel
post_boot_kernel_unchanged: true
post_boot_page_size_unchanged: true
kernel_command_line_unchanged: true
permanent_default_unchanged: true
host_healthy_after_reboot: true
downloaded_artifact_validation: 30/30 passed
phase21c1_one_time_boot_rehearsal: passed
```

GRUB left an empty `next_entry=` key after consuming the one-time value. The
run cleared that empty key with `grub-editenv - unset next_entry` and verified
the final GRUB environment list returned to the pre-arm state.

Phase 21C1 did not install, remove, or boot a 64K kernel; did not alter EFI
BootOrder; did not change the permanent GRUB default; and did not run
performance benchmarks.

The next Phase 21C step may attempt a real reboot-required kernel/profile only
one profile at a time, starting from the restored known-good 4K state and using
the validated serial-console recovery path.

## Phase 21C2A: Real 4K Boot Isolation Validation

Phase 21C2A executed guarded 4K boot-parameter experiments on the existing C7g
/ Neoverse-V1 target with the installed known-good AWS 4K kernel. It did not
install or remove kernel packages, did not boot a 64K kernel, did not change
EFI BootOrder, and did not change the permanent GRUB default.

Git source record:

```text
phase21c2a_branch: validation/phase21c2a-4k-boot-isolation
source_commit: 4e5a11bc10906649bbe756a3094488c7d30824b9
```

Executed 4K profiles:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
```

The implementation used temporary GRUB custom entries, boot-parameter guards,
mandatory return to G0 before each next profile, and E1/E2/E3 comparison
identities:

```text
E1 full-capacity baseline: G0 CPUs 0-7, threads 8
E2 matched causal baseline: G0 CPUs 2-7, threads 6
E3 experimental profile: G1/G2/G3 CPUs 2-7, threads 6
```

Only E2 versus E3 can support a causal boot-isolation percentage. E1 is a
secondary product-capacity tradeoff.

Result:

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
entry, cleanup passed, the harness was corrected to preserve original GRUB
kernel/initrd paths, and corrected G1/G2/G3 transactions restarted from
verified G0. The failed pre-measurement boot is not included in performance
comparisons.

## Phase 21C2B0: Local Preparation For C7g 64K Kernel Validation

Phase 21C2B0 is local implementation, fixture testing, package transaction
planning, boot-plan validation, recovery planning, and documentation only.

Boundary:

```text
aws_accessed: false
kernel_or_boot_state_modified: false
kernel_installed: false
grub_or_efi_modified: false
machine_rebooted: false
```

Implemented:

- exact generated GRUB menu-entry parser for kernel and initrd paths
- hard rejection of guessed `/boot/vmlinuz-*` temporary-entry paths
- official AWS ARM64 64K package transaction plan model
- apt simulation safety policy
- resolved-default safety distinction from configuration-text equality
- guarded G0 safety-pin plan model
- G4/G5/G6/G7 experiment plan model
- H1/H2/H3/H4/H5 comparison identities
- exact package cleanup plan model
- plan-only CLI commands for package, default, experiment, and cleanup plans

Phase 21C2B0 prepares future real validation. It does not authorize or perform
kernel installation, GRUB writes, initramfs generation, EFI changes, reboot, or
AWS access.

Next real Phase 21C2B remains blocked until the user manually starts the C7g
target and supplies private connection details. The real phase must use fresh
same-session 4K and 64K baselines and must not reuse Phase 21C2A measurements
as denominators.

## Phase 21C2B: C7g 4K Versus 64K Kernel Validation

Phase 21C2B executed the guarded real 64K validation on the C7g target. It
installed the official Ubuntu AWS ARM64 64K kernel package transaction, used a
temporary G0 default safety pin, booted G4/G5/G6/G7 through one-time GRUB
selection, restored G0 after every profile, and removed exactly the introduced
packages.

```text
branch: validation/phase21c2b-c7g-64k-real
final_source_commit: f2175f69e37e2d826237d395a04b1c04db40e786
original_kernel: 6.17.0-1019-aws
candidate_64k_kernel: 6.17.0-1019-aws-64k
artifact_manifest_revalidation: 621/621 passed
```

Results:

```text
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
ready_to_stop_c7g: true
```

Two GRUB renderer fixes were committed during the phase after defects were
detected before the affected experimental reboot:

```text
10a2f02 fix: preserve generated GRUB conditionals in 64K boot entries
f2175f6 fix: parse tab-separated GRUB linux commands
```
