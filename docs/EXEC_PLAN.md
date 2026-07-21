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

## Phase 21C2A: Real 4K Boot Isolation Local Preparation

Phase 21C2A begins from the exact Phase 21C1 evidence commit and prepares the
guarded 4K boot-parameter experiment locally. AWS remains untouched during this
preparation boundary.

Git source record:

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21c1_commit: e26af1e59ddd267a316ab826cb367fbbd74ff025
phase21c2a_initial_commit: e26af1e59ddd267a316ab826cb367fbbd74ff025
phase21c1_remote_branch: origin/validation/phase21c0-boot-readiness
phase21c2a_branch: validation/phase21c2a-4k-boot-isolation
pr6_status: open draft
```

Prepared 4K profiles:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
```

The local implementation adds temporary-entry planning, boot-parameter guards,
and E1/E2/E3 comparison identities:

```text
E1 full-capacity baseline: G0 CPUs 0-7, threads 8
E2 matched causal baseline: G0 CPUs 2-7, threads 6
E3 experimental profile: G1/G2/G3 CPUs 2-7, threads 6
```

Only E2 versus E3 can support a causal boot-isolation percentage. E1 is a
secondary product-capacity tradeoff.

Current boundary:

```text
phase21c2a_local_preparation: passed
ready_for_manual_c7g_start: true
kernel_or_boot_state_modified: false
aws_accessed: false
```

Real execution starts only after the user manually starts C7g and provides the
new private connection details. The next step must revalidate controller and
remote source equality, two read-only host inventories, recovery markers,
Serial Console access, fallback kernel, original 4K command line, and absence
of stale temporary entries before any one-time boot profile is armed.
