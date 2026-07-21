# Phase 21C2A C7g 4K Boot Isolation

This document records the local preparation boundary for Phase 21C2A. It does
not claim that a real C7g reboot, boot-parameter experiment, or benchmark has
run yet.

## Source Boundary

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21c1_commit: e26af1e59ddd267a316ab826cb367fbbd74ff025
phase21c2a_initial_commit: e26af1e59ddd267a316ab826cb367fbbd74ff025
phase21c1_remote_branch: origin/validation/phase21c0-boot-readiness
phase21c2a_branch: validation/phase21c2a-4k-boot-isolation
pr6_status: open draft
pr6_head_branch: feature/linux-system-kernel-autotuning
pr6_head_commit: 09e5809d05dda933c7c27b88b1222e067674090e
```

The worktree was created from the exact Phase 21C1 evidence commit. Phase 21C2A
local preparation does not modify v0.2.0 release artifacts or competition
submission claims.

## Local Scope

Phase 21C2A prepares guarded one-time 4K boot-parameter experiments for the
currently installed known-good AWS 4K kernel:

```text
expected_kernel: 6.17.0-1019-aws
expected_page_size: 4096
```

Profiles:

| profile | label | appended parameters |
| --- | --- | --- |
| G0 | `original-4k-default` | none |
| G1 | `4k-nohz-rcu-irq` | `nohz_full=2-7`, `rcu_nocbs=2-7`, `irqaffinity=0-1` |
| G2 | `4k-managed-irq-isolation` | G1 parameters plus `isolcpus=managed_irq,2-7` |
| G3 | `4k-domain-isolation` | G1 parameters plus `isolcpus=domain,managed_irq,2-7` |

G3 remains the most invasive 4K isolation profile and must run last in any
future real execution.

## Boot Transaction Model

Experimental profiles are represented as an isolated temporary GRUB custom
entry plan. The plan records:

- exact current `vmlinuz` path
- exact current `initrd` path
- preserved original kernel arguments
- exact current `root=` argument
- allowlisted appended parameters only
- unique temporary entry ID
- explicit cleanup and removal verification
- mandatory return to `original-4k-default`
- permanent default refusal
- direct `grub.cfg` edit refusal
- one reboot maximum per experimental transaction

Phase 21C2A local preparation still leaves real reboot authorization as an
external boundary. Fixture experimental results therefore remain blocked until
the user explicitly starts the target and the remote preflight revalidates
recovery, source, and host identity.

## Baseline Identities

Phase 21C2A separates causal and capacity comparisons:

| identity | kernel/profile | workload CPUs | threads | role |
| --- | --- | --- | --- | --- |
| E1 | G0 `original-4k-default` | `0-7` | 8 | full-capacity baseline |
| E2 | G0 `original-4k-default` | `2-7` | 6 | matched causal baseline |
| E3 | G1/G2/G3 experimental | `2-7` | 6 | experimental six-CPU profile |

Only E2 versus E3 can support a boot-isolation percentage. E1 is a secondary
product-capacity tradeoff and must not be used as the causal denominator.

## Local Guard Coverage

The local fixture tests cover:

- G0, G1, G2, and G3 plan generation
- kernel argument preservation
- allowlisted boot parameters only
- refusal of security-disabling parameters
- duplicate and conflicting kernel parameter refusal
- malformed CPU list refusal
- at least two housekeeping CPUs
- workload CPUs fixed to `2-7`
- command-line length guard
- required `CONFIG_NO_HZ_FULL`, `CONFIG_RCU_NOCB_CPU`, and `CONFIG_CPU_ISOLATION`
- temporary-entry removal and non-reuse requirements
- mandatory return to G0
- experimental-to-experimental chain refusal
- permanent default refusal
- maximum reboot count
- Serial Console policy preservation
- E1/E2/E3 baseline identities
- thread count greater than cpuset refusal
- no causal percentage claim against mismatched CPU counts
- fixture health failure and pending cleanup failure handling

## Current Status

```text
phase21c2a_local_preparation: passed
ready_for_manual_c7g_start: true
kernel_or_boot_state_modified: false
aws_accessed: false
real_reboot_performed: false
performance_improvement_verified: false
latency_or_noise_improvement_verified: false
original_4k_state_restored: not_applicable_local_preparation
permanent_default_unchanged: not_applicable_local_preparation
ready_for_phase21c2b_64k_validation: false
```

The next boundary is manual C7g start by the user, followed by strict remote
preflight, two read-only host inventories, G0 E1/E2 measurement, one-time G1,
G2, and G3 transactions, artifact validation, and final restoration to the
original 4K state.
