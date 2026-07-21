# Phase 21C0 C7g Boot And Kernel Validation Readiness

Phase 21C0 is local implementation, fixture validation, and public planning for
future reboot-required C7g kernel experiments. It does not begin Phase 21C.

## Scope

```text
AWS_accessed: false
kernel_or_boot_state_modified: false
real_reboot_performed: false
kernel_installed: false
GRUB_or_EFI_written: false
phase21c_real_boot_changes_started: false
```

The C7g and T4g instances are outside this local stage. Future read-only remote
audit may start only after the user manually starts C7g and provides connection
details.

## Git Consolidation Record

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21b2_commit: 09e5809d05dda933c7c27b88b1222e067674090e
pr6_head_before: 27c73e69ada9d3d5ba501de2151111f30c5d0bb3
pr6_head_after: 09e5809d05dda933c7c27b88b1222e067674090e
phase21c0_initial_commit: 09e5809d05dda933c7c27b88b1222e067674090e
pr6_state: open draft
pr6_head_branch: feature/linux-system-kernel-autotuning
phase21c0_branch: validation/phase21c0-boot-readiness
```

The PR #6 head branch was fast-forwarded without force push to include Phase
20, Phase 21A, Phase 21B, and Phase 21B2 work before creating the Phase 21C0
branch.

## Why Real Phase 21C Remains Blocked

Phase 21B2 intentionally ended with:

```text
ready_for_phase21c_boot_kernel_validation: false
```

Phase 21C0 replaces that generic flag with an exact blocker matrix. A future
real boot run may proceed only after every relevant blocker is clear in private
evidence and the user gives explicit authorization for that real reboot
sequence.

| blocker | meaning |
| --- | --- |
| `missing_recovery_snapshot` | No recent private EBS snapshot or AMI marker is recorded. |
| `missing_snapshot_marker` | No local marker proves the recovery snapshot was intentionally prepared. |
| `serial_console_not_verified` | EC2 Serial Console or equivalent out-of-band recovery is not verified. |
| `no_out_of_band_recovery` | No volume detach/repair or management recovery path is verified. |
| `fallback_kernel_missing` | No separate installed fallback kernel entry is available. |
| `fallback_kernel_unverified` | Fallback kernel files, package state, config, initrd, or modules are incomplete. |
| `bootloader_unknown` | GRUB, systemd-boot, direct EFI stub, or another boot arrangement was not identified. |
| `one_time_boot_unavailable` | The bootloader cannot prove next-boot-only selection and clearing behavior. |
| `kernel_package_candidate_unknown` | Official AWS ARM64 64K candidate metadata was not identified. |
| `kernel_config_support_unknown` | Required kernel config evidence is unavailable and remains unknown. |
| `insufficient_boot_partition_space` | `/boot` free space is unknown or below the planning estimate. |
| `secure_boot_or_lockdown_unknown` | Secure Boot or kernel lockdown state cannot be classified. |
| `initramfs_generation_unknown` | Initramfs generation capability is unknown; Phase 21C0 still does not run it. |
| `ssh_post_boot_check_missing` | Post-boot SSH or authorized management health check is not defined. |
| `rollback_command_missing` | Rollback commands are not documented for the identified boot mechanism. |
| `user_authorization_missing` | Real boot changes require explicit user authorization at execution time. |
| `other` | A specific non-standard blocker is preserved with evidence text. |

## Boot Transaction Model

Every future boot experiment must follow one transaction:

```text
read-only inventory
-> recovery verification
-> package qualification
-> boot entry generation
-> generated-file validation
-> one-time next boot selection
-> reboot authorization boundary
-> post-boot health check
-> benchmark
-> one-time rollback boot
-> original-state verification
```

Phase 21C0 adds versioned planning schemas:

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

Every top-level serialized structure includes `schema_version`.

## Read-Only Inventory

The new read-only inventory collects, when available:

```text
uname / kernel release
base page size
/proc/cmdline and command-line digest
/proc/config.gz or /boot/config-<kernel>
/etc/os-release
dpkg kernel inventory
/boot vmlinuz/initrd/config inventory
/lib/modules inventory
filesystem free-space estimates
GRUB environment and menu entries
systemd-boot entries
direct EFI-stub hints
EFI state
initramfs inventory
Secure Boot and lockdown state
private fixture recovery markers
```

Unknown values remain `unknown` or `not_available`; they are not converted into
unsupported states.

## Planned Boot Profiles

| profile | page size | purpose | first modified boot |
| --- | --- | --- | --- |
| `original-4k-default` | 4K | Current known-good kernel and original command line. | baseline |
| `4k-nohz-rcu-irq` | 4K | `nohz_full`, `rcu_nocbs`, and `irqaffinity` for workload CPUs 2-7. | yes |
| `4k-managed-irq-isolation` | 4K | `isolcpus=managed_irq,2-7` without domain isolation. | no |
| `4k-domain-isolation` | 4K | Domain plus managed IRQ isolation; marked `advanced_less_flexible`. | no |
| `64k-default` | 64K | Official AWS ARM64 64K kernel metadata, no isolation additions. | no |
| `64k-nohz-rcu-irq` | 64K | 64K kernel plus bounded nohz/RCU/IRQ isolation. | no |
| `64k-managed-irq-isolation` | 64K | 64K managed IRQ isolation after earlier 64K validation. | no |

The planner rejects malformed CPU ranges, empty housekeeping CPU sets, and
overlapping housekeeping/workload CPU sets. Existing kernel command-line
parameters are not duplicated or overwritten silently.

## CLI

Read-only Phase 21C0 commands:

```text
autotuneai system boot-doctor --json
autotuneai system kernel-inventory --json
autotuneai system boot-plan --profile <profile> --inventory <inventory.json>
autotuneai system boot-plan verify --plan <plan.json>
autotuneai system recovery-plan --inventory <inventory.json>
```

No real reboot or bootloader apply command exists in Phase 21C0.

## Future Phase 21C Benchmark Protocols

Future real boot profiles must keep these protocols separate:

| protocol | primary metric | secondary evidence |
| --- | --- | --- |
| `steady_state_decode` | decode tokens/s | prefill, variability, context switches, migrations, interrupts, page faults, counters when available |
| `cold_model_load` | model-ready latency | major/minor faults, bytes read, RSS |
| `warm_first_request` | TTFT median and p95 | E2E median/p95, faults, RSS |
| `stability` | variability and tail latency | failed trials and cleanup |

The protocols must not be aggregated into a single speedup number.

## Future Phase 21C Execution Order

```text
G0 original 4K baseline
-> one-time boot G1
-> measure and return to G0
-> one-time boot G2
-> measure and return to G0
-> qualify/install G4 only after explicit authorization
-> one-time boot G4
-> measure and return to G0
-> one-time boot G5
-> measure and return to G0
-> test G3 or G6 only when justified
-> final boot into original G0
-> verify permanent default and original system state
```

Experimental profiles must not chain directly into other experimental profiles.

## Required Future Remote Read-Only Information

Before any real Phase 21C boot attempt, the private remote audit must capture:

```text
current kernel release, package, flavor, base page size
current boot ID
current command-line digest
current default boot entry
known-good fallback entry
bootloader type and one-time mechanism
vmlinuz/initrd/modules/config evidence for current and fallback kernels
official AWS ARM64 64K package metadata
candidate 64K config evidence
/boot and / free-space inventory
Secure Boot and lockdown state
initramfs generation capability
snapshot marker and recovery marker
Serial Console or equivalent recovery verification
SSH post-boot health command
rollback command placeholder verification
two normalized read-only snapshots
```

Private identifiers such as IPs, SSH key paths, snapshot IDs, instance IDs,
host-key fingerprints, model paths, and private config paths must stay out of
tracked files.

## Manual Recovery Prerequisites

The operator must verify privately:

```text
recent EBS snapshot or AMI exists
snapshot marker recorded
original root volume identity recorded
volume detach-and-repair path documented
EC2 Serial Console or equivalent recovery path works
fallback kernel is separate from the running kernel
fallback vmlinuz/initrd/modules/config files are present
one-time boot selection can be verified and cleared
rollback commands are documented
explicit user authorization is present for each real reboot sequence
```

## Local Fixture Evidence

Phase 21C0 local tests exercise:

```text
4K running kernel inventory
64K running kernel inventory
kernel package ownership
missing kernel config remains unknown
missing initrd and modules blockers
insufficient /boot space blocker
unknown bootloader blocker
valid separate fallback
fallback package not installed
NO_HZ_FULL / RCU / page-size config parsing
G0 through G6 boot plans
duplicate command-line parameter blocking
malformed CPU-range refusal
one-time plan verification
recovery plan generation
post-boot health failure classification
final original-state fixture verification
```
