# Phase 21C1 C7g Same-Kernel Boot Rehearsal

Phase 21C1 performed a single authorized same-kernel GRUB one-time boot
rehearsal on a user-started C7g / Neoverse-V1 Linux Arm64 target. The goal was
to validate recovery readiness, one-time boot arming, post-boot health checks,
and exact bootloader restoration before any future 64K kernel or kernel
parameter experiment.

## Scope

```text
phase: 21C1_same_kernel_one_time_boot_rehearsal
aws_lifecycle_controlled_by_tooling: false
kernel_installed: false
kernel_removed: false
64k_kernel_installed: false
GRUB_permanent_default_changed: false
EFI_BootOrder_changed: false
experimental_kernel_parameters_added: false
guest_reboots_authorized: 1
guest_reboots_performed: 1
performance_benchmarks_run: false
```

Connection details, public/private IP addresses, instance identifiers, SSH key
paths, host-key fingerprints, recovery credentials, model paths, and private
artifact locations are intentionally excluded from this public document.

## Recovery Preconditions

Private evidence recorded these recovery markers before any bootloader write:

```text
recovery_snapshot_or_ami_verified: true
serial_or_out_of_band_recovery_verified: true
serial_console_interactive_login_verified: true
serial_console_sudo_verified: true
serial_getty_persists_across_boot: true
```

The temporary recovery account was verified through the EC2 Serial Console with
interactive login and sudo validation. The account remains a recovery control
for follow-on Phase 21C work and is not part of any benchmark workflow.

## Source And Target Gate

```text
controller_commit: 4beb3803335888788082829e546b693c2d903d85
remote_commit: 4beb3803335888788082829e546b693c2d903d85
controller_remote_commit_match: true
remote_tracked_worktree_clean: true
architecture: aarch64
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
system_state_before_reboot: running
serial_getty_enabled: true
serial_getty_active: true
existing_next_entry_before_arm: false
```

The selected one-time GRUB entry was the non-recovery entry for the already
running known-good kernel:

```text
selected_one_time_entry: Advanced options for Ubuntu > Ubuntu, with Linux 6.17.0-1019-aws
selected_entry_uses_current_kernel: true
selected_entry_non_recovery: true
```

## Boot Sequence

The rehearsal followed a bounded sequence:

```text
immutable pre-reboot inventory
-> GRUB environment backup
-> grub-reboot current known-good entry
-> independent next_entry verification
-> exactly one guest OS reboot
-> bounded SSH reconnect
-> post-boot kernel, page-size, command-line, and health validation
-> GRUB environment restoration check
```

The SSH reconnect window observed the guest become unavailable and then return:

```text
ssh_down_seen: true
ssh_up_seen: true
```

## Post-Boot Validation

```text
boot_id_changed: true
post_kernel_release: 6.17.0-1019-aws
post_base_page_size: 4096
kernel_unchanged: true
page_size_unchanged: true
kernel_command_line_unchanged: true
system_state_after_reboot: running
remote_commit_still_matches: true
remote_worktree_still_clean: true
serial_getty_still_enabled: true
serial_getty_still_active: true
```

GRUB consumed the one-time entry value but left an empty `next_entry=` key in
`grubenv`. The run treated that as a pending-state cleanup requirement rather
than as final restoration. The empty key was removed with the documented GRUB
environment syntax:

```text
cleanup_command: grub-editenv - unset next_entry
next_entry_value_consumed_before_cleanup: true
next_entry_absent_after_cleanup: true
grubenv_restored_to_pre_list_after_cleanup: true
permanent_default_unchanged: true
```

## Artifact Integrity

```text
remote_artifact_manifest_validation: passed
downloaded_artifact_manifest_validation: passed
artifact_files_checked: 30
artifact_mismatches: 0
```

The public repository contains only this sanitized summary. Full command
outputs, GRUB environment hashes, reconnect logs, serial-console screenshots,
and connection-specific evidence are stored only in ignored private artifacts.

## Result

```text
phase21c1_one_time_boot_rehearsal: passed
recovery_snapshot_verified: true
serial_or_out_of_band_recovery_verified: true
one_time_boot_verified: true
permanent_default_unchanged: true
host_healthy_after_reboot: true
ready_for_phase21c_real_boot_validation: true
performance_improvement_verified: false
```

## Boundary

Phase 21C1 proves that the target can safely execute and recover from one
same-kernel GRUB one-time boot transaction with serial recovery validated and
the permanent boot default preserved.

It does not prove 64K kernel availability, 64K performance, boot-parameter
performance, or any workload speedup. Those require a later Phase 21C run that
installs or selects the experimental kernel/profile one at a time, validates
post-boot health, and compares workload performance under a separate benchmark
protocol.
