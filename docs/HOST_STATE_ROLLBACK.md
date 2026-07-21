# Host State Snapshot, Lease, And Rollback

Every non-read-only system profile change is lease-bound.

## Flow

```text
snapshot
-> capability validation
-> plan
-> pre-apply state check
-> apply one setting
-> verify effective state
-> benchmark
-> collect evidence
-> reverse-order rollback
-> verify exact restoration
```

`SystemTuningLease` records the lease ID, profile ID, snapshot digest, expiry
timestamp, per-setting before/requested/effective values, apply status, and
watchdog scope.

The watchdog scope is:

```text
reversible_non_boot_settings_only
```

It is intentionally not allowed to modify boot settings.

If exact restoration fails, rollback sets:

```text
host_state_dirty: true
```

Further tuning must be refused until the host is inspected and restored.

Phase 21A apply and rollback operate on fake roots for validation. Phase 21B
then validated bounded real-host reversible profiles on C7g without persistent
configuration changes.

Phase 21B confirmed:

```text
thp-never: rollback passed
thp-always: rollback passed
vm-system-conservative: rollback passed
final normalized snapshot == original normalized snapshot
active runtime lease or lock files: none
```

Historical lease artifacts preserve their apply-time `active` value for audit
purposes. Runtime lease release is recorded separately by the release report.

Phase 21B2 extended the same rollback discipline to cpuset, block storage,
THP mmap exercise, IRQ affinity, workqueue cpumask, and temporary irqbalance
control. The run intentionally stopped when an intermediate private harness
attempt tried to benchmark IRQ/workqueue isolation from a state that no longer
matched the original snapshot. After exact restoration, continuation completed
and confirmed:

```text
cpuset cgroups removed: passed
block scheduler restored: passed
storage read-ahead restored: passed
THP enabled mode restored: passed
IRQ affinity masks restored: passed
unbound workqueue cpumask restored: passed
irqbalance service state restored when temporarily stopped: passed
loopback test port range released: passed
residual workload processes: none
final normalized snapshot == original normalized snapshot
```

If downloaded artifact validation detects a digest mismatch, the run is not
documented as complete until the manifest ordering is repaired and the artifact
set validates both remotely and after download.

## Reboot-Required Profiles

Phase 21C0 keeps reboot-required profiles outside the reversible lease model.
The watchdog scope remains:

```text
reversible_non_boot_settings_only
```

Boot and kernel experiments require a separate transaction:

```text
read-only inventory
-> recovery verification
-> one-time boot plan
-> explicit reboot authorization boundary
-> post-boot health check
-> benchmark
-> one-time rollback boot
-> original-state verification
```

No future boot profile may become the permanent default until a one-time boot
has passed health checks, benchmark artifact validation, rollback, and final
original-state verification. Experimental profiles must return to the original
known-good boot state before the next experimental profile starts.

Phase 21C0 only validates this logic with fixtures and read-only inventory. It
does not write GRUB, EFI variables, initramfs files, kernel packages, or kernel
command lines.
