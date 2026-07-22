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

The C7g Phase 21C0 remote audit preserved this boundary. It inspected the
running 4K kernel, fallback evidence, bootloader type, and official 64K package
metadata, then left:

```text
kernel_or_boot_state_modified: false
ready_for_phase21c_real_boot_validation: false
```

The result remained blocked because the private evidence did not verify the
recovery snapshot marker, out-of-band recovery path, one-time boot clearing,
post-boot SSH health check, rollback command, or user authorization for a real
reboot sequence.

## Temporary Boot Entries

Phase 21C2A executes experimental 4K boot profiles through isolated temporary
GRUB custom entries. The generated plan must be inspected before a real reboot
and must satisfy:

```text
original generated Ubuntu menu entry: untouched
permanent default: unchanged
direct grub.cfg edit: refused
temporary custom entry: unique and removable
experimental entry reuse: refused
maximum reboot count per experiment: 1
transaction start: original-4k-default
transaction end: original-4k-default
```

Cleanup requires removal or restoration of the temporary custom-entry file,
`update-grub`, clearing any consumed one-time `next_entry`, and verifying the
temporary entry ID is absent. If the host does not return exactly to the
original known-good 4K state, the next experimental profile must not begin.

The real Phase 21C2A run validated this recovery model. A pre-measurement G1
temporary entry failed because the local harness generated the wrong kernel
path. Serial Console was used to select the original Ubuntu entry, cleanup
removed the temporary entry and `next_entry`, and G0 validation passed before
corrected G1/G2/G3 measurements continued. Final validation confirmed the
original 4K command line, kernel, page size, and permanent default were
restored.

## Kernel Package Rollback

Phase 21C2B0 adds plan-only rollback coverage for future official AWS ARM64 64K
package transactions. The rollback plan removes only the exact package set
introduced by the transaction. It explicitly does not run `apt autoremove`, does
not remove the running 4K kernel, does not remove the fallback kernel, and does
not change repository configuration.

Final restoration for a future real 64K session requires:

```text
running original 4K kernel
PAGE_SIZE 4096
original command line
original resolved permanent default
original /etc/default/grub bytes
original grubenv state
no next_entry, including no empty next_entry key
no temporary 64K experimental entries
no residual 64K kernel package
no residual 64K modules directory
fallback kernel intact
no package-manager lock
no broken dpkg state
no benchmark or server process
```

The temporary G0 safety-pin plan is separate from the reversible system-profile
lease model. It may be used only as a future recovery safeguard after explicit
authorization, must pin a stable verified G0 entry, and must be restored
byte-for-byte after validation.
