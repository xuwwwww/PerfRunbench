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
