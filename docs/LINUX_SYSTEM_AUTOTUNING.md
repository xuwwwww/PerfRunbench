# Linux System Autotuning

PerfRunbench Phase 21A adds a guarded Linux host-system profile framework for
future Arm validation. It does not apply real host changes in this phase.
Implementation and tests use fake `/proc`, `/sys`, cgroup, boot, block-device,
IRQ, and service fixtures.

Phase 20 only exhausted the tested compiler and simple runtime candidates for
one `llama.cpp` workload. It did not evaluate CPU frequency governors, cpusets,
IRQ/workqueue isolation, THP, HugeTLB, allocators, storage startup policy, or
boot-kernel isolation.

## Default Flow

```text
snapshot host state
-> discover capabilities
-> generate allowlisted profiles
-> create a bounded plan
-> apply one reversible profile under a lease
-> verify effective state
-> benchmark one protocol
-> rollback in reverse order
-> verify exact restoration
-> bind evidence to Passport and cache metadata
```

## Host Snapshot

`HostStateSnapshot` records host identity, CPU frequency, CPU idle,
scheduler/cgroup, interrupts/workqueues, memory, storage, network, and
security/recovery evidence. Unavailable fields remain:

```text
not_available
```

They are not converted to `false`, `0`, or guessed values.

## Allowlisted Profiles

Profiles are capability driven. Missing interfaces produce `unsupported`,
`partially_supported`, `requires_root`, or `requires_user_snapshot` states
instead of fabricated candidates.

Implemented safe catalog IDs:

```text
baseline-current
cpufreq-performance
affinity-only
cpuset-partition-1h7w
cpuset-partition-2h6w
irq-workqueue-isolation
irqbalance-controlled
scheduler-batch
process-priority
thp-always
thp-madvise
thp-never
thp-defrag-always
thp-defrag-defer
thp-defrag-defer-madvise
thp-defrag-madvise
thp-defrag-never
compaction-conservative
mmap-default
mmap-mlock
no-mmap
no-mmap-mlock
prefault-warm
allocator-glibc
allocator-glibc-bounded-arenas
allocator-jemalloc
allocator-mimalloc
storage-read-ahead
io-scheduler
numa-policy
vm-system-conservative
hugetlb-bounded
```

Disallowed examples remain permanently unsafe:

```text
disable-mitigations
disable-security-modules
panic-on-oom
unsafe-overcommit
disable-watchdogs
ssh-firewall-change
sched-fifo
no-housekeeping
offline-housekeeping
remove-kernel
aws-metadata-change
hide-restore-failure
```

## CLI

Read-only by default:

```bash
autotuneai system doctor --json
autotuneai system snapshot --output host-state.json
autotuneai system plan --catalog safe-system --json
autotuneai system show-profile cpufreq-performance --json
autotuneai system boot-plan --profile aws-4k-default --json
autotuneai system compare --run-id <run-id> --json
```

Phase 21A reversible apply uses fake roots only unless a future phase adds an
explicit real-host opt-in:

```bash
autotuneai system apply \
  --root <fake-root> \
  --profile cpufreq-performance \
  --lease-minutes 30 \
  --snapshot host-state.json \
  --lease-output lease.json \
  --json

autotuneai system verify \
  --root <fake-root> \
  --profile cpufreq-performance \
  --snapshot host-state.json \
  --lease lease.json \
  --json

autotuneai system rollback \
  --root <fake-root> \
  --snapshot host-state.json \
  --lease lease.json \
  --json
```

## Claim Boundary

Phase 21A is local implementation and fixture validation only. It does not
claim a C7g or T4g performance improvement, does not access AWS, and does not
modify a real host.
