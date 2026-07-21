# Linux System Autotuning

PerfRunbench Phase 21A adds a guarded Linux host-system profile framework.
Phase 21B validates a bounded subset of reversible real-host mechanisms on a
C7g / Neoverse-V1 Arm64 target. Phase 21B does not produce a verified
performance improvement; it validates rollback, artifact, and noise-band guard
behavior.

Phase 20 only exhausted the tested compiler and simple runtime candidates for
one `llama.cpp` workload. It did not evaluate CPU frequency governors, cpusets,
IRQ/workqueue isolation, THP, HugeTLB, allocators, storage startup policy, or
boot-kernel isolation.

Phase 21B executed:

```text
baseline-current
thp-never
allocator-glibc-bounded-arenas
scheduler-batch
affinity-cpus-1-7
vm-system-conservative
thp-always
```

The final result was:

```text
performance_improvement_verified: false
host_restored_to_original_state: true
```

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

Real-host reversible apply requires an explicit opt-in and is intended for
controlled validation only:

```bash
autotuneai system apply \
  --profile cpufreq-performance \
  --lease-minutes 30 \
  --snapshot host-state.json \
  --lease-output lease.json \
  --allow-real-system \
  --json

autotuneai system verify \
  --profile cpufreq-performance \
  --snapshot host-state.json \
  --lease lease.json \
  --allow-real-system \
  --json

autotuneai system rollback \
  --snapshot host-state.json \
  --lease lease.json \
  --allow-real-system \
  --json
```

## Claim Boundary

Phase 21A is local implementation and fixture validation only. Phase 21B is
real C7g reversible-system guard validation. It does not claim a performance
improvement because no positive candidate cleared the configured 5% noise band.
