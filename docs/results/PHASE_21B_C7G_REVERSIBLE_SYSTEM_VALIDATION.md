# Phase 21B C7g Reversible System Validation

Phase 21B validates reversible Linux host-system profile handling on a real
C7g / Neoverse-V1 Arm64 target. This is guard and portability evidence for
system-scoped tuning. It did not produce a verified performance improvement.

The result is:

```text
phase21b_c7g_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
```

## Scope

Target summary:

```text
target_class: user-provisioned AWS C7g.2xlarge / Graviton3
operating_system: Ubuntu 24.04.4 LTS
kernel_release: 6.17.0-1019-aws
architecture: Linux aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
memory_class: approximately 15 GiB available RAM
root_filesystem: ext4
base_page_size: 4096
cgroup: cgroup v2
numa_nodes: 1
```

Source-under-test:

```text
branch: validation/phase21b-c7g-reversible-system
phase21a_base_commit: 27c73e69ada9d3d5ba501de2151111f30c5d0bb3
final_validation_commit: f723384c6ead0111bc5ae534a29504b34da15422
```

Two focused correctness fixes were required by real validation:

```text
891a3c4 fix: tolerate volatile system queue masks
f723384 fix: normalize bracketed sysfs mode verification
```

Workload identity:

```text
llama.cpp source commit: 47c786924ad1ab7e91da2cdc72fcdb563780c2bd
llama.cpp tag context: b9999
binary profile: upstream-default
llama-bench_sha256: 60371f61d38df8b9247a64b6c4db2c0afb867a2ef11c496ff9b77223d572aff3
model: Qwen2.5-7B Q4_K_M
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
```

No model, dataset, `llama.cpp` source update, AWS lifecycle action, EBS change,
GRUB change, bootloader change, kernel switch, or persistent system
configuration was performed by PerfRunbench during this validation.

## Preflight

Read-only preflight passed:

```text
source_match: controller and target checkout matched f723384
remote_worktree: clean
ssh: stable
sudo_nopasswd: available
memory_headroom: >= 10 GiB
root_free_space: >= 20 GiB
load: low
swap: absent
CPUFreq: not exposed on this virtualized target
THP: available and writable
workqueue_cpumask: available and writable
irqbalance: active, but bounded control was not exercised
perf_stat: not_available
```

Perf hardware counters were left `not_available`; the run did not lower
`perf_event_paranoid` or modify security settings.

## Original Snapshot And Watchdog

The user supplied a private recovery marker. PerfRunbench used only the marker
digest in run evidence.

Two independent original snapshots normalized equal after excluding volatile
runtime fields:

```text
normalized_snapshot_sha256: 64e940d57ae8dcfd7adacee04af9b4b3e796098153c4b8d7e3e0ba54207b7787
normalized_equal: true
```

Ignored volatile fields were counters and runtime accounting only:

```text
captured_at_utc
snapshot_digest
available_memory
interrupt and softirq counters
vmstat / buddyinfo / pagetypeinfo
softnet counters
khugepaged scan/collapse counters
```

The watchdog rehearsal used a harmless private fixture setting. It restored the
fixture value after simulated controller interruption and removed its lease:

```text
watchdog_rehearsal: passed
```

## Executed Candidates

Protocol:

```text
operation: llama-bench steady_state_decode
result_origin: measured
binary: upstream-default
threads_baseline: 8
prompt_tokens: 512
generation_tokens: 128
repetitions_per_candidate: 5
noise_band_percent: 5.0
quality_gate_status: not_evaluated
```

Candidate order was deterministic:

```text
baseline-current
thp-never
allocator-glibc-bounded-arenas
scheduler-batch
affinity-cpus-1-7
vm-system-conservative
thp-always
```

| candidate | family | host write | decode median tok/s | min | max | CV % | decision vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `baseline-current` | baseline | no | `17.44` | `17.17` | `17.49` | `0.672` | baseline |
| `thp-never` | THP | yes | `17.42` | `17.41` | `17.47` | `0.122` | inconclusive, `-0.115%` |
| `allocator-glibc-bounded-arenas` | allocator | no | `17.46` | `17.44` | `17.47` | `0.067` | inconclusive, `+0.115%` |
| `scheduler-batch` | scheduler | no | `17.43` | `16.79` | `17.45` | `1.483` | inconclusive, `-0.057%` |
| `affinity-cpus-1-7` | CPU placement | no | `15.61` | `15.55` | `15.61` | `0.150` | regressed, `-10.493%` |
| `vm-system-conservative` | targeted sysctl | yes | `17.43` | `17.26` | `17.47` | `0.429` | inconclusive, `-0.057%` |
| `thp-always` | THP | yes | `17.50` | `17.40` | `17.53` | `0.264` | inconclusive, `+0.344%` |

No candidate exceeded the configured 5% noise band in the positive direction.
No recommendation was created.

The `affinity-cpus-1-7` candidate regressed because it reduced the workload CPU
set and thread count from the eight-CPU baseline. This is valid negative
evidence, not an optimization.

## Rollback Evidence

Host-level candidates were applied one at a time from the verified original
state and rolled back before the next candidate:

```text
thp-never: rollback passed
vm-system-conservative: rollback passed
thp-always: rollback passed
```

The targeted sysctl profile temporarily changed only:

```text
vm.stat_interval: 1 -> 10 -> 1
kernel.sched_autogroup_enabled: 1 -> 0 -> 1
```

THP profiles temporarily changed only:

```text
transparent_hugepage/enabled: madvise -> never -> madvise
transparent_hugepage/enabled: madvise -> always -> madvise
```

Final effective state:

```text
transparent_hugepage/enabled: always [madvise] never
transparent_hugepage/defrag: always defer defer+madvise [madvise] never
vm.stat_interval: 1
kernel.sched_autogroup_enabled: 1
workqueue_cpumask: ff
```

Final normalized snapshot matched the original:

```text
host_restored_to_original_state: true
```

Historical lease artifacts retain their apply-time `active: true` field for
auditability. The separate lease release report verified that all corresponding
rollback reports passed and that no active runtime lease or lock remained:

```text
lease_release_status: passed
residual_processes: none
```

## Unsupported Or Unexercised Profiles

Unsupported or not exercised in Phase 21B:

```text
cpufreq-performance: unsupported, CPUFreq policies not exposed
irqbalance-controlled: not exercised
cpuset-partition-1h7w / cpuset-partition-2h6w: planning only in this run
irq-workqueue-isolation: planning only in this run
mmap / mlock / no-mmap: not exercised in this run
jemalloc / mimalloc: unsupported, allocator not preinstalled
storage-read-ahead / io-scheduler: not exercised; cold-load ranking out of scope
numa-policy: unsupported, single NUMA node
hugetlb-bounded: supported_but_not_exercised; no verified HugeTLB mapping path
boot-kernel profiles: out of scope; no reboot or bootloader changes
```

## Artifacts

Remote artifact validation passed and the downloaded archive was independently
validated on the controller:

```text
remote_artifacts_checked: 712
remote_artifacts_failed: 0
local_artifacts_checked: 712
local_artifacts_failed: 0
artifact_archive_sha256: 80d6e1e06a52b2666a3346e237fde171fbcd02c585501f799f85f4ac85897acc
```

The archive remains private because it contains raw host evidence. Public docs
contain only sanitized summaries.

## Result Boundary

Legal wording:

> PerfRunbench validated reversible Linux host-system tuning safeguards on a
> real C7g / Neoverse-V1 Arm64 target and correctly produced no recommendation
> when all positive differences remained inside the 5% noise band.

Do not claim:

```text
system tuning accelerated llama.cpp on C7g
THP always is better globally
allocator arena limiting is a performance recommendation
affinity search found an Arm optimum
kernel or boot-profile tuning was validated
```

Phase 21B is a safety and guard validation. Lack of improvement is a valid
result because the system refused to promote noise-band changes.
