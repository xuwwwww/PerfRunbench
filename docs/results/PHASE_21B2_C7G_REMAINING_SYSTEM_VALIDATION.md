# Phase 21B2 C7g Remaining Reversible System Validation

Phase 21B2 validates the remaining safe reversible Linux host-system
mechanisms that were intentionally left unexercised in Phase 21B. The run used
the same C7g / Neoverse-V1 Arm64 target class, fixed Qwen2.5-7B Q4_K_M model,
pinned `llama.cpp` source, and `upstream-default` `llama-bench` binary family
as Phase 20 and Phase 21B.

This is safety, protocol-bound measurement, rollback, and guard evidence. It
does not produce a performance recommendation.

The result is:

```text
phase21b2_remaining_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
all_reversible_families_exhausted: true
ready_for_phase21c_boot_kernel_validation: false
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
branch: validation/phase21b2-c7g-remaining-system
initial_expected_commit: a11c9a7abfa0c56db053179ac29c78712e817200
final_validation_commit: 88a2cee992a7250b1e957e9c4fb3915c54e74721
```

One focused source fix was required by real validation:

```text
88a2cee fix: normalize bracketed block scheduler mode
```

The defect was limited to effective-state normalization for block queue
scheduler sysfs values such as:

```text
[none] mq-deadline
none [mq-deadline]
```

No model, dataset, `llama.cpp` source update, AWS lifecycle action, EBS
change, reboot, GRUB change, bootloader write, kernel package change, or
persistent system configuration change was performed by PerfRunbench during
this validation.

## Preflight And Recovery Boundary

Preflight verified:

```text
source_match: controller and target checkout matched 88a2cee
remote_worktree: clean
ssh: stable with strict host-key checking after private enrollment
sudo_nopasswd: available
architecture: Linux aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
THP: writable
cgroup_cpuset: available through cgroup v2
workqueue_cpumask: writable
perf_stat: root available without security weakening
```

Two normalized read-only original snapshots matched before system-profile
measurement:

```text
original_snapshot_stability: true
normalized_snapshot_sha256: d74a8534d79cbf2d0d7229246ad73f081b1de0368681f39f1cc37e6f444177ba
```

The rollback-watchdog rehearsal passed before candidate execution.

An intermediate private harness attempt stopped on a dirty-host guard after
IRQ/workqueue isolation was applied in the wrong ordering for that family.
The guard behavior was correct: it refused to benchmark from a state that did
not match the original verified host state. The host was restored from the
original snapshot, and a same-marker normalized snapshot comparison passed
before the continuation resumed.

Downloaded artifact validation later caught a private harness manifest-ordering
issue where the final summary was updated after manifest generation. The
finalizer was rerun so the final summary was stable before manifest generation;
remote validation and downloaded validation both passed.

## Protocol

Primary ranking protocol:

```text
operation: llama-bench steady_state_decode
result_origin: measured
binary: upstream-default
threads_baseline: 8
prompt_tokens: 512
generation_tokens: 128
baseline_repetitions: 5
candidate_repetitions: 2 or 3 depending on family
noise_band_percent: 5.0
quality_gate_status: not_evaluated
```

Other protocol families were preserved but not ranked against the steady-state
decode baseline:

```text
cold_model_load
warm_page_cache_decode
interactive_server secondary smoke
perf-counter availability and one root perf baseline
boot-validation readiness audit
```

## Executed Candidates

Steady-state decode candidates:

| candidate | family | protocol | decode median tok/s | min | max | CV % | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `baseline-current` | baseline | `steady_state_decode` | `16.260` | `16.230` | `16.350` | `0.255` | baseline |
| `no-mmap` | mapping | `steady_state_decode` | `16.750` | `16.710` | `16.790` | `0.195` | inconclusive, `+3.014%` |
| `mmap-explicit` | mapping | `steady_state_decode` | `16.320` | `16.320` | `16.380` | `0.173` | inconclusive, `+0.369%` |
| `allocator-glibc-bounded-arenas` | allocator | `steady_state_decode` | `16.230` | `16.130` | `16.230` | `0.291` | inconclusive, `-0.185%` |
| `thp-always-mmap` | THP exercise | `steady_state_decode` | `16.525` | `16.520` | `16.530` | `0.030` | inconclusive, `+1.630%` |
| `thp-madvise-mmap` | THP exercise | `steady_state_decode` | `16.200` | `16.200` | `16.200` | `0.000` | inconclusive, `-0.369%` |
| `thp-never-mmap` | THP exercise | `steady_state_decode` | `16.305` | `16.290` | `16.320` | `0.092` | inconclusive, `+0.277%` |
| `cpuset-1h7w` | cpuset partition | `steady_state_decode` | `1.070` | `1.070` | `1.070` | `0.000` | regressed, `-93.419%` |
| `cpuset-2h6w` | cpuset partition | `steady_state_decode` | `1.190` | `1.190` | `1.200` | `0.395` | regressed, `-92.681%` |
| `cpuset-1h7w-irq-workqueue` | IRQ/workqueue isolation | `steady_state_decode` | `1.080` | `1.070` | `1.080` | `0.438` | regressed, `-93.358%` |

No steady-state decode candidate exceeded the configured 5% noise band in the
positive direction. No combined profile was run because no independent
mechanism was eligible.

Protocol-specific evidence not ranked against the steady-state baseline:

| candidate | family | protocol | recorded median tok/s | classification |
| --- | --- | --- | ---: | --- |
| `io-scheduler-mq-deadline` | storage cold start | `cold_model_load` | `17.520` | protocol baseline missing |
| `storage-readahead-higher` | storage cold start | `cold_model_load` | `17.590` | protocol baseline missing |
| `storage-readahead-lower` | storage cold start | `cold_model_load` | `17.500` | protocol baseline missing |
| `prefault-warm` | mapping | `warm_page_cache_decode` | `16.180` | protocol baseline missing |

The storage and prefault rows are preserved for future protocol-specific work.
They are not throughput recommendations.

## THP, HugeTLB, Allocator, And Perf Evidence

THP was genuinely exercised only for the `thp-always-mmap` candidate:

```text
thp-always-mmap max_anon_huge_pages_kb: 4483072
thp-madvise-mmap max_anon_huge_pages_kb: 0
thp-never-mmap max_anon_huge_pages_kb: 0
```

This did not clear the 5% guard.

HugeTLB remained not qualified because there was no verified `llama.cpp`
HugeTLB consumer or `hugetlbfs` mapping path in this run. No HugeTLB pages were
reserved.

`jemalloc` and `mimalloc` were not used:

```text
system_jemalloc: unavailable
system_mimalloc: unavailable
non_persistent_extraction_result: unavailable
```

Perf counters were inspected without weakening host security:

```text
nonroot_perf: denied by target security policy
root_perf: available without sysctl change
perf_baseline: passed
```

## Secondary Server Smoke

The `llama-server` smoke is secondary runtime evidence only. It does not
participate in BootTrace quality gating and does not create a recommendation.

| candidate | requests | nonempty output | TTFT median ms | TTFT p95 ms | E2E median ms | E2E p95 ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `baseline-current` | 5 | yes | `57.335` | `131.960` | `341.199` | `417.163` |
| `cpuset-1h7w` | 5 | yes | `927.291` | `1310.087` | `5561.149` | `6398.993` |

The reduced-CPU cpuset server smoke regressed sharply and is not eligible.

## Rollback And Cleanup

Every writeable family began from the verified original host state. Each
profile was applied, measured or inspected, rolled back, and checked before
the next family continued.

Validated rollback/cleanup:

```text
cpuset cgroups removed: passed
block scheduler restored: passed
storage read-ahead restored: passed
THP enabled mode restored: passed
IRQ affinity masks restored: passed
unbound workqueue cpumask restored: passed
irqbalance service state restored when temporarily stopped: passed
loopback test port range released: passed
residual llama-bench/llama-server/runner processes: none
final normalized snapshot == original normalized snapshot: true
```

Final state:

```text
host_restored_to_original_state: true
active runtime lease or lock files: none
```

## Boot-Validation Readiness

The boot-validation readiness audit was read-only. It did not start Phase 21C.

Outcome:

```text
status: read_only_completed
ready_for_phase21c_boot_kernel_validation: false
blocker: no explicit user-authorized reboot and out-of-band recovery path for Phase 21C in this run
```

No reboot-required kernel profile was applied.

## Artifacts

Artifact validation passed on the target and again after download:

```text
remote_artifacts_checked: 327
remote_artifacts_failed: 0
local_artifacts_checked: 327
local_artifacts_failed: 0
downloaded_archive_sha256: b00de09c2c4b1bc224c0e69af182dc7f736b48ed8b31f1db108976e56a01dc06
```

The archive remains private because it contains raw host evidence. Public docs
contain only sanitized summaries.

## Result Boundary

Legal wording:

> PerfRunbench validated the remaining reversible Linux system-profile
> mechanisms on a real C7g / Neoverse-V1 Arm64 target, restored the final host
> state exactly, and correctly produced no performance recommendation because
> no positive steady-state decode candidate exceeded the 5% noise band.

Do not claim:

```text
system tuning accelerated llama.cpp on C7g
no-mmap is a recommendation
THP always is globally better
storage scheduler or read-ahead improved steady-state decode
cpuset or IRQ isolation improved this workload
HugeTLB was validated
jemalloc or mimalloc was validated
Phase 21C boot/kernel tuning was performed
```

Phase 21B2 is a successful guard and rollback validation. The lack of a
performance recommendation is the expected safe outcome when measured
differences remain inside the configured noise band or belong to a different
protocol.
