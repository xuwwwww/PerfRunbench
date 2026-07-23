# Phase 21D3 Serving Autotuning Productization

Phase 21D3 packages the Phase 21D2B remote-serving validation into public
recommendation, dry-run, explain, demo, release-candidate, and claim-consistency
artifacts. It does not access AWS, does not rerun measurements, and does not
change any host state.

## Source Ledger

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21d2b_commit: d89bada8088cdc933d53081e4e730d00325d16b0
phase21d3_initial_commit: d89bada8088cdc933d53081e4e730d00325d16b0
pr6_number: 6
pr6_status: open draft
pr6_head_branch: feature/linux-system-kernel-autotuning
pr6_head: 09e5809d05dda933c7c27b88b1222e067674090e
```

The release-candidate branch starts from the exact Phase 21D2B evidence commit:

```text
release_branch: release/v0.3.0-rc1-serving-autotuning
branch_base: d89bada8088cdc933d53081e4e730d00325d16b0
```

## Productized Recommendation

The public package creates workload-scoped serving recommendations only for
evidence-qualified remote-client workloads:

```text
profile: slots4-cont-on
recommendation_classes:
  - maximum_throughput_remote
  - interactive_latency_remote
eligible_workloads:
  - interactive-short, concurrency 4
  - decode-heavy, concurrency 4
  - prefill-heavy, concurrency 4
```

No balanced, shared-prefix, memory-constrained, or remote-network tuning
recommendation is created.

## CLI Surface

```text
autotuneai serve recommend --workload <id> --concurrency 4
autotuneai serve explain --recommendation <id>
autotuneai serve run --best --workload <id> --concurrency 4 --dry-run
```

`serve run --best` loads an existing recommendation and never launches a
search. The public dry-run path reports the selected profile, expected command
shape, hard-drift checks, memory/context preflight, port preflight, canary
requirement, and cleanup plan without mutating the host.

## Hard-Drift Scope

The recommendation is guarded by:

```text
model digest
server binary digest
server capability digest
architecture
CPU model and logical CPU count
memory class
network topology and direct path
workload ID
traffic mode
concurrency
CPU assignment
parallel slots
continuous batching
context size
artifact manifest identity
cleanup status
```

Hard-drift states include:

```text
recommendation_applicable
diagnostic_only
hard_drift
memory_guard_failed
capability_mismatch
topology_mismatch
workload_mismatch
```

## Evidence Boundary

The package uses preserved Phase 21D2B evidence:

```text
measured_requests_total: 608
paired_block_pairs: 6
retries: 0
request_failures: 0
quality_failures: 0
network_tuning_status: network_tuning_not_executed_evidence_absent
artifact_validation: passed
dual_host_restored: true
```

Phase 21D3 does not create new measured performance data.

