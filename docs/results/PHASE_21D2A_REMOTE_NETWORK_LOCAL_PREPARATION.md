# Phase 21D2A Remote Network Local Preparation

Phase 21D2A completed local preparation for a future two-host serving
validation. It did not access AWS, SSH to a target, start an instance, stop an
instance, or run a real benchmark.

```text
aws_accessed: false
remote_accessed: false
real_server_measured: false
network_state_modified: false
phase21d2a_remote_network_local_preparation: passed
```

## Planned Topology

Phase 21D2B will use fresh real evidence from this topology:

```text
C7g / Neoverse-V1: llama-server target
T4g / Neoverse-N1 class: remote load generator
```

The Phase 21D1 loopback result is intentionally excluded from remote-network
claims. A loopback RecommendationBundle must be treated as hard drift when the
current topology is `c7g_server_t4g_remote_client`.

## Local Contract

`autotuneai serve network-plan` now emits a versioned plan with:

- server and remote-client host roles
- expected architecture, CPU class, and logical CPU count
- source-state and clean-worktree requirements for both hosts
- private-connection evidence policy
- client/server timing policy
- artifact validation policy for both hosts and downloaded artifacts
- Phase 21D2B execution order
- explicit `slots4-cont-on` confirmation requirement
- network bottleneck gate for IRQ, RPS, XPS, and listen-backlog changes

The command remains plan-only:

```bash
autotuneai serve network-plan --output results/serving/phase21d2_network_plan.json
autotuneai serve network-plan --json
```

## Phase 21D2B Entry Gate

Before running any remote serving measurement, Phase 21D2B must validate:

- controller, server, and client source commits match exactly
- server and client tracked worktrees are clean
- server target is Linux Arm64 / Neoverse-V1 / 8 logical CPUs
- client target is Linux Arm64 / Neoverse-N1 class / 2 logical CPUs
- pinned server binary, model digest, and server capability digest are
  unchanged or explicitly recorded as new fresh evidence
- `slots4-cont-on` is inspectable on the server
- server ports are free before measurement
- client resource saturation checks are recorded
- public IPs, SSH key paths, host-key fingerprints, and model paths remain only
  in ignored private evidence/configuration

## Measurement Boundary

Phase 21D2B must first confirm `slots4-cont-on` and measure full-capacity
remote serving. The remote-client topology removes the Phase 21D1 local
load-generator CPU reservation, so the server CPU policy is:

```text
full_capacity_server_cpu_set_no_local_load_generator_reservation
```

Client-observed TTFT and E2E latency are the primary remote-topology timing
measurements. Server internal timing may be recorded, but unsynchronized server
and client wall clocks must not be subtracted.

## Network Tuning Gate

IRQ, RPS, XPS, and listen-backlog profiles are skipped by default:

```text
skip_irq_rps_xps_until_network_bottleneck_is_observed
```

They may be tested only when the real remote run records sufficient evidence,
such as:

- remote-client TTFT or E2E regression for the same workload shape
- server CPU not saturated while the client observes delay
- client CPU not saturated
- socket retransmits, drops, or listen overflow
- RX/TX queue or softirq hotspot
- before/after network counters from both hosts

Blind IRQ reassignment, blind RPS/XPS changes, network sysctl sweeps, and
persistent network configuration changes remain out of scope.

## Prepared Phase 21D2B Order

```text
validate_dual_host_source_and_target_fingerprints
take_server_and_client_read_only_snapshots
confirm_slots4_cont_on_profile_on_remote_client_topology
measure_full_capacity_remote_serving
classify_server_client_network_bottleneck
run_irq_rps_xps_profiles_only_if_network_bottleneck_gate_passes
generate_remote_network_recommendation_or_inconclusive_result
validate_server_client_artifacts_and_cleanup
```

## Local Validation

The local fake E2E validates:

- `serve network-plan` writes a versioned JSON plan
- JSON and text output preserve the local-only boundary
- no real IP address, `.pem` path, or private key material enters the plan
- loopback-to-remote topology reuse is classified as hard drift
- network tuning is gated until bottleneck evidence exists

```text
ready_for_phase21d2b_real_remote_network_validation: true
ready_for_phase21d3_packaging: false
```
