# Phase 21D2 Remote Network Plan

Phase 21D2A completed local preparation for the future real-target phase. The
planned Phase 21D2B topology is:

```text
C7g: llama-server target
T4g: remote load generator
```

Phase 21D2A does not access either instance. Private connection details remain
outside tracked files.

## Entry Gate

Before any Phase 21D2B real measurement:

- controller, C7g server, and T4g client source commits must match exactly
- both remote tracked worktrees must be clean
- C7g must validate as Linux Arm64 / Neoverse-V1 / 8 logical CPUs
- T4g must validate as Linux Arm64 / Neoverse-N1 class / 2 logical CPUs
- the pinned `llama-server` binary, model digest, and capability digest must be
  inspectable
- `slots4-cont-on` must be confirmed on the remote-client topology
- loopback recommendations from Phase 21D1 must be treated as hard drift
- public IPs, SSH key paths, host-key fingerprints, and model paths must stay
  in ignored private evidence/configuration only

## Measurement Order

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

Remote-client TTFT and E2E latency are client-observed metrics. Server internal
timings may be preserved, but unsynchronized server and client wall clocks must
not be subtracted.

## Network Discovery

Future discovery must record:

- network interface and driver
- RX/TX queue counts
- IRQ mapping
- writable IRQ affinity
- RPS state
- XPS state
- irqbalance state
- connection tracking and socket counters
- packet drops
- retransmissions
- listen backlog
- softirq distribution

## Allowed Profiles

Profiles may include only when capability and bottleneck evidence supports
them:

- original network state
- IRQs concentrated on housekeeping CPUs
- RPS aligned with selected server CPUs
- XPS aligned with queue/CPU topology
- one evidence-based listen-backlog profile
- connection reuse versus connection-per-request

Network sysctls must not be blindly swept. A sysctl candidate requires evidence
of listen overflow, queue drops, socket-buffer pressure, retransmission-related
limitation, or backlog saturation.

Every network mutation must be lease-bound and exactly reversible.

By default Phase 21D2B skips IRQ, RPS, XPS, and listen-backlog changes until
the full-capacity remote-serving run shows a network bottleneck rather than a
server CPU, client CPU, model/runtime, or request-shape limit.
