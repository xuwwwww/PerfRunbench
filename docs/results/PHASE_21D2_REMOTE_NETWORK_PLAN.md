# Phase 21D2 Remote Network Plan

Phase 21D2 is a future real-target phase. The planned topology is:

```text
C7g: llama-server target
T4g: remote load generator
```

Phase 21D0 does not access either instance.

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
