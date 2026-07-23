# Serving Benchmark Protocols

Serving protocols keep throughput, latency, capacity, and network evidence
separate. Results from different workload identities, traffic modes,
concurrency classes, client topology, or network topology must not be combined.

## Traffic Modes

| mode | Semantics |
| --- | --- |
| `closed-loop` | Maintain bounded in-flight clients; each client submits the next request after the previous request completes. |
| `open-loop` | Submit requests against a deterministic arrival schedule and record requested versus actual submission time. |
| `burst` | Submit a synchronized bounded burst to expose queue and slot behavior. |
| `shared-prefix` | Preserve an identical prefix and vary a controlled suffix for cache/prefix qualification. |

No measured request is silently retried.

## Metrics

Throughput:

- completed requests/s
- aggregate output tokens/s
- aggregate total tokens/s
- input/prefill tokens/s
- per-slot throughput when exposed by the server

Latency:

- queue delay when derivable
- TTFT p50/p95/p99
- E2E p50/p95/p99
- inter-token latency when available

Correctness:

- HTTP failures
- timeouts
- malformed responses
- incomplete generation
- output-token shortfall
- quality/output digest result

Fairness:

- per-client completed requests
- per-client throughput
- latency spread
- starvation detection

Resource:

- peak and steady RSS
- CPU and per-CPU utilization
- context switches and CPU migrations
- page faults
- interrupt and softirq deltas
- network bytes
- connection count
- server process and child count
- port cleanup
- OOM or kernel-warning evidence

Unavailable metrics remain unavailable. They are not converted to zero.

## Eligibility Guards

Throughput candidates require at least a 5% aggregate-throughput improvement,
fresh confirmation, zero failed requests, quality pass, no unacceptable p95
TTFT regression, and safe memory headroom.

Latency candidates require at least a 10% p95 TTFT or E2E improvement,
sufficient request samples, fresh confirmation, aggregate-throughput regression
no worse than 5%, zero failures, and quality pass.

Capacity candidates may be eligible when they support a higher safe concurrency
or context size inside the declared latency SLO. They are capacity results, not
raw speedup claims.

## Phase 21D1 Loopback Protocol

Phase 21D1 used a target-local loopback protocol:

```text
client_cpu_set: CPU 0
server_cpu_set: CPUs 1-7
traffic_mode: closed-loop
server_candidate_startup_limit: 24
measured_http_request_limit: 600
```

The confirmed `interactive-short` concurrency-4 result used 50 measured
requests per side for each eligible class, zero request failures, zero quality
failures, and client CPU utilization below 1% of one CPU. The recommendations
are loopback recommendations only; Phase 21D2 must produce fresh
remote-network baselines and confirmations before any remote-client claim.

Phase 21D1 did not create eligible capacity, shared-prefix, or
balanced-multi-user recommendations. A screening point estimate without fresh
confirmation remains ineligible.

## Phase 21D2 Remote-Client Protocol

Phase 21D2B used a two-host protocol:

```text
server: C7g / Neoverse-V1 running pinned llama-server
client: T4g / Neoverse-N1 class running the deterministic load generator
topology: c7g_server_t4g_remote_client
primary profile: slots4-cont-on
server_cpu_policy: full_capacity_server_cpu_set_no_local_load_generator_reservation
```

The first remote-client measurement confirmed `slots4-cont-on` and measured
full-capacity remote serving. It did not reuse Phase 21D1 loopback
recommendations as remote-network evidence.

```text
measured_requests_total: 608
paired_block_pairs: 6
request_failures: 0
quality_failures: 0
direct_private_network_path: true
```

Client-observed TTFT and E2E latency are primary for the remote-client
topology. Server-side timestamps and internal timings are useful supporting
evidence, but PerfRunbench must not subtract unsynchronized client and server
wall clocks.

IRQ affinity, RPS, XPS, listen backlog, or other network profiles remain
skipped unless the remote run first records network-bottleneck evidence:

- client-observed latency regression for the same workload shape
- server CPU not saturated while the client observes delay
- client CPU not saturated
- retransmits, drops, listen overflow, or backlog saturation
- RX/TX queue or softirq hotspot
- before/after counters from both hosts

Without that evidence, the valid Phase 21D2B outcome is a remote-client
recommendation or an inconclusive result from the serving runtime profile only.
Phase 21D2B followed this rule and skipped IRQ/RPS/XPS/backlog tuning because
network-bottleneck evidence was absent.
