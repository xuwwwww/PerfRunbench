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
