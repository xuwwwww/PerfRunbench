# Serving Recommendation Safety

Serving RecommendationBundles are protocol-specific. They are bound to:

- serving workload ID
- traffic mode
- concurrency class
- request rate
- prompt digest
- input/output token distribution
- runtime profile
- network topology
- client identity
- latency percentiles
- throughput metrics
- failure rate
- fairness evidence
- quality evidence
- server binary digest
- model digest
- server capability digest

## Hard Drift

Hard drift requires retuning or refusal:

- binary or model digest changed
- unsupported server option
- different workload shape
- different concurrency class
- different network topology for network-bound recommendations
- different server capability digest
- insufficient memory
- kernel or architecture mismatch

Loopback recommendations cannot be reused as proof of remote-network
improvement. Shared-prefix recommendations apply only to the shared-prefix
workload identity.

## Remote-Client Drift

Phase 21D1 RecommendationBundles are bound to the `loopback` topology. When the
current topology is `c7g_server_t4g_remote_client`, reuse validation must
return hard drift and must not launch `run --best` from the loopback bundle.

Phase 21D2B requires fresh remote-client evidence:

- fresh remote-client baseline
- fresh remote-client candidate confirmation
- matching server binary, model, capability, workload, and protocol digests
- matching server and client topology fingerprints
- artifact validation on each host and after download

Network-specific RecommendationBundles require network-bottleneck evidence
before IRQ, RPS, XPS, listen-backlog, or other network profile changes become
eligible. A runtime-only result can still be recommended or reported
inconclusive without running network profiles.

## Cleanup

Real serving candidates must follow:

```text
preflight
-> ensure port free
-> start exact candidate
-> wait for health
-> verify effective state
-> warmup
-> measure
-> collect metrics
-> terminate gracefully
-> bounded forced termination when needed
-> verify port release
-> verify no worker remains
-> residual-state scan
```

A failed cleanup makes subsequent results ineligible.

## Phase 21D1 Validation

Phase 21D1 created two eligible loopback RecommendationBundle entries for
`interactive-short` concurrency 4:

```text
maximum_throughput: bb4ce9fdbdf96138
interactive_latency: 30a00122420a0b6b
```

The validation also recorded:

```text
run_best_performs_search: false
workload_drift: hard_drift
concurrency_drift: hard_drift
binary_digest_drift: hard_drift
model_digest_drift: hard_drift
capability_digest_drift: hard_drift
loopback_to_remote_network: hard_drift
cleanup_passed: true
```

No recommendation was created for `balanced_multi_user`, `shared_prefix`, or
`memory_constrained` because those classes lacked a complete eligible fresh
confirmation in this run.
