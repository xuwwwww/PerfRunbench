# Serving Load Generator

The serving load generator is a first-class PerfRunbench component. It is not a
shell loop around `curl`.

## Evidence

Every request emits versioned JSONL evidence with:

- request ID
- workload ID
- client ID
- scheduled timestamp
- submitted timestamp
- first-token timestamp
- completed timestamp
- HTTP status
- input tokens
- output tokens
- finish reason
- server-reported timings when available
- client-observed timings
- error category
- response digest
- quality-guard result

Durations use a monotonic clock. Token counts must come from server-reported
usage or tokenizer evidence; character counts are not token counts.

## Client Validity

The load generator records whether the client is likely the bottleneck:

- scheduling delay
- request count
- clock source
- client-overload flag

In future remote mode, server-side timing and client-observed timing must be
kept separate. Timestamps from unsynchronized machines must not be subtracted
to invent queue delay.

## Phase 21D0 Boundary

Phase 21D0 validates deterministic fixture modes only. It does not start a real
`llama-server`, does not contact AWS, and does not modify machine-wide state.

## Phase 21D1 Loopback Evidence

Phase 21D1 exercised the real target-local load generator against a pinned
`llama-server` binary. The load generator process was pinned to CPU 0 while the
server process was pinned to CPUs 1-7.

```text
measured_http_requests: 569
server_startups: 18
request_artifacts: JSONL per workload/concurrency block
streaming_TTFT: recorded
server_reported_tokens: required
client_overload_guard: passed
```

The confirmed recommendations used zero request retries and zero measured
request failures. Client-observed timing remains loopback timing and is not
mixed with future remote-client timing.

## Phase 21D2 Remote Client

Phase 21D2A prepares the remote-client mode without running it. In Phase 21D2B
the load generator runs on the T4g client and the server runs on C7g. The load
generator must preserve:

- client identity and resource saturation evidence
- scheduled, submitted, first-token, and completed monotonic timestamps
- request JSONL and raw response digests
- network counter snapshots from the client side
- explicit `client_observed` timing labels

The server may preserve internal timing and slot metrics, but those timestamps
must remain separate unless a future protocol establishes clock
synchronization. The first real remote run confirms `slots4-cont-on` and
full-capacity serving before any IRQ, RPS, XPS, or listen-backlog profile is
considered.
