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
