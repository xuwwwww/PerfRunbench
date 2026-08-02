# ArmTune Plain-Language Glossary

This glossary explains the terms used in ArmTune reports, charts, and CLI
output. Run `autotuneai arm explain [term]` for the same core definitions at the
terminal.

## Inference Stages

### Prefill

The model reads and encodes the input tokens before generating the first output
token. Long prompts, retrieved documents, and large conversation histories
increase prefill work.

Common measurements:

- prefill tokens per second: higher is better;
- TTFT: lower is better;
- memory bandwidth and queueing evidence.

### Decode

The model generates output tokens one at a time after prefill. Long answers
spend most of their inference time in decode.

Common measurements:

- decode or output tokens per second: higher is better;
- end-to-end latency: lower is better;
- failed or incomplete generations.

## Workload Profiles

These profiles are controlled traffic shapes, not model personalities or build
profiles.

### `interactive-short`

A short prompt and short-to-medium response, similar to a user asking an
assistant a normal question. It emphasizes responsiveness. ArmTune reads p95
TTFT, end-to-end latency, output throughput, failures, and output guards.

### `decode-heavy`

A short prompt and longer response. Prompt setup is small relative to token
generation, so sustained decode throughput and concurrent scheduling dominate.
ArmTune still rejects a throughput gain when latency or correctness guards
regress beyond policy.

### `prefill-heavy`

A long prompt and shorter response. Prompt ingestion, memory traffic, queueing,
and TTFT dominate. A continuous-batching profile can improve decode traffic but
regress this workload, which is why ArmTune does not create one universal
recommendation.

## Runtime Terms

### Continuous batching

The server schedules active requests together at token boundaries instead of
waiting for one fixed batch to complete. This often raises multi-user decode
throughput, but it can increase queueing or prompt-processing latency.

### Slot

A server execution lane for an active request. `slots4-cont-on` means four
parallel request slots with continuous batching enabled. It does not mean four
models or four machines.

### Thread count

The number of CPU worker threads assigned to the inference runtime. More
threads are not always faster because synchronization, memory bandwidth, and
client/server CPU contention can dominate.

### Affinity

The CPU placement policy for a process or its workers. ArmTune tests bounded
policies such as scheduler default, compact, spread, or reserving one CPU for
the client. It records the effective placement rather than treating a requested
mask as evidence.

## Metrics

### TTFT

Time to first token. It measures the delay between sending a request and
receiving the first generated token. Lower is better.

### E2E latency

End-to-end latency. It measures the complete request duration from submission
until the final response token arrives. Lower is better.

### Median

The middle value after ordering samples. It is less sensitive to an isolated
extreme sample than a mean.

### p95

The 95th percentile. Ninety-five percent of measured samples are at or below
this value. For latency, p95 exposes the slow tail that a median can hide.

### Variability

The spread between repeated measurements. High variability reduces confidence
that a small observed improvement is repeatable.

## Statistical And Safety Guards

### Noise band

A configured range around the baseline where a difference is treated as
measurement noise. A candidate inside the band is `inconclusive`, not faster.

### Paired confirmation

Baseline and candidate blocks are measured under the same target identity and
protocol, often in deterministic randomized order with fresh restarts. Pairing
reduces bias from time drift.

### Bootstrap interval

An uncertainty interval produced by repeatedly resampling measured pairs. A
positive point estimate is not sufficient when the interval still includes
zero.

### Hard drift

A material environment mismatch such as CPU class, logical CPU count, model
digest, runtime digest, or protocol identity. Hard drift blocks recommendation
reuse and workload launch through `run --best`.

### Performance Passport

A versioned, auditable record that binds a result to target, model, runtime,
protocol, distributions, guards, provenance, and artifact digests. It is not a
claim that the configuration is globally optimal.

### `eligible_preliminary`

The evidence cleared the implemented guards for the tested scope, but the
result remains preliminary because it does not establish broad portability or
global optimality.

## Reading The Current C7g Evidence

Across the same tested C7g size-scaling protocol, four-slot continuous batching
improved eligible `interactive-short` and `decode-heavy` serving. The same
family regressed `prefill-heavy` screening on all three sizes and was rejected
for that workload. The negative result is part of the product evidence: ArmTune
kept the fast profile from becoming an unsafe universal recommendation.
