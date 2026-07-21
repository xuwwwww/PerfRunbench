# System Benchmark Protocols

System profiles are protocol-specific. PerfRunbench must not combine
throughput, latency, startup, and jitter into one score.

| protocol | primary metric | selected secondary evidence |
| --- | --- | --- |
| `steady_state_decode` | `decode_tokens_per_second` | variability, context switches, migrations, page faults, dTLB/cache evidence when available |
| `cold_model_load` | `model_load_time` | major faults, minor faults, bytes read, peak RSS |
| `warm_first_request` | `ttft_ms` | E2E latency, faults, RSS |
| `interactive_server` | `ttft_median_and_p95` | E2E median/p95, stream parsing, cleanup |
| `concurrent_server` | `aggregate_output_tokens_per_second` | p95 latency, failures, memory headroom, quality guard |

Examples:

- storage read-ahead is relevant to `cold_model_load`, not warm decode
- prefault-warm is startup/TTFT protocol evidence, not decode acceleration
- THP profiles require workload hugepage evidence before being called exercised
- HugeTLB must prove a mapped HugeTLB page, not just a reserved pool

Search is staged and never a full Cartesian product.

## Phase 21B Protocol Boundary

Phase 21B used only `steady_state_decode` for ranking reversible system
profiles:

```text
binary: upstream-default llama-bench
prompt_tokens: 512
generation_tokens: 128
repetitions_per_candidate: 5
noise_band_percent: 5.0
```

No `cold_model_load`, `warm_first_request`, `interactive_server`, or
`concurrent_server` result was ranked in Phase 21B. THP, sysctl, allocator,
scheduler, and affinity results must therefore be read only as steady-state
decode evidence.

The only clear Phase 21B outcome outside the noise band was a regression for
the reduced CPU-set affinity candidate. Positive deltas stayed inside the 5%
noise band and were not promoted.

## Phase 21B2 Protocol Boundary

Phase 21B2 ranked only candidates using the same `steady_state_decode`
protocol as `baseline-current`:

```text
binary: upstream-default llama-bench
prompt_tokens: 512
generation_tokens: 128
baseline_repetitions: 5
noise_band_percent: 5.0
```

`no-mmap`, explicit mmap, THP mmap modes, allocator arena limiting, cpuset
partitions, and IRQ/workqueue isolation were compared only within that
protocol. The largest positive steady-state delta was `no-mmap` at `+3.014%`,
inside the noise band.

Storage scheduler/read-ahead measurements used `cold_model_load`, and
`prefault-warm` used `warm_page_cache_decode`. Those rows were labeled
`protocol_baseline_missing` because they do not have a matching Phase 21B2
baseline and must not be ranked against steady-state decode throughput.

`llama-server` request timing was secondary `interactive_server` smoke
evidence only. It validated request completion, nonempty output, stream
parsing, TTFT/E2E recording, cleanup, and port release; it did not participate
in candidate eligibility.

## Phase 21C2A Boot-Isolation Comparison Identities

Phase 21C2A uses three measurement identities so CPU-allocation effects are not
mistaken for boot-parameter effects:

| identity | protocol role | profile | workload CPUs | generation threads | batch threads |
| --- | --- | --- | --- | --- | --- |
| E1 | full-capacity baseline | G0 `original-4k-default` | `0-7` | 8 | 8 |
| E2 | matched causal baseline | G0 `original-4k-default` | `2-7` | 6 | no more than 6 |
| E3 | experimental profile | G1/G2/G3 | `2-7` | 6 | no more than 6 |

Only E2 versus E3 can support a causal boot-isolation percentage. E1 versus the
best experimental profile is a secondary product-capacity tradeoff and must be
labeled as such.

Phase 21C2A keeps protocols separate:

| protocol | primary use |
| --- | --- |
| `steady_state_decode` | primary throughput classification with the 5% noise band |
| `interactive_server` | secondary warm-server TTFT/E2E evidence |
| `system_noise` | scheduler, IRQ, softirq, migration, and utilization evidence |
| `cold_model_load` | optional secondary cold-load check only |

Throughput, latency, system-noise, and cold-load outcomes must not be combined
into one score or one improvement percentage.
