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
