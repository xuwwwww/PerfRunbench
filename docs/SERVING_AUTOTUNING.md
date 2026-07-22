# Workload-Aware Serving Autotuning

Phase 21D0 prepares PerfRunbench for workload-aware `llama-server` tuning. It
does not measure a real server result. Real C7g execution starts only in
Phase 21D1 after the target is manually started and private connection details
are supplied outside tracked files.

## Scope Shift

The fixed single-request path has completed compiler/backend tuning, runtime
tuning, reversible Linux system tuning, 4K boot-time isolation, 4K-versus-64K
kernel comparison, and 64K boot-time isolation. Do not continue searching those
same single-request kernel profiles for this fixed target.

Serving optimization is a separate workload scope:

```text
server capability discovery
-> workload and traffic profile selection
-> bounded staged candidate planning
-> deterministic load generation
-> per-request evidence
-> protocol-specific metrics
-> Pareto frontier
-> named recommendation classes
-> serving Passport / RecommendationBundle evidence
```

There is no universal best server configuration. PerfRunbench keeps separate
recommendation classes for `interactive_latency`, `balanced_multi_user`,
`maximum_throughput`, `shared_prefix`, and `memory_constrained`. A class may
have no eligible recommendation.

## Capability Discovery

The real phases must discover exact capability from the pinned server binary:

- exact `--help` output
- startup logs
- supported HTTP endpoints
- supported request fields
- runtime metrics and slot endpoints when exposed
- build and backend evidence

Unavailable options are recorded as `not_available` and are not used for
candidate generation. Exact option names are preserved in the
`ServerCapabilityReport`.

## Search Families

Candidate generation is staged and bounded, not Cartesian:

1. baseline
2. parallel slots and continuous batching
3. generation and batch threads
4. batch and micro-batch
5. context and slot capacity
6. mmap, mlock, no-mmap, and prefault
7. prompt cache and shared-prefix reuse
8. KV-cache profiles
9. loopback confirmation
10. remote-network validation in Phase 21D2

Each stage preserves the baseline, filters unsupported candidates, applies
memory and context guards, and retains only Pareto-relevant candidates.

## Workload Classes

The built-in workload identities are:

| workload | Purpose |
| --- | --- |
| `interactive-short` | Interactive TTFT and E2E latency at concurrency 1, 2, and 4. |
| `decode-heavy` | Decode throughput under concurrent short-prompt requests. |
| `prefill-heavy` | Prompt-processing and queueing behavior for long prompts. |
| `shared-prefix` | Prefix-cache or prefix-reuse qualification for repeated-prefix traffic. |

Shared-prefix claims are scoped only to the shared-prefix workload identity.
Capacity improvements are not raw speedups.

## Phase 21D0 Commands

Phase 21D0 commands are local and fixture/plan-only:

```bash
autotuneai serve doctor --json
autotuneai serve workload show interactive-short --json
autotuneai serve plan --workload interactive-short --traffic closed-loop --concurrency 4
autotuneai serve search --workload interactive-short --candidate-cap 8 --fixture
autotuneai serve pareto --experiment results/serving/search
autotuneai serve network-plan
autotuneai serve cleanup
```

`serve run` and `serve search` require `--fixture` in Phase 21D0. Real server
execution is intentionally blocked until Phase 21D1.
