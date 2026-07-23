# Phase 21D1 C7g Loopback Plan

Phase 21D1 has completed. The validated result is recorded in
`docs/results/PHASE_21D1_C7G_LOOPBACK_SERVING_VALIDATION.md`.

This file remains as the execution-plan boundary for evaluating whether future
reruns still match the intended loopback protocol.

## Required Preconditions

- source branch contains Phase 21D0 commits
- C7g target is Linux Arm64 / Neoverse-V1
- controller and target source commits match exactly
- pinned `llama.cpp` source and server binary identity are unchanged
- model digest is unchanged
- host is restored to the original 4K state
- no stale `llama-server` process exists
- selected port is free
- memory and context preflight pass

## Execution Outline

```text
capability discovery
-> baseline at concurrency 1 for each standard workload
-> parallel slot / continuous batching stage
-> thread allocation stage
-> batch and micro-batch stage
-> context and capacity stage
-> mmap / mlock / prefault stage
-> shared-prefix cache stage when supported
-> optional KV-cache profiles when supported
-> fresh loopback confirmation
-> Pareto frontier
-> serving Passport / RecommendationBundle evidence
```

Loopback results may support loopback serving recommendations only. They are
not proof of remote-network improvement.

## Validated Outcome

```text
source_under_test: 51a304a6ce83ac5aab4a984b8fc0558e90f8e554
target: Linux aarch64 / Neoverse-V1 / 8 logical CPUs
kernel: 6.17.0-1019-aws
page_size: 4096
server_startups: 18 / 24
measured_http_requests: 569 / 600
eligible_recommendations: 2
remote_artifact_revalidation: 170/170 passed
local_artifact_revalidation: 170/170 passed
host_restored_to_original_state: true
```

The eligible recommendations apply only to `interactive-short` concurrency 4
under the loopback topology where the load generator is pinned to CPU 0 and
`llama-server` is pinned to CPUs 1-7.
