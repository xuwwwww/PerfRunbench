# Phase 21D1 C7g Loopback Plan

Phase 21D1 is a future real-target phase. It must not start until the C7g
target is manually started and private connection details are supplied outside
tracked files.

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
