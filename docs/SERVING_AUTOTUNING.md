# Workload-Aware Serving Autotuning

Phase 21D0 prepared PerfRunbench for workload-aware `llama-server` tuning.
Phase 21D1 then validated the first real C7g loopback serving search. Loopback
evidence is still topology-specific: it must not be reused as remote-network
evidence.

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

## Phase 21D2A Remote-Network Preparation

Phase 21D2A keeps AWS and remote hosts stopped and prepares the two-host
remote-client contract:

```text
C7g server target -> pinned llama-server
T4g remote client -> deterministic load generator
required serving profile -> slots4-cont-on
server CPU policy -> full capacity; no local load-generator CPU reservation
network tuning -> skipped unless a network bottleneck is observed
```

`autotuneai serve network-plan --json` emits the versioned local plan. The plan
records server/client roles, source-state requirements, artifact policy,
privacy policy, timing policy, and the Phase 21D2B execution order. It is still
plan-only and reports:

```text
aws_accessed: false
remote_accessed: false
```

Loopback RecommendationBundles from Phase 21D1 are hard drift for the
`c7g_server_t4g_remote_client` topology. Phase 21D2B must create a fresh
remote-client baseline and confirmation before any remote-network claim.

## Phase 21D2B Real Remote-Client Result

Phase 21D2B used C7g as the `llama-server` target and T4g as the remote load
generator over a direct private network path. The run confirmed `slots4-cont-on`
on the remote-client topology and measured three paired J2 seven-CPU blocks
plus three paired J1 full-eight-CPU blocks.

```text
source_commit: b46460ee989baf54ebbeff6275ac9fb0bf866f3f
measured_requests_total: 608
paired_block_pairs: 6
retries: 0
request_failures: 0
quality_failures: 0
network_tuning_status: network_tuning_not_executed_evidence_absent
artifact_validation: passed
dual_host_restored: true
```

Eligible remote-client `slots4-cont-on` recommendations were observed for:

| stage | workload | result |
| --- | --- | --- |
| J2 seven CPU | `interactive-short` | output throughput `15.334 -> 28.295 tok/s`, `+84.529%`; TTFT p95 `3389.856 -> 300.557 ms` |
| J1 full eight CPU | `interactive-short` | output throughput `17.032 -> 31.355 tok/s`, `+84.092%`; TTFT p95 `3055.682 -> 271.088 ms` |
| J1 full eight CPU | `decode-heavy` | output throughput `17.046 -> 31.790 tok/s`, `+86.490%`; TTFT p95 `10126.795 -> 202.424 ms` |
| J1 full eight CPU | `prefill-heavy` | output throughput `16.563 -> 28.054 tok/s`, `+69.375%`; TTFT p95 `3967.424 -> 341.693 ms` |

The recommendation remains scoped to the tested server, remote client,
topology, model, binary, workload set, concurrency 4, and protocol. It is not
combined with the Phase 21D1 loopback result or any `llama-bench` measurement.
Network IRQ/RPS/XPS tuning was not executed because the prerequisite
network-bottleneck evidence was absent.

## Phase 21D1 Real Loopback Result

Phase 21D1 used the pinned `llama-server` binary and existing Qwen2.5-7B
Q4_K_M model on a Linux Arm64 C7g / Neoverse-V1 target.

```text
target: Linux aarch64 / Neoverse-V1 / 8 logical CPUs
kernel: 6.17.0-1019-aws
page_size: 4096
server_startups: 18 / 24
measured_http_requests: 569 / 600
capability_digest: 46c690b0a2c9aa6e1adc49dd26bf9cb2b7bbc9017b79af46a82a8e59a155a612
artifact_revalidation: 170/170 passed remotely and after download
```

Eligible loopback recommendations:

| class | workload | concurrency | profile | observed result |
| --- | --- | ---: | --- | --- |
| `maximum_throughput` | `interactive-short` | 4 | `slots4-cont-on` | output throughput `16.390 -> 27.470 tok/s`, `+67.602%` |
| `interactive_latency` | `interactive-short` | 4 | `slots4-cont-on` | TTFT p95 `3362.692 -> 300.476 ms`, `+91.064%` lower latency |

These are not universal server settings. They are bound to the tested binary,
model, server capability digest, workload, concurrency, and loopback topology.
`balanced_multi_user`, `shared_prefix`, and `memory_constrained` produced no
eligible recommendation in Phase 21D1.
