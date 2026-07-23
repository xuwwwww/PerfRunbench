# Phase 21D1 C7g Loopback Serving Validation

Phase 21D1 validates workload-aware `llama-server` autotuning on the existing
Linux Arm64 C7g / Neoverse-V1 target. This is a loopback serving result with a
target-local load generator. It must not be reused as a remote-network serving
result.

No AWS lifecycle operation, reboot, EBS change, kernel package change,
bootloader change, sysctl/sysfs mutation, IRQ affinity change, THP change,
cgroup change outside process-scoped placement, model download, or llama.cpp
rebuild was performed.

## Source and Target Identity

```text
branch: validation/phase21d1-c7g-loopback-serving
source_under_test: 51a304a6ce83ac5aab4a984b8fc0558e90f8e554
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
pr6_status_at_start: open draft
controller_remote_source_match: true
remote_tracked_worktree_clean: true
target: Linux aarch64 / Neoverse-V1 / 8 logical CPUs
kernel: 6.17.0-1019-aws
page_size: 4096
server_binary_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
model: Qwen2.5-7B Q4_K_M
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
llama_cpp_source_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
```

Private hostnames, IP addresses, private connection details, host-key fingerprints,
model paths, and private artifact locations are intentionally omitted.

## Capability Discovery

The exact pinned `llama-server` binary was inspected through `--help`,
startup evidence, health checks, and endpoint use.

```text
server_capability_digest: 46c690b0a2c9aa6e1adc49dd26bf9cb2b7bbc9017b79af46a82a8e59a155a612
parallel_slots: supported
continuous_batching: supported
generation_threads: supported
batch_threads: supported
context_size: supported
batch_size: supported
micro_batch_size: supported
mmap: supported
mlock: supported
prompt_cache: supported
kv_cache_type: supported
flash_attention: supported
metrics_endpoint: supported
slots_endpoint: supported
speculative_decoding: supported_asset_missing
```

Speculative decoding was audited only. No draft model was downloaded or used.

## Loopback Topology

Primary eligible topology:

```text
load_generator_cpu_set: CPU 0
server_cpu_set: CPUs 1-7
server_threads_max: 7
traffic_mode: closed-loop
client_overload_guard: client CPU utilization must stay below 80% of one CPU
```

The measured client CPU utilization was below 1% of one CPU in the recorded
baseline and confirmed candidate blocks, so the confirmed loopback result was
not client-limited.

## Workloads

| workload | concurrency covered | purpose |
| --- | --- | --- |
| `interactive-short` | 1, 2, 4 | short interactive generation |
| `decode-heavy` | 1, 2, 4 | short prompt, longer decode |
| `prefill-heavy` | 1, 2, 4 | long prompt, short decode |
| `shared-prefix` | 1, 2, 4 | repeated-prefix traffic |

All token counts used server-reported timing/token evidence. Character counts
were not used as token counts.

## Baseline Screening Metrics

Baseline profile:

```text
profile: baseline-slots1-threads7
parallel_slots: 1
threads: 7
threads_batch: 7
ctx_size: 2048
batch_size: 512
ubatch_size: 128
```

| workload | concurrency | output tok/s | TTFT p95 ms | E2E p95 ms | failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| interactive-short | 1 | 17.209 | 59.120 | 1111.008 | 0 |
| decode-heavy | 1 | 17.249 | 58.952 | 3717.600 | 0 |
| prefill-heavy | 1 | 16.754 | 62.148 | 1441.770 | 0 |
| shared-prefix | 1 | 17.188 | 63.316 | 1866.312 | 0 |
| interactive-short | 2 | 15.871 | 1165.109 | 2213.321 | 0 |
| interactive-short | 4 | 14.725 | 3377.536 | 4429.389 | 0 |
| decode-heavy | 2 | 15.858 | 3798.587 | 7479.185 | 0 |
| decode-heavy | 4 | 14.732 | 11243.641 | 14909.058 | 0 |
| prefill-heavy | 2 | 15.489 | 1501.621 | 2870.332 | 0 |
| prefill-heavy | 4 | 14.324 | 4387.007 | 5760.544 | 0 |
| shared-prefix | 2 | 15.771 | 1948.216 | 3775.163 | 0 |
| shared-prefix | 4 | 14.702 | 5663.339 | 7470.011 | 0 |

## Candidate Screening

```text
server_startups: 18 / 24
measured_http_requests: 569 / 600
screened_server_profiles: 10
remote_artifacts_checked_final: 170
remote_artifacts_failed_final: 0
local_artifacts_checked_final: 170
local_artifacts_failed_final: 0
```

Completed screening candidates had zero measured request failures and zero
client-overload flags. Rejected candidates were rejected by workload-specific
guards, startup evidence, or lack of fresh confirmation.

| family | best observed screening point | outcome |
| --- | --- | --- |
| parallel slots / continuous batching | `slots4-cont-on`, `interactive-short` c4, 21.142 output tok/s, TTFT p95 1478.051 ms | promoted to confirmation |
| threads | 4-thread and 6-thread variants regressed throughput and latency | ineligible |
| batch / micro-batch | `slots2-batch1024-ub256` improved no confirmed class and regressed versus baseline on prefill-heavy c4 | ineligible |
| context capacity | no larger context was needed by the qualified workloads | not expanded |
| mmap / no-mmap | `slots2-no-mmap` matched baseline within noise at concurrency 1 | inconclusive |
| mlock / prefault | not exercised because no lock-limit/effective-prefault protocol was qualified in this run | not evaluated |
| shared-prefix | `shared-prefix-cache-on-slots4` showed a screening point estimate, but no fresh confirmation | no recommendation |
| KV-cache | `slots2-kv-q8` failed startup; runtime log reported V-cache quantization requires flash attention | rejected |
| flash attention / backend flags | not used for a recommendation; KV-cache interaction remained unqualified | not evaluated |
| speculative decoding | supported flag surface, but draft asset missing and no download authorized | audit only |

## Confirmed Recommendations

Only `interactive-short` at concurrency 4 produced eligible loopback
recommendations.

### Maximum Throughput

```text
recommendation_id: bb4ce9fdbdf96138
profile: slots4-cont-on
workload: interactive-short
concurrency: 4
baseline_output_tok_s: 16.390
candidate_output_tok_s: 27.470
throughput_improvement: 67.602%
baseline_TTFT_p95_ms: 3376.619
candidate_TTFT_p95_ms: 300.300
TTFT_p95_improvement: 91.106%
baseline_E2E_p95_ms: 4423.193
candidate_E2E_p95_ms: 3310.010
E2E_p95_improvement: 25.167%
failures: 0
client_limited: false
```

### Interactive Latency

```text
recommendation_id: 30a00122420a0b6b
profile: slots4-cont-on
workload: interactive-short
concurrency: 4
baseline_output_tok_s: 16.423
candidate_output_tok_s: 27.226
throughput_improvement: 65.772%
baseline_TTFT_p95_ms: 3362.692
candidate_TTFT_p95_ms: 300.476
TTFT_p95_improvement: 91.064%
baseline_E2E_p95_ms: 4406.052
candidate_E2E_p95_ms: 3319.602
E2E_p95_improvement: 24.658%
failures: 0
client_limited: false
```

The same physical server profile can serve multiple result classes, but the
recommendations remain bound to the exact workload, concurrency, loopback
topology, binary, model, and capability digest.

## Recommendation Classes

```text
interactive_latency: 1 eligible recommendation
maximum_throughput: 1 eligible recommendation
balanced_multi_user: 0 eligible recommendations
shared_prefix: 0 eligible recommendations
memory_constrained: 0 eligible recommendations
```

The balanced multi-user decode-heavy finalist did not produce an eligible
recommendation because the remaining measured-request budget was insufficient
for a complete candidate-side confirmation block. The shared-prefix screening
point estimate was not promoted to an eligible recommendation without fresh
confirmation.

## Passport and Reuse Guards

Serving RecommendationBundle and Performance Passport extension artifacts bind
the recommendations to:

- server binary digest
- model digest
- server capability digest
- workload ID and prompt digest
- concurrency
- loopback topology
- parallel slots and continuous batching state
- thread, batch, micro-batch, context, mmap/cache/KV state
- latency and throughput measurements
- zero-failure quality evidence
- cleanup evidence

Validated reuse policy:

```text
run_best_performs_search: false
workload_drift: hard_drift
concurrency_drift: hard_drift
binary_digest_drift: hard_drift
model_digest_drift: hard_drift
capability_digest_drift: hard_drift
loopback_to_remote_network: hard_drift
```

Loopback recommendations cannot be reused as Phase 21D2 remote-network
recommendations.

## Cleanup

Final cleanup evidence:

```text
no_llama_server: true
no_llama_bench: true
no_load_generator: true
selected_port_free: true
active_tuning_lease: none
dirty_host_marker: none
kernel_unchanged: true
page_size_unchanged: true
kernel_cmdline_unchanged: true
tracked_worktree_clean: true
```

## Limitations

- This is a target-local loopback result. It does not prove remote-client
  throughput, latency, or network behavior.
- The positive recommendation is workload-specific: `interactive-short`,
  concurrency 4.
- The result does not identify a universal best `llama-server` configuration.
- Shared-prefix and balanced multi-user classes need fresh confirmations before
  any eligible recommendation.
- `mlock`, explicit prefault, flash-attention backend behavior, and speculative
  decoding remain unqualified for this target.
- The result is preliminary and single-target.

## Completion Flags

```text
phase21d1_c7g_loopback_serving_validation: passed
serving_throughput_improvement_verified: true
serving_latency_improvement_verified: true
serving_capacity_improvement_verified: false
shared_prefix_improvement_verified: false
eligible_serving_recommendations_created: true
host_restored_to_original_state: true
ready_for_phase21d2_remote_network_validation: true
ready_to_stop_c7g: true
```
