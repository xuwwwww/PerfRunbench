# Phase 21D2B C7g and T4g Remote Serving Validation

Phase 21D2B validates a real two-host serving topology:

```text
C7g / Neoverse-V1: pinned llama-server target
T4g / Neoverse-N1 class: lightweight remote load generator
traffic path: direct private network path
```

This is fresh remote-client serving evidence. It does not reuse the Phase 21D1
loopback RecommendationBundle, and it must not be combined with loopback,
single-request `llama-bench`, kernel, or build-profile measurements.

No AWS lifecycle operation, reboot, EBS change, public Security Group
broadening, kernel package change, bootloader change, persistent network
configuration change, model download, model copy to the client, or `llama.cpp`
update was performed by the measurement controller. The benchmark ingress rule
was scoped to the remote-client security group only; public benchmark-port
exposure was not used.

Private public IPs, private IPs, Security Group IDs, SSH key paths, host-key
fingerprints, model paths, and private artifact locations are intentionally
omitted.

## Source and Target Identity

```text
branch: validation/phase21d2b-c7g-t4g-remote-serving
source_under_test: b46460ee989baf54ebbeff6275ac9fb0bf866f3f
controller_head_matches_c7g_head_matches_t4g_head: true
controller_tracked_worktree_clean: true
c7g_tracked_worktree_clean: true
t4g_tracked_worktree_clean: true
c7g_target: Linux aarch64 / Neoverse-V1 / 8 logical CPUs
t4g_target: Linux aarch64 / Neoverse-N1 class / 2 logical CPUs
c7g_kernel: 6.17.0-1019-aws
t4g_kernel: 6.17.0-1017-aws
server_binary_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
model: Qwen2.5-7B Q4_K_M
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
llama_cpp_source_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
```

The T4g client did not contain the GGUF model.

## Protocol

The run used deterministic randomized block order, fresh server startup for
every block, no request retries, zero failed measured requests, and
client-observed TTFT/E2E timing.

```text
traffic_mode: closed-loop
concurrency: 4
warmup_requests_per_workload_per_block: 4
measured_requests_per_workload_per_block: 25
paired_block_pairs: 6
measured_requests_total: 608
measured_requests_in_paired_blocks: 600
client_validity_rehearsal_measured_requests: 8
retries: 0
request_failures: 0
quality_failures: 0
```

The remote-client topology covered:

- `J2`: C7g server on CPUs 1-7, three paired baseline/candidate blocks,
  `interactive-short` only.
- `J1`: C7g server on CPUs 0-7, three paired baseline/candidate blocks,
  `interactive-short`, `decode-heavy`, and `prefill-heavy`.

The `slots4-cont-on` runtime semantics were confirmed through startup command
evidence and the server `/slots` endpoint:

```text
parallel_slots: 4
continuous_batching: enabled
ctx_size: 2048
batch_size: 512
ubatch_size: 128
flash_attention: off
J2_threads: 7
J2_cpu_range: CPUs 1-7
J1_threads: 8
J1_cpu_range: CPUs 0-7
```

Concurrency 8 was not executed because the T4g remote client has only two
logical CPUs and the concurrency-4 protocol already satisfied the remote-client
confirmation requirement without needing to expand the client load shape.

## Results

All numbers below compare only baseline versus `slots4-cont-on` within the
same target, workload, topology, concurrency, and protocol.

| stage | workload | baseline output tok/s | candidate output tok/s | throughput delta | baseline TTFT p95 ms | candidate TTFT p95 ms | TTFT p95 reduction | throughput eligible | latency eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| J2 seven CPU | `interactive-short` | 15.334 | 28.295 | +84.529% | 3389.856 | 300.557 | 91.134% | yes | yes |
| J1 full eight CPU | `interactive-short` | 17.032 | 31.355 | +84.092% | 3055.682 | 271.088 | 91.128% | yes | yes |
| J1 full eight CPU | `decode-heavy` | 17.046 | 31.790 | +86.490% | 10126.795 | 202.424 | 98.001% | yes | yes |
| J1 full eight CPU | `prefill-heavy` | 16.563 | 28.054 | +69.375% | 3967.424 | 341.693 | 91.388% | yes | yes |

The paired bootstrap lower bound was above zero for every measured comparison:

| stage | workload | bootstrap p05 delta | bootstrap median delta | bootstrap p95 delta |
| --- | --- | ---: | ---: | ---: |
| J2 seven CPU | `interactive-short` | +83.957% | +84.778% | +85.599% |
| J1 full eight CPU | `interactive-short` | +83.467% | +84.105% | +84.742% |
| J1 full eight CPU | `decode-heavy` | +86.231% | +86.668% | +87.104% |
| J1 full eight CPU | `prefill-heavy` | +68.817% | +69.547% | +70.281% |

## Recommendation

`slots4-cont-on` is eligible for this remote-client topology under the tested
target, model, workload set, concurrency, binary, and measurement protocol.

```text
remote_slots4_throughput_improvement_verified: true
remote_slots4_latency_improvement_verified: true
full_capacity_remote_recommendation_created: true
```

Recommended wording:

```text
Best observed eligible serving configuration under the tested C7g server,
T4g remote-client topology, Qwen2.5-7B Q4_K_M model, pinned llama-server
binary, workload set, concurrency, and measurement protocol.
```

## Network Tuning Gate

Read-only network snapshots were collected on both hosts before and after
blocks. The run did not show evidence requiring IRQ, RPS, XPS, or backlog
tuning, so no network settings were modified.

```text
remote_network_improvement_verified: false
network_tuning_status: network_tuning_not_executed_evidence_absent
network_tuning_skipped_due_to_no_bottleneck: true
client_bottleneck_detected: false
```

This preserves the Phase 21D2 rule: network tuning is executed only after
network-bottleneck evidence exists. No network mutation was performed merely to
claim it was tested.

## Artifact Validation and Cleanup

The C7g and T4g manifests were validated on the remote hosts before download,
then the downloaded artifacts were independently revalidated by digest and
size.

```text
c7g_downloaded_artifacts_checked: 326
c7g_downloaded_artifacts_failed: 0
t4g_downloaded_artifacts_checked: 272
t4g_downloaded_artifacts_failed: 0
artifact_validation: passed
```

Final independent cleanup probes confirmed:

```text
c7g_llama_server_residual_processes: 0
c7g_http_server_residual_processes: 0
c7g_benchmark_port_listeners: 0
c7g_benchmark_port_connections: 0
t4g_client_agent_residual_processes: 0
t4g_benchmark_port_connections: 0
t4g_large_model_absent: true
dual_host_restored: true
```

## Completion Flags

```text
phase21d2b_remote_serving_validation: passed
remote_slots4_throughput_improvement_verified: true
remote_slots4_latency_improvement_verified: true
full_capacity_remote_recommendation_created: true
remote_network_improvement_verified: false
network_tuning_skipped_due_to_no_bottleneck: true
client_bottleneck_detected: false
dual_host_restored: true
ready_for_phase21d3_packaging: true
ready_to_stop_c7g_t4g: true
```

## Limitations

- This is a two-host C7g/T4g remote-client result, not a public-internet result.
- The recommendation is bound to the tested workload set and concurrency 4.
- It does not prove a universal best `llama-server` configuration.
- Concurrency 8 was not executed in this phase.
- Network tuning was not executed because its prerequisite bottleneck evidence
  was absent.
- Results must not be compared directly with T4g-only, loopback, `llama-bench`,
  kernel, build-profile, or PyTorch training measurements.
