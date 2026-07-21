# Phase 20 C7g Build Backend And Runtime Tuning

Phase 20 validates the CPU-aware `llama.cpp` build-profile workflow on a real
Linux Arm64 C7g target with an existing Qwen2.5-7B Q4_K_M model. This is a
real measured Arm workflow validation, but it did not produce an eligible
speedup claim: the best evidence-valid build and runtime candidates remained
inside the configured 5% noise band.

The result is therefore:

```text
phase20_c7g_build_backend_runtime_tuning: passed
performance_improvement_verified: false
```

## Scope

Target summary:

```text
target_class: user-provisioned AWS C7g.2xlarge / Graviton3
operating_system: Ubuntu 24.04.4 LTS
architecture: Linux aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
memory_class: approximately 15 GiB available RAM
root_filesystem: ext4
```

Source and workload lock:

```text
PerfRunbench validation source: b8bcdd6c33bd1d9b3a5a761f3061df2e02316e73
llama.cpp source commit: 47c786924ad1ab7e91da2cdc72fcdb563780c2bd
llama.cpp tag context: b9999
model: Qwen2.5-7B Q4_K_M
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
```

No model, dataset, `llama.cpp` source update, AWS lifecycle operation, or
system-scoped tuning was performed by PerfRunbench during this validation.

## Source Fixes Required By Real Validation

Two focused source fixes were made before recording this evidence:

```text
2047d90 fix: add explicit Neoverse V1 build profile
b8bcdd6 fix: write Passport artifact manifest
```

The first fix made the C7g CPU target explicit instead of forcing a generic
`explicit-current-cpu` profile name. The second fix made Performance Passport
output independently artifact-manifest validated like the benchmark and search
outputs.

## Build Profiles

Build run:

```text
run_id: 20260721_040809_445855
profile_count: 5
completed_profile_count: 4
artifact_validation_status: passed
```

| profile | build status | eligibility in comparison | KleidiAI evidence | llama-bench SHA-256 | size bytes |
| --- | --- | --- | --- | --- | --- |
| `upstream-default` | `completed` | `eligible` | `unavailable` | `60371f61d38df8b9247a64b6c4db2c0afb867a2ef11c496ff9b77223d572aff3` | `72480` |
| `portable-reference` | `build_failed` | `rejected_build_evidence` | `unavailable` | not produced | not produced |
| `explicit-neoverse-v1` | `completed` | `eligible` | `unavailable` | `6b257db63be87969feb47ed3d79b6e49f61a4b949cbe8b1a2d67f1448c35d695` | `72480` |
| `native-kleidiai` | `completed` | `rejected_build_evidence` | `requested_but_runtime_unverified` | `0ddacc52f6f070d7fd30a0f9db7ad889365650f0309118d0e7b9a29736d99725` | `72480` |
| `native-kleidiai-lto` | `completed` | `rejected_build_evidence` | `requested_but_runtime_unverified` | `40b1f78f3a0b9293fd9f19d8e9cf583ff96446f06a0581705c486998d5eeed45` | `70424` |

`native-kleidiai` and `native-kleidiai-lto` were built and measured, but they
were not eligible for recommendation because the run did not prove KleidiAI
runtime backend usage. No KleidiAI speedup claim is made.

## Fixed Build-Profile Comparison

Protocol:

```text
operation: profile_comparison.arm_build_runtime
result_origin: measured
primary_metric: decode_tokens_per_second
secondary_metric: prefill_tokens_per_second
warmup_rounds: 2
measured_rounds: 5
noise_band_percent: 5.0
quality_gate_status: not_evaluated
```

| profile | eligible | decode median tok/s | decode p95 tok/s | prefill median tok/s | failed trials |
| --- | --- | ---: | ---: | ---: | ---: |
| `upstream-default` | yes | `17.71` | `17.734` | `43.31` | `0` |
| `explicit-neoverse-v1` | yes | `17.78` | `17.796` | `43.43` | `0` |
| `native-kleidiai` | no | `17.50` | `17.596` | `43.29` | `0` |
| `native-kleidiai-lto` | no | `17.56` | `17.592` | `43.55` | `0` |
| `portable-reference` | no | not available | not available | not available | `1` |

Recommendation:

```text
status: inconclusive
profile_id: explicit-neoverse-v1
baseline_profile_id: upstream-default
baseline_decode_median_tok_s: 17.71
candidate_decode_median_tok_s: 17.78
delta_percent: 0.395257
noise_band_percent: 5.0
decision: candidate median is within the configured noise band
```

This comparison did not verify a build-profile performance improvement.

## Bounded Runtime Search

Runtime search used the `explicit-neoverse-v1` build because it was the best
observed evidence-valid build profile. The search kept a bounded candidate set
instead of running a full Cartesian product.

Protocol:

```text
result_origin: measured
build_profile_id: explicit-neoverse-v1
llama-bench_sha256: 6b257db63be87969feb47ed3d79b6e49f61a4b949cbe8b1a2d67f1448c35d695
candidate_count: 6
max_unique_candidates: 6
warmups_excluded: true
search_repetitions_per_candidate: 5
confirmation_repetitions: 5
complete_confirmation_repetitions: 7
noise_band_percent: 5.0
quality_gate_status: not_evaluated
```

The first runtime-search attempt correctly refused to launch when the expected
binary digest did not match the selected v3 build. The corrected run then
executed the six-candidate plan below.

| candidate | threads | affinity | batch/ubatch | decode median tok/s | decode p95 tok/s | prefill median tok/s | failed trials |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `baseline` | `8` | scheduler default | default | `17.72` | `17.736` | `43.44` | `0` |
| `batch-b512-u256` | `8` | scheduler default | `512/256` | `17.65` | `17.682` | `41.73` | `0` |
| `threads-4` | `4` | scheduler default | default | `9.76` | `9.77` | `23.54` | `0` |
| `threads-4-compact-0_1_2_3` | `4` | compact `0-3` | default | `9.76` | `9.778` | `23.52` | `0` |
| `threads-6` | `6` | scheduler default | default | `13.79` | `13.832` | `32.36` | `0` |
| `threads-6-compact-0_1_2_3_4_5` | `6` | compact `0-5` | default | `13.82` | `13.83` | `32.36` | `0` |

Recommendation:

```text
status: inconclusive
selected_candidate: baseline
search_decode_median_tok_s: 17.72
confirmation_decode_median_tok_s: 17.72
complete_confirmation_decode_median_tok_s: 17.72
delta_percent: 0.0
decision: selected candidate did not clear the configured noise band
```

This search did not verify a runtime-search performance improvement.

## Secondary llama-server Measurement

The server benchmark used `llama-server` only as a secondary runtime
measurement. It checked request completion, nonempty output, stream parsing,
TTFT, end-to-end latency, cleanup, and port release. BootTrace semantic
correctness was not evaluated and did not participate in eligibility.

```text
run_id: 20260721_053653_978316
result_origin: measured
benchmark_status: passed
startup_status: ready
readiness_status: ready
request_status: completed
stream_parse_status: parsed
measured_request_count: 3
failed_request_count: 0
median_ttft_ms: 61.470
median_end_to_end_latency_ms: 911.976
cleanup_status: passed
quality_gate_status: not_evaluated
```

These latency numbers are secondary protocol measurements and must not be
combined with `llama-bench` throughput percentages.

## Performance Passport

Passport run:

```text
run_id: 20260721_054511_296268
result_origin: measured
claim_eligibility_state: ineligible_provenance
verification_status: passed
claim_summary: Performance claim is not eligible because required evidence is incomplete or failed.
```

The Passport was generated and verified successfully, but it is not an eligible
performance-improvement Passport because the build-profile comparison was
inconclusive and several build profiles were rejected by build/backend
evidence.

## Artifact Validation

Artifact manifests were validated on the target and again after downloading the
evidence bundle.

```text
remote_artifact_revalidation: 77/77 passed
downloaded_artifact_revalidation: 77/77 passed
evidence_bundle_sha256: 7af8e520d0f80fa5e3b8b484d124ccaeabe3ea42f18a6eee80f6da0dcc12877f
```

No private target address, SSH key path, credential, private target config, or
model file is included in this public document.

## Cleanup

The post-run cleanup check found no residual `llama-server`, `llama-bench`,
profile-comparison, runtime-search, or server-secondary benchmark process. The
server benchmark reported `cleanup_status: passed`.

## Claim Boundary

Allowed wording:

> Phase 20 validated PerfRunbench's CPU-aware `llama.cpp` build-profile,
> backend-evidence, runtime-search, server-measurement, Passport, and artifact
> validation pipeline on a real Linux Arm64 C7g / Neoverse-V1 target.

Disallowed wording:

- PerfRunbench found a faster C7g configuration in Phase 20.
- KleidiAI was enabled or accelerated the workload.
- `explicit-neoverse-v1` is globally better than `upstream-default`.
- The Phase 20 server latency can be combined with throughput deltas.
- The C7g result is portable to T4g, Windows, or another Arm target without
  portability and cache validation.
