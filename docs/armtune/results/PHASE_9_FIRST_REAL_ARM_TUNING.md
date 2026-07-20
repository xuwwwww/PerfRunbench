# Phase 9 First Real Arm Build And Runtime Tuning

## Scope

This document records the first real ArmTune build and runtime autotuning run
after the Phase 9 refocus on Arm build and runtime tuning.

The run used the existing user-provisioned C7g / Graviton3 Linux Arm64 target,
the existing Qwen2.5-7B Q4_K_M model, the pinned llama.cpp source tree, the
existing `llama-bench` and `llama-server` pipelines, the staged runtime search
engine, the build-profile evidence framework, and the Performance Passport
framework.

Workload-specific semantic model qualification and prompt research were out of
scope. Quality status is intentionally `not_evaluated` for the default Phase 9
tuning path.

## Code And Target State

```text
controller_commit: f17297111a66716d0f65c2a4336ebf8f4ea817a6
remote_PerfRunbench_commit: f17297111a66716d0f65c2a4336ebf8f4ea817a6
llama_cpp_source_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
target_family: AWS C7g / Graviton3
target_instance_class: c7g.2xlarge
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1019-aws
target_architecture: aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
online_cpus: 0-7
numa_nodes: 1
production_benchmark_ready: true
model_repository: Qwen/Qwen2.5-7B-Instruct-GGUF
model_basename: qwen2.5-7b-instruct-q4_k_m.gguf
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
result_origin: measured
strict_ssh_host_key_checking: true
```

The private target configuration, public IP address, SSH key path, identity-file
contents, private absolute controller paths, and model contents are
intentionally omitted.

## Commands

The public command shapes were:

```text
autotuneai arm doctor --target-config <redacted-private-target-config> --require-production
autotuneai arm inspect-host --target-config <redacted-private-target-config>
autotuneai arm build-plan --config <redacted-build-profile-config>
autotuneai arm build-profile --config <redacted-build-profile-config>
autotuneai arm optimize --config <redacted-llama-bench-profile-config>
autotuneai arm optimize --config <redacted-llama-bench-runtime-search-config>
autotuneai arm optimize --config <redacted-llama-server-secondary-config>
autotuneai arm profile-compare --config <redacted-profile-comparison-config>
autotuneai arm passport --comparison-result <profile_comparison_result.json>
autotuneai arm verify-passport --passport <performance_passport.json>
```

No llama.cpp source update, model download, new quality suite, prompt update,
semantic qualification, cloud provisioning, or AWS lifecycle action was part of
this evidence commit.

## Build Evidence

Three profiles were built with fixed CMake invocations on the Linux Arm64
target. Raw configure/build stdout and stderr, `CMakeCache.txt`,
`compile_commands.json`, `ldd` output, binary size, and binary SHA-256 were
preserved in the private evidence bundle and summarized through ArmTune's
versioned build-profile inspection result.

```text
build_profile_run_id: 20260719_143632_632820
build_profile_result_origin: measured
build_profile_artifact_validation: passed
```

| profile | build status | KleidiAI evidence state | llama-server SHA-256 | size bytes |
| --- | --- | --- | --- | ---: |
| `portable-baseline` | `completed` | `unavailable` | `cf923c3fccff018eda2f413d843e4b19629472ccd5fd60926c3b163b6d02cc26` | 72488 |
| `arm-native` | `completed` | `unavailable` | `47d3511f52871ebe71e8266403cf273e966c9932eecdccf1be5d0b8be0e48cd8` | 72488 |
| `arm-optimized` | `completed` | `enabled_and_evidenced` | `c01454be6cdebb4243eb0b2833c4da3eb647f3e01fe40d7dc215e14a602b7d17` | 72488 |

`arm-optimized` is marked `enabled_and_evidenced` because the inspected CMake
cache and compile-command evidence include the requested KleidiAI build option
and `GGML_USE_CPU_KLEIDIAI` / KleidiAI include-path evidence. This is not a
global KleidiAI performance claim and does not prove global optimality.

## llama-bench Build-Profile Comparison

Each profile was measured with the same Linux Arm64 target, Qwen2.5-7B model,
`prompt_tokens=512`, `generation_tokens=128`, `threads=8`, `repetitions=1`,
warm-up exclusion, six measured samples, and measured provenance. All artifact
validations passed and no trials failed.

| profile | run ID | prefill median tok/s | decode median tok/s | decode p95 | decode min | decode max | decode stdev | failed trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `portable-baseline` | `20260719_223950_839601` | 13.440 | 10.825 | 10.840 | 10.800 | 10.840 | 0.018348 | 0 |
| `arm-native` | `20260719_225534_336499` | 43.310 | 17.530 | 17.5575 | 17.480 | 17.560 | 0.030822 | 0 |
| `arm-optimized` | `20260719_230502_878403` | 43.305 | 17.380 | 17.450 | 17.360 | 17.470 | 0.040988 | 0 |

Relative to `portable-baseline`, `arm-native` improved observed decode median
by `61.939954%` under this fixed `llama-bench` protocol. `arm-optimized`
improved observed decode median by `60.554273%`, but did not beat
`arm-native` in this run.

## Bounded Runtime Search

The bounded runtime search used the `arm-native` `llama-bench` binary because
it was the best observed build profile in the fixed throughput comparison.

```text
runtime_search_run_id: 20260719_231546_862081
primary_metric: decode_tokens_per_second
quality_gate_status: not_evaluated
candidate_count: 10
repetition_count: 50
result_origin: measured
artifact_validation: passed for all repetitions
failed_repetition_count: 0
noise_band_percent: 2.0
recommendation_status: recommended
selected_candidate: affinity-compact-0_1_2_3_4_5_6_7
```

The configured baseline used `threads=6`, scheduler default placement, and no
extra batch parameters. The staged engine covered the requested `1`, `2`, `4`,
`6`, and `8` thread points and also generated `threads-7` from the existing
topology-aware deterministic thread-count logic. This was not a full Cartesian
product.

| candidate | decode median tok/s | valid samples | failed samples |
| --- | ---: | ---: | ---: |
| `baseline` | 13.630 | 22 | 0 |
| `threads-1` | 2.695 | 2 | 0 |
| `threads-2` | 5.335 | 2 | 0 |
| `threads-4` | 9.640 | 2 | 0 |
| `threads-7` | 15.640 | 2 | 0 |
| `threads-8` | 17.495 | 6 | 0 |
| `batch-batch_size512-ubatch_size512` | 17.180 | 2 | 0 |
| `affinity-compact-0_1_2_3_4_5_6_7` | 17.480 | 7 | 0 |
| `affinity-spread-0_1_2_3_4_5_6_7` | 17.270 | 2 | 0 |
| `affinity-reserve-one-core-0_1_2_3_4_5_6` | 12.450 | 2 | 0 |

Noise-band decision:

```text
baseline_median_decode_tokens_per_second: 13.630
selected_median_decode_tokens_per_second: 17.480
observed_improvement_percent: 28.246515
classification: improved
message: candidate median is above the configured noise band
```

`threads-8` had a slightly higher observed decode median than the staged
selected compact-affinity candidate (`17.495` versus `17.480` tok/s). The
difference is only `0.085812%`, well inside the configured 2% noise band. The
result is therefore reported as a staged-search recommendation, not a global
optimum.

## llama-server Secondary Measurements

`llama-server` was used only for secondary runtime evidence: request
completion, nonempty output, stream parsing, TTFT, end-to-end latency, cleanup,
and loopback port release. Workload-specific semantic correctness was not
required and quality status remained `not_evaluated`.

All three profiles used:

```text
threads: 8
context_size: 1024
batch_size: 512
ubatch_size: 512
affinity_mode: compact
cpu_affinity: 0-7
temperature: 0.0
seed: 42
prompt_suite: configs/armtune/prompt_suite.example.json
measured_requests_per_profile: 9
```

| profile | run ID | TTFT median ms | TTFT p95 ms | E2E median ms | E2E p95 ms | nonempty outputs | cleanup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `portable-baseline` | `20260720_004101_288713` | 97.260 | 2612.5034 | 2114.831 | 4135.1272 | 9/9 | `passed` |
| `arm-native` | `20260720_004503_508540` | 62.826 | 717.1202 | 1325.070 | 1764.9424 | 9/9 | `passed` |
| `arm-optimized` | `20260720_004853_386484` | 63.318 | 718.9682 | 1329.552 | 1788.9090 | 9/9 | `passed` |

## Performance Passport

The profile-comparison step consumed the server secondary measurements and the
build-profile evidence.

```text
profile_comparison_run_id: 20260720_005316_963862
baseline_profile_id: portable-baseline
profile_count: 3
eligible_profile_count: 3
recommendation_state: recommended
recommended_profile_id: arm-native
result_origin: measured
```

Profile-comparison TTFT decision:

```text
baseline_median_ttft_ms: 97.260
candidate_median_ttft_ms: 62.826
observed_ttft_improvement_percent: 35.404072
classification: improved
noise_band_percent: 2.0
```

Passport:

```text
passport_run_id: 20260720_005329_881317
claim_eligibility_state: eligible_preliminary
result_origin: measured
verification_status: passed
verify_passport_checked_items: 6
verify_passport_issues: []
```

This supports the bounded wording:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

It does not support a claim of global optimality, cross-instance portability,
workload-specific semantic correctness, Performix evidence, or formal
competition-final performance.

## Artifact And Cleanup Validation

Independent manifest revalidation was run after artifact download and Passport
generation:

```text
artifact_manifest_count_revalidated: 82
artifact_manifest_failed_count: 0
```

Cleanup verification after the run:

```text
residual_llama_server_processes: none observed
residual_armtune_worker_processes: none observed
residual_listening_llama_ports: none observed
cleanup_check_exit_code: 0
```

## Before / After Summary

| comparison | before | after | observed delta |
| --- | ---: | ---: | ---: |
| `llama-bench` build profile decode median, `portable-baseline` to `arm-native` | 10.825 tok/s | 17.530 tok/s | +61.939954% |
| `llama-bench` runtime search decode median, `threads=6` baseline to selected compact `threads=8` | 13.630 tok/s | 17.480 tok/s | +28.246515% |
| `llama-server` profile comparison TTFT median, `portable-baseline` to `arm-native` | 97.260 ms | 62.826 ms | +35.404072% lower latency |
| `llama-server` E2E median, `portable-baseline` to `arm-native` | 2114.831 ms | 1325.070 ms | +37.343930% lower latency |

These are preliminary measured results on one user-provisioned C7g target, one
model, one llama.cpp source commit, and the listed protocols. They are not
fixture results, but they also are not a final competition-wide claim.

## Limitations

- The result covers one `c7g.2xlarge` target and must not be generalized to all
  Arm systems.
- `llama.cpp` was the first workload adapter, not ArmTune's product identity.
- Workload-specific semantic gates were not evaluated and did not participate
  in default candidate eligibility.
- `arm-optimized` had build evidence for KleidiAI enablement, but did not beat
  `arm-native` in this measured run.
- The staged runtime search found an eligible improvement; it did not exhaust
  all thread, batch, ubatch, affinity, or OS-placement combinations.
- Server TTFT distributions showed high p95 variability for some requests; the
  Passport remains `eligible_preliminary`, not final production evidence.
- No AWS lifecycle action is recorded in this repository evidence.
