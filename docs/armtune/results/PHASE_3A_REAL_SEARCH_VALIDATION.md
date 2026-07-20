# Phase 3A Real Search Validation Evidence

## Scope

This document records a bounded real-target validation of the Phase 3A
repeated benchmark search pipeline on a user-provisioned AWS Graviton2
development instance.

This run validates the controller, SSH target path, worker execution,
candidate scheduling, repeated measurement, statistics, conservative
recommendation, and downloaded artifact digest validation against a real
`llama-bench` binary and a real GGUF model on Linux Arm64.

This is not a formal competition benchmark result and must not be used as
evidence of final optimization, stable production performance, KleidiAI usage,
quality preservation, build-profile improvement, Performix support, or an
optimal ArmTune configuration.

## Public Evidence

```text
run_id: 20260718_001757_718789
target_type: ssh
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1017-aws
cpu_model: Neoverse-N1
logical_cpus: 2
online_cpus: 0,1
PerfRunbench_commit: ee4cbb305d788e8405e65ab839b1848cb743a2eb
llama_cpp_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama_bench_sha256: 990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8
model_sha256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
result_origin: measured
search_status: completed
quality_gate_status: not_evaluated
parse_statuses: parsed=34
benchmark_statuses: passed=34
failed_repetitions: 0
artifact_validation: passed
artifact_validation_repetitions: 34
checked_artifact_records: 442
fake_executable_marker: false
```

The public CLI commands were:

```text
autotuneai arm doctor --target-config <redacted-target-config> --require-production
autotuneai arm inspect-host --target-config <redacted-target-config>
autotuneai arm optimize --config <redacted-optimize-config>
```

The private target and optimize configuration files, public IP address, SSH key
path, and identity-file contents are intentionally omitted.

## Search Protocol

```text
warmup_runs: 1
search_runs: 3
confirmation_runs: 5
cooldown_seconds: 2.0
random_seed: 42
insert_paired_baseline_every: 2
noise_band_percent: 5.0
primary_metric: decode_tokens_per_second
```

The inspected `llama-bench` help output supported `--batch-size` and
`--ubatch-size`, but did not expose `--threads-batch`. The private optimize
config therefore did not include `threads_batch`.

The current Phase 3A implementation does not yet expose a hard candidate cap or
configurable affinity-mode allowlist. On this 2-vCPU target the resulting
candidate plan contained five candidates:

```text
baseline
threads-1
affinity-compact-0_1
affinity-spread-0_1
affinity-reserve-one-core-0
```

This is still a staged, pruned search and not a full Cartesian product. It is a
known limitation for later Phase 3 hardening that the real-target smoke profile
cannot currently enforce a maximum candidate count from configuration.

## Schedule Evidence

```text
warmup: 1
thread_count: 7
affinity: 16
confirmation: 10
total_repetitions: 34
```

Candidate repetition counts:

```text
baseline: 17
threads-1: 3
affinity-spread-0_1: 8
affinity-compact-0_1: 3
affinity-reserve-one-core-0: 3
```

Warmup exclusion was verified: the baseline appeared in 17 total repetitions,
but the baseline primary-metric distribution had 16 valid samples because the
warmup repetition was excluded from distributions.

Paired baseline insertion was observed in the stage schedules. With
`insert_paired_baseline_every: 2`, baseline repetitions were inserted after
every two non-baseline stage repetitions.

Confirmation compared the retained baseline against the selected observed
candidate:

```text
confirmation baseline repetitions: 5
confirmation affinity-spread-0_1 repetitions: 5
```

## Primary Metric Distributions

Primary metric: `decode_tokens_per_second`.

| candidate | valid | failed | median | p95 | min | max | iqr | stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 16 | 0 | 30.165 | 30.285 | 29.31 | 30.3 | 0.215 | 0.238272 |
| threads-1 | 3 | 0 | 16.08 | n/a | 15.74 | 16.1 | n/a | 0.20232 |
| affinity-compact-0_1 | 3 | 0 | 30.16 | n/a | 30.15 | 30.28 | n/a | 0.072342 |
| affinity-spread-0_1 | 8 | 0 | 30.2 | 30.375 | 27.94 | 30.41 | 0.305 | 0.814559 |
| affinity-reserve-one-core-0 | 3 | 0 | 15.49 | n/a | 15.38 | 15.53 | n/a | 0.077675 |

Candidates with fewer than five valid samples intentionally omit p95 and emit
the small-sample warning recorded in `search_result.json`.

## Recommendation Boundary

```text
selected_candidate: affinity-spread-0_1
recommendation_status: inconclusive
noise_band_classification: inconclusive
delta_percent: 0.116029
noise_band_percent: 5.0
quality_gate_status: not_evaluated
```

The selected candidate was the best observed candidate under the tested
`llama-bench` configuration, but it was not recommended as an improvement
because it was inside the configured 5% noise band. Semantic quality was not
evaluated.

## Evidence Integrity

- `autotuneai arm doctor --require-production` validated the configured target
  as Linux Arm64 production-ready before the search.
- `autotuneai arm inspect-host` reported host capabilities separately from
  workload execution evidence.
- Every repetition preserved raw stdout/stderr artifacts.
- Every repetition reported `result_origin: measured`.
- Every repetition reported `parse_status: parsed`.
- Every repetition reported `benchmark_status: passed`.
- Binary SHA-256 matched the expected digest in every repetition.
- Model SHA-256 matched the expected digest in every repetition.
- Host fingerprint was stable across all repetitions:
  `Linux|aarch64|6.17.0-1017-aws|Neoverse-N1|2|0,1`.
- Downloaded artifact digest validation passed for all 34 repetition
  manifests, covering 442 artifact records.

## Claim Boundary

This evidence supports only the following claim:

> ArmTune's repeated benchmark, staged search, statistics, conservative
> recommendation, SSH worker, provenance, parser, and artifact-validation
> pipeline was validated against a real `llama-bench` workload on a Linux Arm64
> target.

This evidence does not show:

- final ArmTune optimization
- stable production performance
- KleidiAI execution
- Arm optimized-path usage beyond host capability evidence
- quality-gate validation
- build-profile comparison
- llama-server behavior
- TTFT or streaming latency
- Arm Performix evidence
- Performance Passport generation
- formal competition benchmark results

## Privacy Boundary

This public evidence intentionally omits:

- public IP address
- SSH key path
- private controller configuration paths
- credentials or identity-file contents
- model file contents
- raw artifacts containing private absolute paths
