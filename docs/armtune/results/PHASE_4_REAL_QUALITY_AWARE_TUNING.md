# Phase 4 Real Quality-Aware Server Tuning Evidence

## Scope

This document records a bounded real-target validation of the Phase 4
quality-aware `llama-server` tuning MVP on a user-provisioned AWS T4g /
Graviton2 development instance.

This run validates the real SSH controller path, real Linux Arm64 target
execution, repeated `llama-server` candidate execution, target-local streaming
TTFT and end-to-end latency measurement, deterministic BootTrace quality-gate
evaluation, quality-aware candidate rejection, cleanup, downloaded artifacts,
and independent digest validation.

This run did not produce an eligible recommendation. Every candidate completed
execution and cleanup, but every candidate failed the deterministic quality
gate. The result is evidence that ArmTune rejects faster or measured candidates
when quality evidence is invalid; it is not evidence of a quality-preserving
optimization.

## Public Evidence

```text
run_id: 20260719_035956_457526
created_at_utc: 2026-07-18T20:10:40Z
target_type: ssh
target_family: AWS T4g / Graviton2
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1019-aws
cpu_model: Neoverse-N1
logical_cpus: 2
numa_nodes: 1
PerfRunbench_commit: 33504d731bdb11316d439609cb3bf3145401b9e8
llama_cpp_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama_server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
model_sha256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
result_origin: measured
runtime_adapter: llama_server
workload_mode: server
quality_suite_id: synthetic_boottrace_quality_v1
quality_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
prompt_suite_sha256: 93c4b2f13a60a20cc1860bcf403b17272a05dd5231aee30366f7566ad537a143
quality_cases: 5
required_quality_cases: 5
candidate_count: 3
eligible_candidate_count: 0
failed_candidate_count: 3
quality_gate_status: failed
recommendation_state: no_eligible_candidate
cli_exit_code: 1
artifact_validation: passed
checked_artifacts: 224
remote_llama_server_processes_after_run: 0
remote_armtune_worker_processes_after_run: 0
```

The public CLI command shape was:

```text
autotuneai arm optimize --config <redacted-private-server-optimize-config>
```

The private target configuration, public IP address, SSH key path,
identity-file contents, model contents, and private absolute paths are
intentionally omitted.

## Protocol

```text
server_bind: 127.0.0.1
port: auto loopback port
concurrency: 1
temperature: 0.0
seed: 42
context_size: 512
baseline_threads: 2
baseline_batch_size: 128
baseline_ubatch_size: 128
warmup_runs: 1
search_runs: 2
confirmation_runs_configured: 2
confirmation_runs_executed: 0
cooldown_seconds: 1.0
random_seed: 42
noise_band_percent: 5.0
max_candidates: 3
affinity_modes: scheduler-default, compact
```

Confirmation was configured but not executed because no eligible candidate
existed after the bounded search. This is expected for a `no_eligible_candidate`
result.

The Phase 4 MVP implementation uses the following staged order for
`llama-server` search:

```text
baseline
batch_parameters
thread_count
affinity
confirmation of baseline and selected candidate, when an eligible selected
candidate exists
```

The bounded real run tested:

```text
baseline
server-batch256-ubatch128
threads-1
```

## Candidate Results

Measured requests only are included in latency distributions. Warm-up requests
are preserved as artifacts and excluded from these distributions.

| candidate | stage | threads | batch | ubatch | eligible | eligibility | quality | measured | failed | median TTFT ms | p95 TTFT ms | median E2E ms | p95 E2E ms |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 2 | 128 | 128 | false | rejected_quality | failed | 10 | 0 | 43.8675 | 50.4393 | 2796.8285 | 3048.82065 |
| server-batch256-ubatch128 | batch_parameters | 2 | 256 | 128 | false | rejected_quality | failed | 10 | 0 | 42.8005 | 49.16655 | 2776.765 | 2874.91025 |
| threads-1 | thread_count | 1 | 128 | 128 | false | rejected_quality | failed | 10 | 0 | 71.479 | 79.90565 | 5086.705 | 5315.07015 |

All candidates had:

```text
validation_status: passed
startup_status: ready
readiness_status: ready
request_status: completed
stream_parse_status: parsed
benchmark_status: passed
cleanup_status: passed
failed_request_count: 0
```

All candidates were rejected only after deterministic quality validation.

## Quality Gate Outcome

The public synthetic BootTrace suite contains five required cases:

```text
boot_success
boot_retry_recovered
boot_config_missing
boot_device_timeout
boot_firmware_component_failure
```

The suite uses deterministic validators including JSON validity, required
schema fields, exact fields, allowed status values, required substrings,
forbidden substrings, regex checks, numeric ranges, and evidence-line ID
validation.

The real model output did not satisfy the strict canonical output contract:

- Outputs were wrapped in Markdown JSON fences instead of raw JSON.
- Failure cases were repeatedly labeled as `success`.
- At least one boolean field was emitted as a string.
- Required canonical substrings were missing.
- JSON validity rate was `0.0` because the validator intentionally requires
  raw JSON, not fenced JSON.

ArmTune therefore produced:

```text
required_quality_cases_passed: 0/5
quality_gate_status: failed
recommendation_state: no_eligible_candidate
recommendation_message: No candidate is eligible because execution evidence or the quality gate failed.
```

The validators were not weakened to force a pass.

## Artifact Integrity

The controller downloaded the completed `server_optimize` artifact tree and
independently re-ran the repository artifact-manifest validator. Validation
passed with 224 checked artifact records and no issues.

The artifact tree includes:

```text
optimize_job.json
effective_config.json
quality_suite_manifest.json
quality_suite_snapshot.json
prompt_suite_snapshot.json
server_optimize_result.json
artifact_validation.json
artifact_manifest.json
candidates/<candidate-id>/search/0001/server_job.json
candidates/<candidate-id>/search/0001/benchmark_result.json
candidates/<candidate-id>/search/0001/artifact_manifest.json
candidates/<candidate-id>/search/0001/raw/...
candidates/<candidate-id>/search/0001/requests/...
```

No model contents are stored in the ArmTune artifacts.

## Claim Boundary

This evidence supports only the following claim:

> ArmTune's real SSH controller path, Linux Arm64 `llama-server` worker
> execution, streaming TTFT and end-to-end latency measurement, deterministic
> BootTrace quality-gate evaluation, quality-aware candidate rejection,
> cleanup, and artifact digest validation were exercised on a real AWS
> Graviton2 target.

This evidence does not show:

- a quality-preserving performance recommendation
- final ArmTune optimization
- stable production performance
- KleidiAI execution
- build-profile improvement
- Arm optimized-path usage beyond host capability evidence
- model quality for real firmware logs
- Performix evidence
- Performance Passport generation
- formal competition benchmark results

## Follow-Up

The next Phase 4 hardening work should improve prompt adherence and/or add a
separate response-normalization policy only if it remains deterministic,
auditable, and explicitly documented. Markdown-fence stripping, if added, must
be treated as a versioned parser behavior and tested against adversarial
outputs; it must not silently hide semantic quality failures.
