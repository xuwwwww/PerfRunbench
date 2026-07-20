# Phase 5 Competition Evidence MVP Validation

## Scope

This document records a bounded Phase 5 quality-calibration validation on a
user-provisioned AWS T4g / Graviton2 Linux Arm64 development target.

This run validates the Phase 5 real SSH controller path, fixed
`llama-server` worker execution, structured-output capability inspection,
strict BootTrace quality calibration, artifact preservation, cleanup, and
artifact digest validation.

This run did not produce an eligible build-profile comparison or Performance
Passport claim. The measured model/profile remained unsuitable under the strict
raw-JSON BootTrace quality contract, so Phase 5 correctly stopped before
profile comparison.

## Public Evidence

```text
run_id: 20260719_132056_332743
created_at_utc: 2026-07-19T05:22:42Z
operation: quality_calibrate.llama_server
target_type: ssh
target_family: AWS T4g / Graviton2
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1019-aws
cpu_model: Neoverse-N1
logical_cpus: 2
numa_nodes: 1
PerfRunbench_commit: d4f1896acd8446189a219d040f0637d5c76feffb
llama_cpp_runtime_build: version: 10050 (b15ca938a) built with GNU 13.3.0 for Linux aarch64
llama_cpp_source_commit: unknown
llama_server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
model_basename: qwen2.5-0.5b-instruct-q5_k_m.gguf
model_sha256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
result_origin: measured
runtime_adapter: llama_server
workload_mode: server
quality_suite_id: synthetic_boottrace_quality_v1
quality_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
prompt_contract_sha256: 2e0fb2277df14a6c46a89d28524eeb22df41bde953aeaf6611e1bfd72712a974
normalization_enabled: false
normalization_strict: true
quality_cases: 5
required_quality_cases: 5
measured_request_count: 5
request_success_count: 5
raw_json_valid_count: 0
normalized_json_valid_count: 0
required_quality_cases_passed: 0/5
critical_failure_count: 45
quality_gate_status: failed
model_profile_suitability: unsuitable
benchmark_validation_status: passed
benchmark_startup_status: ready
benchmark_readiness_status: ready
benchmark_request_status: completed
benchmark_stream_parse_status: parsed
benchmark_cleanup_status: passed
benchmark_artifact_validation: passed
benchmark_checked_artifacts: 32
quality_calibration_artifact_validation: passed
quality_calibration_checked_artifacts: 40
fixture_marker_detected: false
cli_exit_code: 1
```

The public CLI command shape was:

```text
autotuneai arm quality-calibrate --config <redacted-private-quality-calibration-config>
```

The private target configuration, public IP address, SSH key path,
identity-file contents, private absolute paths, and model contents are
intentionally omitted.

## Structured Capability Evidence

The inspected `llama-server` binary reported:

```text
inspection_help_status: passed
capability_probe: completed
grammar: supported
json_schema: supported
response_format: unsupported
openai_compatible_structured_output: supported
seed: supported
temperature: supported
deterministic_sampling: supported
```

These are capability and command-line inspection results only. They do not
prove that the model followed the requested output contract.

## Quality Calibration Outcome

The public synthetic BootTrace suite contains five required cases:

```text
boot_success
boot_retry_recovered
boot_config_missing
boot_device_timeout
boot_firmware_component_failure
```

The strict Phase 5 calibration policy required raw JSON. No normalization was
enabled, and Markdown-fenced JSON was rejected rather than silently repaired.

The measured output failed the canonical contract:

```text
failure_category_histogram:
  invalid_json: 5
  markdown_fence_present: 5
  model_capability_failure: 5
  response_format_violation: 5
  wrong_root_cause: 5
  wrong_status: 5
```

ArmTune therefore reported:

```text
quality_gate_status: failed
model_profile_suitability: unsuitable
required_quality_cases_passed: 0/5
```

The validators were not weakened to force a pass.

## Artifact Integrity

The run produced nested benchmark artifacts and a Phase 5 quality-calibration
artifact set. Both artifact manifests passed digest validation:

```text
benchmark_artifact_validation: passed
benchmark_checked_artifacts: 32
quality_calibration_artifact_validation: passed
quality_calibration_checked_artifacts: 40
quality_calibration_artifact_issues: 0
```

The artifact tree includes:

```text
quality_calibration_job.json
effective_config.json
quality_suite_manifest.json
quality_suite_snapshot.json
normalization_policy.json
prompt_contract.json
prompt_suite_snapshot.json
benchmark/
quality_calibration_result.json
artifact_manifest.json
artifact_validation.json
```

No model contents are stored in the ArmTune artifacts.

## Claim Boundary

This evidence supports only the following claim:

> ArmTune's Phase 5 real SSH quality-calibration path, Linux Arm64
> `llama-server` worker execution, structured capability inspection, strict
> BootTrace quality rejection, cleanup, and artifact digest validation were
> exercised on a real AWS Graviton2 target.

This evidence does not show:

- a quality-preserving profile recommendation
- Performance Passport eligibility
- final ArmTune optimization
- stable production performance
- KleidiAI execution
- build-profile improvement
- source-level llama.cpp optimization
- model quality for real firmware logs
- Performix evidence
- formal competition benchmark results

## Follow-Up

Before generating an eligible measured Performance Passport, Phase 5 needs a
model/profile that passes the strict or explicitly normalized BootTrace quality
calibration. Only then should ArmTune run profile comparison and Passport
verification for measured evidence.
