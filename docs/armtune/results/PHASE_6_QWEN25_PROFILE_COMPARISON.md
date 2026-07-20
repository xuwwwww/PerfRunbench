# Phase 6 Qwen2.5 Qualification and Conditional Profile Comparison Evidence

## Scope

This document records a bounded Phase 6B qualification attempt for
`Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M on a user-provisioned AWS C7g /
Graviton3 Linux Arm64 target.

The run reused the Phase 5 quality-calibration pipeline. No framework code,
quality validators, public prompt suite, or public expected answers were
changed to make the model pass.

Qwen2.5-7B did not qualify under the strict public synthetic BootTrace quality
contract. Phase 6C build-profile comparison and Performance Passport
generation were therefore not executed.

## Public Evidence

```text
run_id: 20260719_152540_406176
created_at_utc: 2026-07-19T07:27:42Z
operation: quality_calibrate.llama_server
target_type: ssh
target_family: AWS C7g / Graviton3
target_instance_class: c7g.2xlarge
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1019-aws
cpu_model: Neoverse-V1
logical_cpus: 8
numa_nodes: 1
usable_memory: approximately 15 GiB
production_benchmark_ready: true
controller_commit: afb6c2898f445139c07f5982dcc8c55d695f63cf
remote_PerfRunbench_commit: ff6a31531db48f423915ae37875afa90c949a0c4
llama_cpp_source_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama_cpp_runtime_build: version: 10050 (b15ca938a) built with GNU 13.3.0 for Linux aarch64
llama_server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
llama_gguf_split_sha256: 5896c7282530fed5ba0a7d264951725fc83f5e72576c9364f97afc6d76f2efb8
model_repository: Qwen/Qwen2.5-7B-Instruct-GGUF
model_repository_revision: bb5d59e06d9551d752d08b292a50eb208b07ab1f
model_basename: qwen2.5-7b-instruct-q4_k_m.gguf
model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
model_size_bytes: 4683073536
result_origin: measured
runtime_adapter: llama_server
workload_mode: server
server_threads: 8
server_context_size: 4096
server_batch_size: 256
server_ubatch_size: 128
generation_seed: 42
generation_temperature: 0.0
request_concurrency: 1
streaming: true
quality_suite_id: synthetic_boottrace_quality_v1
quality_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
prompt_suite_sha256: 30e8cd47dd9f53bd0a774bd500888ec5c013dbf8d5772a27763c59deb93154f8
prompt_contract_sha256: 8c4af720b8e5da01d506c4aeba0dde3b588d21693e04eae59c2230ab0a80e01b
normalization_policy_version: boottrace-normalization-v1
normalization_enabled: false
normalization_strict: true
quality_cases: 5
required_quality_cases: 5
measured_request_count: 5
request_success_count: 5
raw_json_valid_count: 5
normalized_json_valid_count: 0
required_quality_cases_passed: 0/5
critical_failure_count: 22
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
profile_comparison_executed: false
performance_passport_generated: false
```

The public CLI command shape was:

```text
autotuneai arm quality-calibrate --config <redacted-private-quality-calibration-config>
```

The private target configuration, public IP address, SSH key path,
identity-file contents, private absolute paths, and model contents are
intentionally omitted.

## Target Verification

Before model acquisition and qualification, the target was verified through
direct SSH checks and ArmTune's own target validation:

```text
uname_machine: aarch64
kernel: 6.17.0-1019-aws
cpu_model: Neoverse-V1
logical_cpus: 8
numa_nodes: 1
memory_total: 15 GiB
arm_doctor_production_benchmark_ready: true
arm_inspect_host_workload_execution_evidence: not_collected
strict_ssh_host_key_checking: true
```

The target initially had less than the requested 25 GiB of free root-disk
space because old GGUF files from prior private qualification attempts were
still present. Those generated model files were removed before the Qwen2.5
download. No run evidence, source code, or private configuration was deleted.

```text
root_disk_available_before_cleanup: 22 GiB
root_disk_available_after_cleanup: 25 GiB
root_disk_available_after_download: 21 GiB
root_disk_available_after_merge: 16 GiB
```

## Model Acquisition

The split GGUF files were downloaded from the public Hugging Face model
repository using the remote PerfRunbench virtual environment's `hf` CLI:

```text
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_k_m-*.gguf" --local-dir <remote-model-dir>
```

Downloaded split files:

| file | size bytes | sha256 |
| --- | ---: | --- |
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3993201344 | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689872288 | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |

`llama-gguf-split` was not present initially, so only that target was built
from the unchanged llama.cpp source tree. The merged GGUF was produced with:

```text
llama-gguf-split --merge <first-split-gguf> <merged-gguf>
```

Merged model:

```text
merged_file: qwen2.5-7b-instruct-q4_k_m.gguf
merged_size_bytes: 4683073536
merged_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
```

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

These are binary and API-surface inspection results only. They do not prove
that the model followed the requested BootTrace output contract.

## Quality Calibration Outcome

The public synthetic BootTrace suite contains five required cases:

```text
boot_success
boot_retry_recovered
boot_config_missing
boot_device_timeout
boot_firmware_component_failure
```

Strict calibration produced 5/5 completed measured requests and 5/5 raw-valid
JSON outputs. However, none of the five required cases passed the deterministic
validators:

```text
failure_category_histogram:
  missing_required_evidence: 2
  model_capability_failure: 5
  response_format_violation: 3
  wrong_root_cause: 5
  wrong_status: 1
```

Per-case failure summary:

| case | parser status | selected failure categories |
| --- | --- | --- |
| `boot_success` | raw_json_valid | `model_capability_failure`, `response_format_violation`, `wrong_root_cause` |
| `boot_retry_recovered` | raw_json_valid | `model_capability_failure`, `wrong_root_cause`, `wrong_status` |
| `boot_config_missing` | raw_json_valid | `model_capability_failure`, `response_format_violation`, `wrong_root_cause` |
| `boot_device_timeout` | raw_json_valid | `missing_required_evidence`, `model_capability_failure`, `response_format_violation`, `wrong_root_cause` |
| `boot_firmware_component_failure` | raw_json_valid | `missing_required_evidence`, `model_capability_failure`, `wrong_root_cause` |

This was not a Markdown-fence-only failure. The existing single-fence
normalization path was therefore not run. Semantic validator failures such as
wrong status, wrong root cause, non-numeric fields, and missing required
evidence IDs must not be normalized away.

ArmTune therefore reported:

```text
quality_gate_status: failed
model_profile_suitability: unsuitable
required_quality_cases_passed: 0/5
```

## Artifact Integrity

The nested benchmark artifacts and the outer Phase 6B quality-calibration
artifact set both passed digest validation:

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

## Phase 6C Boundary

Phase 6C was intentionally not executed:

```text
portable-baseline build: not_run
arm-native build: not_run
arm-optimized build: not_run
build-profile quality calibration: not_run
paired profile comparison: not_run
eligible recommendation: not_run
Performance Passport generation: not_run
```

This follows the Phase 6 rule that build-profile comparison is conditional on
first finding a model/profile that is suitable under the deterministic quality
protocol.

## Claim Boundary

This evidence supports only the following claim:

> ArmTune's Phase 6B real SSH model-qualification path, Linux Arm64
> `llama-server` worker execution, Qwen2.5-7B split GGUF acquisition and
> merge, strict BootTrace quality rejection, cleanup, and artifact digest
> validation were exercised on a real AWS C7g / Graviton3 target.

This evidence does not show:

- a suitable Qwen2.5-7B BootTrace model/profile
- a quality-preserving build-profile recommendation
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

The next model-qualification candidate should be stronger than Qwen2.5-7B for
the synthetic BootTrace contract, or use a model/runtime combination with
stronger enforceable structured-output behavior. Validators and canonical
expected answers should remain unchanged unless the benchmark methodology
itself is intentionally revised in a separate phase.
