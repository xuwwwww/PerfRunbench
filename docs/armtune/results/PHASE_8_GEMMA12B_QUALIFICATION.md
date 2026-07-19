# Phase 8 Gemma 3 12B Held-Out Qualification

## Scope

This document records a bounded Phase 8 real-model qualification run for
Gemma 3 12B QAT Q4_0 on the user-provisioned C7g / Graviton3 Linux Arm64
target.

The strict qualification failed before semantic validation because every
output used a Markdown JSON fence while the frozen contract required raw JSON.
A later offline diagnostic is recorded separately in
`docs/armtune/results/PHASE_8_GEMMA12B_NORMALIZATION_DIAGNOSTIC.md`.

No benchmark infrastructure was added. No held-out validators, canonical
answers, prompt contracts, normalization policy, llama.cpp build, seed, or
temperature settings were changed.

## Access And Download

The Gemma gated-file access blocker from
`docs/armtune/results/PHASE_8_MODEL_QUALIFICATION_PREFLIGHT.md` was resolved
outside the repository. No credential value was printed, stored, or committed.

Windows controller dry run:

```text
hf download google/gemma-3-12b-it-qat-q4_0-gguf gemma-3-12b-it-q4_0.gguf --dry-run
```

Remote target dry run:

```text
hf download google/gemma-3-12b-it-qat-q4_0-gguf gemma-3-12b-it-q4_0.gguf --dry-run
```

Both dry runs reported one downloadable file:

```text
gemma-3-12b-it-q4_0.gguf
size: 8.1G
```

Downloaded model:

```text
repository: google/gemma-3-12b-it-qat-q4_0-gguf
filename: gemma-3-12b-it-q4_0.gguf
size_bytes: 8074473920
sha256: dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd
```

## Target And Code State

Target class:

```text
instance_class: c7g.2xlarge
architecture: aarch64
cpu_model: Neoverse-V1
logical_cpus: 8
kernel: 6.17.0-1019-aws
```

Repository state:

```text
remote_perfrunbench_commit: 5f021962ba9c532f3499dae205635e114581df23
```

llama.cpp evidence:

```text
llama_server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
runtime_build: version: 10050 (b15ca938a) built with GNU 13.3.0 for Linux aarch64
```

## Frozen Qualification Contract

Phase 8-R reused the Phase 7 frozen contract:

```text
prompt_contract_version: taxonomy-rules-and-disjoint-few-shot
held_out_suite: configs/armtune/quality/boottrace_suite.json
held_out_suite_id: synthetic_boottrace_quality_v1
held_out_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
development_suite_digest: 86e4bfc54e36de6a5e05257f621b9404104ce7c42bf9e90fce0019a7a9ea10db
normalization_policy_version: boottrace-normalization-v1
normalization: strict disabled
allow_markdown_fence: false
temperature: 0.0
seed: 42
threads: 8
context_size: 4096
batch_size: 256
ubatch_size: 128
predict_tokens: 192
```

Command shape:

```text
autotuneai arm quality-calibrate --config <private-gemma12b-heldout-config>
```

## Qualification Result

Run:

```text
run_id: 20260719_183725_010839
created_at_utc: 2026-07-19T10:42:14Z
operation: quality_calibrate.llama_server
target_type: ssh
result_origin: measured
```

Outcome:

```text
suite_case_count: 5
request_success_count: 5
raw_json_valid_count: 0
normalized_json_valid_count: 0
required_quality_cases_passed: 0/5
critical_failure_count: 45
quality_gate_status: failed
model_profile_suitability: unsuitable
```

Failure category histogram:

```text
invalid_json: 5
markdown_fence_present: 5
model_capability_failure: 5
response_format_violation: 5
wrong_root_cause: 5
wrong_status: 5
```

Interpretation:

- All five requests completed on the real Linux Arm64 target.
- The model emitted Markdown-fenced JSON for every held-out case.
- Strict raw-output mode intentionally rejected fenced JSON.
- No single-fence normalization was enabled for this held-out run because the
  frozen Phase 7 qualification contract required strict disabled
  normalization.
- The validators and canonical answers were not changed to make Gemma pass.
- The strict result is a raw-output protocol rejection, not an SSH, cleanup,
  digest, artifact, or request-execution failure.

## Post-Hoc Normalization Diagnostic

An offline diagnostic was run after the strict result using only the existing
`boottrace-normalization-v1` single-fence policy:

```text
diagnostic_label: post_hoc_normalization_diagnostic
sidecar: docs/armtune/results/phase8_gemma12b_normalization_diagnostic.json
normalized_json_valid_count: 5/5
required_cases_passed: 1/5
critical_failure_count: 6
invented_evidence_failures: 0
quality_gate_status: failed
```

That diagnostic did not rerun inference, start AWS, change canonical answers,
modify validators, tune prompts, repair malformed JSON, or alter model outputs.
It preserved the original raw outputs and SHA-256 values in the JSON sidecar.

The diagnostic shows that Gemma 3 12B is not the same failure shape as the
Qwen2.5 runs. Qwen2.5-7B and Qwen2.5-14B produced raw-valid JSON with semantic
failures. Gemma first failed the strict raw-JSON protocol, then still passed
only 1/5 required cases after exact single-fence extraction.

## Artifact And Cleanup Validation

Built-in result validation:

```text
quality_calibration_artifact_validation: passed
quality_calibration_checked_artifacts: 42
nested_benchmark_artifact_validation: passed
nested_benchmark_checked_artifacts: 32
benchmark_cleanup_status: passed
model_digest_validation: passed
executable_digest_validation: passed
```

Independent post-download artifact revalidation:

```text
.autotuneai/runs/20260719_183725_010839/armtune/quality_calibration:
  status=passed checked=42 issues=0
.autotuneai/runs/20260719_183725_010839/armtune/quality_calibration/benchmark:
  status=passed checked=32 issues=0
```

Post-run cleanup checks:

```text
llama_server_processes: none
loopback_port_37493_listeners: none
controller_process: exited
remote_memory_available_after_run: approximately 14 GiB
remote_root_free_after_run: approximately 8.4 GiB
```

## Decision

The original Gemma 3 12B QAT Q4_0 strict qualification remains failed and
`unsuitable` under the unchanged Phase 7 frozen held-out BootTrace contract.
The post-hoc diagnostic does not override that result. It shows that Gemma's
preserved outputs are valid JSON after exact single-fence extraction, but still
fail 4/5 required cases under unchanged validators.

Because normalization did not produce a 5/5 semantic pass, a fresh held-out v2
Gemma evaluation is not justified from this diagnostic. ArmTune should stop the
current model ladder here. It should not proceed to build-profile comparison,
runtime tuning, Performance Passport generation, KleidiAI claims, Performix
evidence, or final performance claims without first defining a new
deterministic workload strategy or qualifying a model/profile under a
separately justified protocol.

## Claim Boundary

This result supports:

> ArmTune executed a measured Gemma 3 12B QAT Q4_0 held-out BootTrace
> qualification on a real Linux Arm64 C7g target, preserved artifacts,
> revalidated manifests after download, cleaned up `llama-server`, and
> correctly rejected the strict run when the frozen raw-JSON protocol failed.

This result does not show:

- Gemma 3 12B suitability
- quality-preserving performance improvement
- build-profile comparison
- runtime tuning benefit
- Performance Passport eligibility
- KleidiAI execution
- Performix evidence
- final competition benchmark results
