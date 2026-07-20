# Phase 8 Gemma 3 12B Post-Hoc Normalization Diagnostic

## Scope

This document records an offline post-hoc diagnostic for the preserved Gemma 3
12B Phase 8 artifacts.

No AWS instance was started. No inference was rerun. No held-out canonical
answers, validators, prompt contract, normalization implementation, fixtures,
model output values, evidence IDs, or expected results were changed.

The original strict qualification remains failed. This diagnostic is labeled:

```text
post_hoc_normalization_diagnostic
```

It must not be used to claim held-out qualification success.

## Source Artifacts

Source run:

```text
run_id: 20260719_183725_010839
source_run_directory: .autotuneai/runs/20260719_183725_010839/armtune/quality_calibration
strict_result_path: .autotuneai/runs/20260719_183725_010839/armtune/quality_calibration/quality_calibration_result.json
benchmark_result_origin: measured
```

Held-out suite:

```text
suite_id: synthetic_boottrace_quality_v1
suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
case_count: 5
required_case_count: 5
```

Strict protocol result preserved from the original qualification:

```text
quality_gate_status: failed
model_profile_suitability: unsuitable
raw_json_valid_count: 0
normalized_json_valid_count: 0
required_quality_cases_passed: 0
critical_failure_count: 45
preserved_as_failed: true
```

## Diagnostic Method

The diagnostic used only the existing versioned single-fence policy:

```text
normalization_policy_version: boottrace-normalization-v1
enabled: true
strict: false
allow_markdown_fence: true
allow_prose_outside_fence: false
```

For each preserved `output.txt`, the diagnostic:

1. Preserved the original raw output and SHA-256 in the JSON sidecar.
2. Required exactly one complete Markdown JSON fence.
3. Rejected prose outside the fence, multiple fences, or malformed JSON.
4. Extracted the fenced JSON without changing semantic values.
5. Ran the unchanged held-out validators.
6. Recorded normalized SHA-256, normalization steps, validator results, and
   failure categories.

The complete reproducibility sidecar is:

```text
docs/armtune/results/phase8_gemma12b_normalization_diagnostic.json
```

## Summary

```text
normalized_json_valid_count: 5/5
required_cases_passed: 1/5
critical_failure_count: 6
invented_evidence_failures: 0
quality_gate_status: failed
```

Failure category histogram:

```text
markdown_fence_present: 5
response_format_violation: 5
model_capability_failure: 4
wrong_root_cause: 3
```

All five outputs were valid JSON after exact single-fence extraction, but only
`boot_retry_recovered` passed the unchanged semantic validators.

## Per-Case Results

| Case | Parser | Diagnostic Status | Raw SHA-256 | Normalized SHA-256 | Critical Failures |
| --- | --- | --- | --- | --- | --- |
| `boot_success` | `normalized_json_valid` | `failed` | `c4ffbc4d57dd928629dcccef6bfdf8937374cd7673501b63a413db5aa06c0b3c` | `d7a647ee00040f9401cc948bd891ad43d4346129eb0de9c122a348314f7b2ff2` | `latency_ms` outside expected range |
| `boot_retry_recovered` | `normalized_json_valid` | `passed` | `542c865bf0a44baec36a0857b4f0c9c00a94191a15911e6dfbbab7e8c27e8826` | `a02239071328d52cabe198710e23415f4cbd08001e497193c6c8014cd79523ec` | none |
| `boot_config_missing` | `normalized_json_valid` | `failed` | `208c5fc5c11e42720687c83fbf7e1dd5b997486c9918f6ef3f662e65fb0181d9` | `1991b579359a50776e4d34eb8c3b20ca65ac8fec224fbbfa3ec2277013c50611` | missing required summary substring: `required configuration line is missing` |
| `boot_device_timeout` | `normalized_json_valid` | `failed` | `919c7a73d80bc9e093c44941f2c497e88b0a9eab5778d45817690b017035aad9` | `4f835df9ae250ead08369f3ced7800207f45e0acaf248195c07156731ef44db0` | `latency_ms` outside expected range; missing `NVMe controller timed out`; regex mismatch |
| `boot_firmware_component_failure` | `normalized_json_valid` | `failed` | `f3d063c521522da5b94e129b18142c1eaca95574b46e21ba116fa7477a50e66c` | `8ccf6ca6afcd4141150f3534c5634c0c029ec7485b4340a4a3d87a2c0706e298` | regex mismatch |

Every case applied exactly:

```text
extract_single_markdown_fenced_json_block
```

No case required or allowed semantic repair.

## Interpretation

Gemma 3 12B is not in the same failure bucket as the Qwen2.5 runs:

- Qwen2.5-7B and Qwen2.5-14B produced raw-valid JSON and failed semantic
  validators.
- Gemma 3 12B first failed the strict raw-JSON protocol because every output
  used a Markdown fence.
- The offline diagnostic shows Gemma's preserved fenced outputs become valid
  JSON under `boottrace-normalization-v1`, but still fail 4/5 required cases.

Therefore Gemma appears more than merely format-limited for the current
five-case held-out BootTrace protocol. It remains unsuitable after post-hoc
single-fence normalization, but this diagnostic does not replace the original
strict failed qualification.

## Decision

A fresh held-out v2 evaluation is not justified from this diagnostic because
normalization did not produce a 5/5 semantic pass.

ArmTune should stop the current model ladder and treat the existing five
held-out BootTrace cases as a capability benchmark. The next work should define
a separate deterministic CPU-inference tuning workload with stable, uniquely
checkable outputs before build-profile comparison, runtime tuning, or
Performance Passport generation.

## Claim Boundary

This diagnostic supports:

> ArmTune's preserved Gemma 3 12B Phase 8 outputs are valid JSON after exact
> single-fence extraction, but unchanged held-out validators still pass only
> 1/5 required cases.

This diagnostic does not show:

- strict held-out qualification success
- a suitable Gemma 3 12B model/profile
- quality-preserving performance improvement
- build-profile comparison
- runtime tuning benefit
- Performance Passport eligibility
- KleidiAI execution
- Performix evidence
- final competition benchmark results
