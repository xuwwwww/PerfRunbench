# Phase 8 Model Qualification Preflight

## Scope

This document records the Phase 8 preflight after Phase 7 real contract
selection. It uses only existing Qwen2.5-7B and Qwen2.5-14B held-out
qualification artifacts and an access check for the official Gemma 3 12B GGUF
repository.

No benchmark infrastructure was added. No canonical held-out answers,
validators, prompt contract, normalization policy, or generated model outputs
were modified. AWS was not started for this preflight.

## Phase 7 Carry-Forward

```text
frozen_contract: taxonomy-rules-and-disjoint-few-shot
held_out_suite: configs/armtune/quality/boottrace_suite.json
held_out_suite_id: synthetic_boottrace_quality_v1
held_out_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
development_suite: configs/armtune/quality/boottrace_calibration_suite.json
development_suite_digest: 86e4bfc54e36de6a5e05257f621b9404104ce7c42bf9e90fce0019a7a9ea10db
expected_response_schema_sha256: 32aac96fce93b1ed4f73c3c08483943c258c0828c57e36afe0e10ad4e7668c1e
normalization: strict, disabled
temperature: 0.0
seed: 42
predict_tokens: 192
```

Relevant prior evidence:

```text
docs/armtune/results/PHASE_7_REAL_CONTRACT_SELECTION_AND_MODEL_LADDER.md
```

## Cross-Model Held-Out Failure Matrix

Artifacts compared:

```text
Qwen2.5-7B held-out run: 20260719_162659_750315
Qwen2.5-14B held-out run: 20260719_164004_988884
```

| Case | Expected | Qwen2.5-7B Output | Qwen2.5-14B Output | Qwen2.5-7B Failures | Qwen2.5-14B Failures |
| --- | --- | --- | --- | --- | --- |
| `boot_success` | `status=success`; latency `0..120`; evidence `L001,L002,L003`; summary contains `Boot completed` | passed; `status=success`; latency `102`; evidence `L001,L002,L003`; summary says boot completed successfully after kernel handoff | passed; `status=success`; latency `56`; evidence `L001,L002,L003`; summary says boot completed successfully after kernel handoff | none | none |
| `boot_retry_recovered` | `status=warning`; latency `50..180`; evidence `L010,L011,L012,L013`; summary contains `one retry` | passed; `status=warning`; latency `120`; evidence `L010,L011,L012,L013`; summary says boot completed after one retry | passed; `status=warning`; latency `150`; evidence `L010,L011,L012,L013`; summary says boot completed after one retry | none | none |
| `boot_config_missing` | `status=failure`; latency `0..80`; evidence `L020,L021`; summary contains `required configuration line is missing` | failed; `status=failure`; latency `123`; evidence `L020,L021`; summary says required storage UUID is absent | failed; `status=failure`; latency `56`; evidence `L020,L021`; summary says required storage UUID is absent | `model_capability_failure`, `wrong_root_cause`; numeric range failed; required summary substring missing | `model_capability_failure`, `wrong_root_cause`; required summary substring missing |
| `boot_device_timeout` | `status=failure`; latency `200..5000`; evidence `L030,L031,L032`; summary contains `NVMe controller timed out` | failed; `status=failure`; latency `150`; evidence `L030,L031,L032`; summary says NVMe controller did not become ready before timeout | failed; `status=failure`; latency `150`; evidence `L030,L031,L032`; summary says NVMe controller did not become ready before timeout | `model_capability_failure`, `wrong_root_cause`; numeric range failed; required summary substring missing; regex did not match | `model_capability_failure`, `wrong_root_cause`; numeric range failed; required summary substring missing; regex did not match |
| `boot_firmware_component_failure` | `status=failure`; latency `100..3000`; evidence `L040,L041,L042,L043`; summary contains `firmware component checksum failure` | failed; `status=failure`; latency `120`; evidence `L040,L041,L042,L043`; summary says firmware component checksum failure blocked boot pending service action | failed; `status=failure`; latency `124`; evidence `L040,L041,L042,L043`; summary says firmware component checksum failure blocked boot pending service action | `model_capability_failure`, `wrong_root_cause`; regex did not match | `model_capability_failure`, `wrong_root_cause`; regex did not match |

## Matrix Interpretation

The two models pass and fail the same held-out cases:

```text
passed_by_both: boot_success, boot_retry_recovered
failed_by_both: boot_config_missing, boot_device_timeout, boot_firmware_component_failure
```

The failure categories overlap on every failed case:

```text
shared_failed_categories: model_capability_failure, wrong_root_cause
invented_evidence: 0 for both models
request_failures: 0 for both models
raw_json_valid: 5/5 for both models
cleanup: passed for both models
artifact_validation: passed for both models
```

The observed pattern does not indicate an SSH, parser, cleanup, artifact, or
JSON-generation infrastructure failure. It also does not show that increasing
from 7B to 14B solves the held-out semantic gate. Both models generally copied
the required evidence IDs and selected the expected status enum values, but
they still missed strict summary wording and latency constraints on the same
three failure cases.

This matrix is diagnostic evidence only. It is not a reason to weaken
validators, rewrite canonical answers, or use held-out cases as prompt tuning
examples.

## Gemma 3 12B Access Preflight

The next planned model was:

```text
repository: google/gemma-3-12b-it-qat-q4_0-gguf
candidate_file: gemma-3-12b-it-q4_0.gguf
revision_from_prior_model_info: 7929e34ac7c459bbfd1f38d7358e6839f204578b
```

Access checks run from the controller environment:

```powershell
HF_TOKEN: not set
HUGGINGFACE_HUB_TOKEN: not set
curl https://huggingface.co/api/models/google/gemma-3-12b-it-qat-q4_0-gguf
```

Result:

```text
model_api_status: 200
revision_file_resolve_status: 401
main_file_resolve_status: 401
```

The official repository metadata is visible, but the GGUF file is not
downloadable from the current environment. No AWS instance was started, no
Gemma model file was downloaded, and no Gemma qualification run was attempted.

This is an access blocker, not a model quality result.

## Phase 8 Gate

Before any further AWS run:

```text
required_precondition: official Gemma GGUF file access confirmed
contract: taxonomy-rules-and-disjoint-few-shot
held_out_suite: unchanged
canonical_answers: unchanged
validators: unchanged
normalization: strict disabled
temperature: 0.0
seed: 42
llama.cpp commit: unchanged
aws_lifecycle: user-controlled
```

If Gemma access is later granted, the next run should be one bounded held-out
qualification using the frozen Phase 7 contract. If Gemma is unsuitable, stop
and record evidence. If Gemma is suitable, continue into build profiles,
quality calibration for every profile, runtime tuning, quality-aware
comparison, Performance Passport generation, and Passport verification.

## Follow-Up Real Qualification

Gemma gated-file access was later granted outside the repository. The follow-up
real held-out qualification is recorded in:

```text
docs/armtune/results/PHASE_8_GEMMA12B_QUALIFICATION.md
```

The follow-up run downloaded and hashed
`gemma-3-12b-it-q4_0.gguf`, executed one measured held-out qualification on the
same frozen `taxonomy-rules-and-disjoint-few-shot` contract, and rejected the
model/profile:

```text
run_id: 20260719_183725_010839
result_origin: measured
request_success_count: 5/5
raw_json_valid_count: 0/5
required_quality_cases_passed: 0/5
critical_failure_count: 45
quality_gate_status: failed
model_profile_suitability: unsuitable
artifact_validation: passed
cleanup: passed
```

Gemma emitted Markdown-fenced JSON for every held-out case. Strict
normalization remained disabled by the frozen Phase 7 qualification contract,
so the fenced outputs were rejected rather than normalized into a pass.

## Claim Boundary

This preflight supports:

> ArmTune's Qwen2.5 held-out failures are cross-model semantic gate failures
> under a frozen prompt contract, and Gemma 3 12B qualification is blocked by
> file access until official GGUF access is available.

This preflight does not show:

- Gemma 3 12B quality
- a suitable model/profile
- build-profile comparison
- runtime tuning evidence
- Performance Passport eligibility
- KleidiAI execution
- Performix evidence
- final competition benchmark results
