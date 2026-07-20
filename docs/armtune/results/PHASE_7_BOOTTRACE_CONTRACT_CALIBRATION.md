# Phase 7 BootTrace Contract Calibration Foundation

## Scope

This document records the Phase 7 local foundation work for BootTrace contract
calibration. It does not record a new real-model qualification result.

Phase 7 starts from the Phase 6B observation that Qwen2.5-7B produced
5/5 raw-valid JSON outputs while still failing all five held-out BootTrace
cases. That failure pattern is not sufficient evidence to immediately move to
the next model. The prompt contract, taxonomy clarity, chat-template
integration, and model capability must be separated.

## Added Public Inputs

```text
held_out_evaluation_suite: configs/armtune/quality/boottrace_suite.json
development_calibration_suite: configs/armtune/quality/boottrace_calibration_suite.json
held_out_case_audit: configs/armtune/quality/boottrace_case_audit.json
```

The development calibration suite is disjoint from the held-out evaluation
suite by case ID and evidence line IDs. It may provide examples for prompt
contract calibration, but it must not be counted as final qualification
evidence.

## Held-Out Audit

The audit records, for every held-out case:

```text
trace_sufficiency
unique_expected_classification
expected_status
root_cause_category
failure_stage
required_evidence_line_ids
distractors
unknown_policy_correctness
ambiguity
canonical_answer_change_required
human_rationale
```

No canonical held-out answers were changed in this foundation step. The
`boot_config_missing` case records a low ambiguity between the absent storage
UUID trace and the legacy canonical summary phrase, but the answer remains
unchanged. Phase 7 handles that through explicit taxonomy and decision rules,
not by weakening the held-out validators.

## Prompt Contract Versions

`quality_gate.prompt_contract_version` accepts:

```text
current-zero-shot
taxonomy-and-decision-rules
taxonomy-rules-and-disjoint-few-shot
```

`current-zero-shot` preserves the existing prompt text. The taxonomy variant
adds explicit decision order, unknown policy, and root-cause taxonomy while
keeping the existing output schema. The few-shot variant adds examples from the
separate development calibration suite and does not reuse any of the five
held-out evaluation cases as examples.

Artifacts now record:

```text
prompt_contract_version
taxonomy_version
rendered prompt text
rendered prompt SHA-256
expected response schema SHA-256
generation settings
normalization policy
calibration example digests, when applicable
```

## SSH Contract Application

For SSH `quality-calibrate`, the controller now renders
`prompt_suite_snapshot.json`, uploads that exact file into the versioned remote
run directory, and updates the fixed worker job to reference the uploaded
prompt suite. This keeps the worker command allowlisted while ensuring request
artifacts prove which prompt contract was actually sent to `llama-server`.

The controller does not upload model weights, credentials, private logs,
arbitrary workload files, or remote shell strings as part of this path.

## Validation

Local fixture validation covered:

```text
held-out audit coverage
calibration suite disjointness
prompt contract digest stability
distinct rendered prompts for all three contract versions
few-shot example digest recording
quality-calibrate artifact recording for the selected contract version
request JSON preservation of the rendered prompt
SSH prompt-suite upload command shape
```

Focused tests passed:

```text
python -m unittest discover -s tests -p "test_arm_tune_quality*.py"
Ran 18 tests
OK

python -m unittest discover -s tests -p "test_arm_tune_targets.py"
Ran 8 tests
OK
```

## Claim Boundary

This foundation supports only the following claim:

> ArmTune now separates held-out BootTrace evaluation from public development
> calibration examples, versions prompt contracts, records prompt-contract
> artifacts, and applies controller-rendered prompt suites to SSH
> quality-calibration workers.

This evidence does not show:

- Qwen2.5-7B passed any new Phase 7 contract
- a stronger model passed qualification
- profile comparison
- Performance Passport eligibility
- improved Arm performance
- KleidiAI execution
- final competition results

## Real-Target Follow-Up

The real-target follow-up is recorded in:

```text
docs/armtune/results/PHASE_7_REAL_CONTRACT_SELECTION_AND_MODEL_LADDER.md
```

That run selected `taxonomy-rules-and-disjoint-few-shot` using only the
development calibration suite, froze the selected contract before held-out
evaluation, and then rejected Qwen2.5-7B and Qwen2.5-14B on the held-out suite.
Profile comparison remains blocked because no tested model/profile reached
`model_profile_suitability: suitable`.
