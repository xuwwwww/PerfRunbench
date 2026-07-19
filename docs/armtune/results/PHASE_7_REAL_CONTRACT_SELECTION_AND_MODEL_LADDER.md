# Phase 7 Real Contract Selection And Model Ladder Evidence

## Scope

This document records the Phase 7 real-target BootTrace prompt-contract
selection and conditional model-ladder run on a user-provisioned AWS C7g /
Graviton3 Linux Arm64 target.

The run did not change held-out validators, canonical answers, taxonomy
expected values, or model outputs. It also did not run build-profile
comparison or generate a measured Performance Passport because no tested
model/profile passed the held-out qualification gate.

## Target And Repository State

```text
target_class: user-provisioned AWS c7g.2xlarge
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
target_kernel: 6.17.0-1019-aws
cpu_model: Neoverse-V1
logical_cpus: 8
memory_available_before_run: approximately 14 GiB
root_disk_available_before_model_ladder_download: 16 GiB
PerfRunbench branch: feature/armtune
PerfRunbench commit: a2908b9bf57dde3cba81a1e846beccd9b8257d26
llama.cpp source checkout: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama-server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
```

The target was started manually after explicit user approval. ArmTune did not
provision, resize, or create cloud resources.

`quality_calibration_result.json` recorded `inspection.source_commit:
unknown` for these runs. The llama.cpp checkout commit above was verified by a
separate target-side git command and must not be confused with a parsed
inspection field.

## Fixed Protocol

```text
runtime_adapter: llama_server
server_host: 127.0.0.1
server_port: auto
threads: 8
context_size: 4096
batch_size: 256
ubatch_size: 128
temperature: 0.0
seed: 42
predict_tokens: 192
warmup_requests: 0
repetitions: 1
streaming: true
normalization_enabled: false
normalization_strict: true
```

The controller used the SSH target path, uploaded the rendered
`prompt_suite_snapshot.json`, invoked the fixed worker command, downloaded
versioned artifacts, and revalidated artifact manifests after download.

## Contract Selection Rule

The selection rule was fixed before any held-out run:

1. Higher required case pass count.
2. Lower critical validator failure count.
3. Lower invented evidence count.
4. Lower request/parse failure count.
5. If still tied, prefer the simpler contract.

Only the development calibration suite was used for prompt-contract selection:

```text
development_suite: configs/armtune/quality/boottrace_calibration_suite.json
development_suite_id: synthetic_boottrace_calibration_v1
development_suite_digest: 86e4bfc54e36de6a5e05257f621b9404104ce7c42bf9e90fce0019a7a9ea10db
held_out_suite: configs/armtune/quality/boottrace_suite.json
held_out_suite_id: synthetic_boottrace_quality_v1
held_out_suite_digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
expected_response_schema_sha256: 32aac96fce93b1ed4f73c3c08483943c258c0828c57e36afe0e10ad4e7668c1e
```

## Qwen2.5-7B Development Selection

Model:

```text
repository: Qwen/Qwen2.5-7B-Instruct-GGUF
revision: bb5d59e06d9551d752d08b292a50eb208b07ab1f
quantization: Q4_K_M
merged_model_sha256: 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49
```

| Candidate | Contract | Run ID | Requests | Raw JSON | Required | Critical | Invented Evidence | Request/Parse Failures | Artifact Validation | Suitability |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| A | `current-zero-shot` | `20260719_161817_491101` | 3/3 | 3/3 | 0/3 | 8 | 0 | 0 | passed | unsuitable |
| B | `taxonomy-and-decision-rules` | `20260719_162041_647289` | 3/3 | 3/3 | 0/3 | 3 | 0 | 0 | passed | unsuitable |
| C | `taxonomy-rules-and-disjoint-few-shot` | `20260719_162327_445710` | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | passed | suitable for development selection |

Selected contract:

```text
selected_before_held_out_evaluation: true
selected_contract: taxonomy-rules-and-disjoint-few-shot
development_prompt_contract_sha256: 12fcf5ddcf802f61d9cd10f588e90b05cb2957096317533bfbefc4371d876d4f
```

This selection is development-suite evidence only. It is not final model
qualification.

## Qwen2.5-7B Held-Out Qualification

The selected contract was then frozen and run once on the held-out suite:

```text
run_id: 20260719_162659_750315
prompt_contract_version: taxonomy-rules-and-disjoint-few-shot
held_out_prompt_contract_sha256: 0fae4ec9dfaa4e1ce2fddfc631590de6fc7dc1ff9da39b9f21df4e047b0ad7a3
requests_completed: 5/5
raw_json_valid: 5/5
required_cases_passed: 2/5
critical_validator_failures: 6
failure_categories: model_capability_failure, wrong_root_cause
artifact_validation: passed
nested_benchmark_artifact_validation: passed
cleanup_status: passed
result_origin: measured
suitability: unsuitable
```

Qwen2.5-7B remains unsuitable under the preselected Phase 7 prompt contract.

## Qwen2.5-14B Model-Ladder Check

Model acquisition:

```text
repository: Qwen/Qwen2.5-14B-Instruct-GGUF
revision: b466e1f8c07172155743e8e1307507d8a4f91fbd
quantization: Q4_K_M
downloaded_files:
  qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
    size_bytes: 3991999872
    sha256: a09ea5e7b1eafb1b30b241726c3cc3c905c96f14ad41e246ffa5f44e53904f68
  qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf
    size_bytes: 3989373504
    sha256: 21b9457d079680d284e90ef69607c4b2d8ef64a09d4729cb7b5e1357bdba41ae
  qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf
    size_bytes: 1006737120
    sha256: c8d37006760a387a35216e070e6664d7da927f10be8eb870fef2e3d4833d9976
root_disk_available_after_download: 7.6 GiB
```

The benchmark job used the first split GGUF path. The model manifest therefore
records the first split file SHA-256, while this evidence document records the
complete split-file digest set.

Development sanity:

```text
run_id: 20260719_163602_836249
suite: synthetic_boottrace_calibration_v1
prompt_contract_version: taxonomy-rules-and-disjoint-few-shot
requests_completed: 3/3
raw_json_valid: 3/3
required_cases_passed: 3/3
critical_validator_failures: 0
artifact_validation: passed
cleanup_status: passed
result_origin: measured
suitability: suitable for development sanity
```

Held-out qualification:

```text
run_id: 20260719_164004_988884
suite: synthetic_boottrace_quality_v1
prompt_contract_version: taxonomy-rules-and-disjoint-few-shot
held_out_prompt_contract_sha256: 0fae4ec9dfaa4e1ce2fddfc631590de6fc7dc1ff9da39b9f21df4e047b0ad7a3
requests_completed: 5/5
raw_json_valid: 5/5
required_cases_passed: 2/5
critical_validator_failures: 5
failure_categories: model_capability_failure, wrong_root_cause
artifact_validation: passed
nested_benchmark_artifact_validation: passed
cleanup_status: passed
result_origin: measured
suitability: unsuitable
```

Qwen2.5-14B is also unsuitable under the same frozen contract and held-out
suite. Profile comparison remains blocked.

## Gemma 3 12B Access Check

The next ladder candidate was:

```text
repository: google/gemma-3-12b-it-qat-q4_0-gguf
revision_visible_from_model_info: 7929e34ac7c459bbfd1f38d7358e6839f204578b
candidate_file: gemma-3-12b-it-q4_0.gguf
```

The model repository metadata was visible, but file metadata access for the
GGUF file returned `GatedRepoError 401`. No model file was downloaded and no
Gemma qualification run was attempted. This is an access limitation, not a
model quality result.

## Artifact Revalidation

All outer quality-calibration and nested benchmark manifests were independently
revalidated after download:

| Run | Outer Artifacts Checked | Nested Benchmark Artifacts Checked | Status |
| --- | ---: | ---: | --- |
| `qwen25-7b-dev-A` | 32 | 24 | passed |
| `qwen25-7b-dev-B` | 32 | 24 | passed |
| `qwen25-7b-dev-C` | 34 | 24 | passed |
| `qwen25-7b-heldout` | 42 | 32 | passed |
| `qwen25-14b-dev` | 34 | 24 | passed |
| `qwen25-14b-heldout` | 42 | 32 | passed |

Post-run cleanup checks found no residual `llama-server` or ArmTune worker
processes. The only listening services shown by `ss -ltnp` were local DNS and
SSH.

## Command Shapes Executed

```powershell
git push
git status -sb
git log -1 --oneline
autotuneai arm doctor --target-config <private-target-config> --require-production
autotuneai arm inspect-host --target-config <private-target-config>
autotuneai arm quality-calibrate --config <private-qwen25-7b-dev-A-config>
autotuneai arm quality-calibrate --config <private-qwen25-7b-dev-B-config>
autotuneai arm quality-calibrate --config <private-qwen25-7b-dev-C-config>
autotuneai arm quality-calibrate --config <private-qwen25-7b-heldout-config>
autotuneai arm quality-calibrate --config <private-qwen25-14b-dev-config>
autotuneai arm quality-calibrate --config <private-qwen25-14b-heldout-config>
```

Target-side checks:

```bash
uname -m
lscpu
free -h
df -h /
git -C /home/ubuntu/PerfRunbench fetch origin
git -C /home/ubuntu/PerfRunbench pull --ff-only
git -C /home/ubuntu/PerfRunbench rev-parse HEAD
git -C /home/ubuntu/llama.cpp rev-parse HEAD
sha256sum <llama-server> <model-file>
pgrep -af llama-server
pgrep -af autotune.cli
ss -ltnp
```

## Claim Boundary

This evidence supports:

> ArmTune selected a BootTrace prompt contract using only the development
> calibration suite, froze it before held-out evaluation, rejected
> Qwen2.5-7B and Qwen2.5-14B on a real Linux Arm64 C7g target when they failed
> held-out semantic quality, and preserved validated artifacts and cleanup
> evidence.

This evidence does not show:

- a suitable BootTrace model/profile
- quality-preserving Arm performance improvement
- build-profile comparison
- Performance Passport eligibility
- KleidiAI execution
- Performix evidence
- final competition benchmark results
- Gemma 3 12B model quality
