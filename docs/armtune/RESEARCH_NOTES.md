# ArmTune Research Notes

Access date for external references: 2026-07-13 unless otherwise noted.

## Arm Create: AI Optimization Challenge 2026

- Organization: Arm / Devpost
- URL: https://arm-ai-optimization-challenge.devpost.com/
- URL: https://arm-ai-optimization-challenge.devpost.com/rules
- Relevant conclusion: the challenge targets AI optimization on Arm-powered
  platforms, including cloud AI workflows on Arm64.
- Implementation impact: ArmTune should emphasize measured Arm64 inference
  performance, reproducibility, public setup, conservative claims, and a short
  demo.

## Arm Performix

- Organization: Arm
- URL: https://developer.arm.com/servers-and-cloud-computing/arm-performix
- Relevant conclusion: Arm Performix is a performance analysis toolkit for
  Arm server and cloud developers.
- Implementation impact: Performix remains future optional evidence. Current
  v0.1.0 results do not claim Performix usage.

## llama.cpp

- Organization: ggml-org
- URL: https://github.com/ggml-org/llama.cpp
- Relevant conclusion: llama.cpp provides local/cloud LLM inference with
  binaries including `llama-bench` and `llama-server`.
- Implementation impact: ArmTune inspects actual binaries, help output,
  version output, file digests, build evidence, and runtime behavior instead
  of inferring feature usage from host capabilities.

## llama-bench

- Organization: ggml-org
- URL: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- Relevant conclusion: `llama-bench` supports machine-readable benchmark
  output formats and reports throughput data useful for repeated comparison.
- Implementation impact: ArmTune preserves raw output, parser status, failed
  trials, and distribution summaries.

## KleidiAI

- Organization: Arm
- URL: https://github.com/ARM-software/kleidiai
- Relevant conclusion: KleidiAI provides optimized AI micro-kernels for Arm
  CPUs.
- Implementation impact: ArmTune may report KleidiAI only when build/runtime
  evidence supports the claim. Requested flags alone are not sufficient.

## Native Arm64 CI Validation

- Source: GitHub Actions
- Workflow: `ArmTune Native Arm64 Smoke`
- Run URL: https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566
- Run ID: `29270842566`
- Tested commit: `2074dfac80e699ba22178774b28c66e79afae475`
- Runner OS/architecture: Linux `aarch64`
- Kernel: `6.17.0-1018-azure`
- Uploaded artifact name: `armtune-arm64-smoke-artifacts`
- Artifact run ID: `20260713_173254_650848`
- Result origin: `test_fixture`
- Artifact digest validation: passed; 13 artifacts checked.
- Focused ArmTune suite at the tested commit: 94 tests.
- Claim boundary: fixture values such as `321.5 tokens/s` and
  `87.25 tokens/s` are fake workload constants, not Arm performance results.

## Real Arm Baseline And Runtime Milestones

### Phase 2C: Real SSH Arm64 Baseline

- Evidence file: `docs/armtune/results/PHASE_2C_AWS_GRAVITON2.md`
- Run ID: `20260716_231915_147773`
- Target: user-provisioned AWS T4g / Linux `aarch64` / Neoverse-N1
- PerfRunbench commit: `41b57b70b318a94977a5dd822314782e855295f6`
- llama.cpp commit: `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- Result origin: `measured`
- Parsed preliminary throughput: `43.8 tokens/s` prefill,
  `30.13 tokens/s` decode.
- Claim boundary: this validated the real SSH Arm64 execution pipeline, not
  optimization or stable production performance.

### Phase 3A: Repeated Search

- Evidence file: `docs/armtune/results/PHASE_3A_REAL_SEARCH_VALIDATION.md`
- Run ID: `20260718_001757_718789`
- Result: repeated search completed on real Linux Arm64 with artifact
  validation passed.
- Recommendation: `inconclusive`; the selected delta `0.116029%` was inside
  the configured 5% noise band.
- Claim boundary: this proved search and statistics plumbing, not a measured
  improvement.

### Phase 3A.2: Search Guardrails

- Run ID: `20260718_154049_739709`
- Validated `max_candidates` and `affinity_modes` CLI overrides through the
  Windows controller, SSH protocol, remote worker, effective config artifact,
  and downloaded artifact validation.
- Claim boundary: functional validation only, not a new performance claim.

### Phase 3B.1: Server Runtime

- Evidence file: `docs/armtune/results/PHASE_3B1_REAL_SERVER_VALIDATION.md`
- Run ID: `20260719_004451_487119`
- Validated loopback-only `llama-server`, target-local HTTP, streaming parse,
  TTFT, end-to-end latency, generated-output preservation, cleanup, and
  artifact validation.
- Claim boundary: bounded server-pipeline validation, not final optimization.

## Phase 9 First Real Arm Build And Runtime Tuning

- Evidence file: `docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`
- Controller and remote PerfRunbench commit:
  `f17297111a66716d0f65c2a4336ebf8f4ea817a6`
- Target: user-provisioned C7g / Graviton3 Linux Arm64 / Neoverse-V1 /
  8 logical CPUs.
- Model: Qwen2.5-7B Q4_K_M,
  SHA-256 `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49`.
- llama.cpp source commit:
  `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`.
- Build profiles: `portable-baseline`, `arm-native`, `arm-optimized`.
- Fixed `llama-bench` build-profile comparison:
  - `portable-baseline` decode median: `10.825 tok/s`
  - `arm-native` decode median: `17.530 tok/s`
  - `arm-optimized` decode median: `17.380 tok/s`
- Bounded runtime search:
  - baseline decode median: `13.630 tok/s`
  - selected decode median: `17.480 tok/s`
  - observed improvement: `28.246515%`
- Secondary `llama-server` timing:
  - `portable-baseline` TTFT median: `97.260 ms`
  - `arm-native` TTFT median: `62.826 ms`
- Performance Passport:
  `claim_eligibility_state: eligible_preliminary`,
  `verification_status: passed`.
- Independent artifact-manifest revalidation: 82 manifests checked, 0 failed.
- Cleanup: no residual `llama-server`, no residual ArmTune worker, and no
  observed leftover benchmark listen port.
- Claim boundary: protocol-specific results must not be combined into one
  aggregate improvement.

## Phase 10 Guarded Recommendation Apply

- Evidence file: `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`
- RecommendationBundle ID: `92b0eb69b189a06baae62f15`
- Result:
  - `environment_validation: pass`
  - `dry_run_status: dry_run_passed`
  - `canary_status: passed`
  - `lease_state_after_apply: active`
  - `rollback_state: rolled_back`
  - artifact revalidation passed before and after download
  - cleanup passed
- Claim boundary: guarded apply validation is apply-safety evidence, not a new
  benchmark result.

## Phase 12 External Workload Boundary

Firmware-log model capability evaluation was extracted from PerfRunbench after
the `v0.1.0-armtune` release. The standalone BootTraceBench project owns that
workload's cases, prompt contracts, semantic validators, fixtures, and
historical sanitized experiment notes.

PerfRunbench current main keeps workload-neutral correctness hooks and may
later execute external semantic tools through generic JSON or command-workload
interfaces. It does not import BootTraceBench and BootTraceBench does not
import PerfRunbench.

## Phase 20: C7g Build Backend And Runtime Tuning

- Evidence file:
  `docs/results/PHASE_20_C7G_BUILD_BACKEND_RUNTIME_TUNING.md`
- PerfRunbench validation source:
  `b8bcdd6c33bd1d9b3a5a761f3061df2e02316e73`
- Target: user-provisioned C7g / Graviton3 Linux Arm64 / Neoverse-V1 /
  8 logical CPUs.
- Model: Qwen2.5-7B Q4_K_M,
  SHA-256 `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49`.
- llama.cpp source commit:
  `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.
- Build-profile comparison:
  - `upstream-default` decode median: `17.71 tok/s`
  - `explicit-neoverse-v1` decode median: `17.78 tok/s`
  - observed delta: `0.395257%`
  - recommendation: `inconclusive`
- Runtime search:
  - six unique candidates
  - selected candidate: `baseline`
  - complete confirmation decode median: `17.72 tok/s`
  - recommendation: `inconclusive`
- Secondary server timing:
  - measured requests: `3/3`
  - TTFT median: `61.470 ms`
  - E2E latency median: `911.976 ms`
  - quality gate: `not_evaluated`
- Performance Passport:
  - `claim_eligibility_state: ineligible_provenance`
  - verification passed
- Artifact revalidation:
  - `77/77` manifests passed remotely
  - `77/77` manifests passed after download

Claim boundary: Phase 20 validated the real build/backend/runtime evidence
pipeline. It did not verify a performance improvement, global Arm optimality,
or KleidiAI execution.
