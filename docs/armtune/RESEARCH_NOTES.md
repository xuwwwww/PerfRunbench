# ArmTune Research Notes

Access date for all entries: 2026-07-13.

## Arm Create: AI Optimization Challenge 2026

- Organization: Arm / Devpost
- URL: https://arm-ai-optimization-challenge.devpost.com/
- URL: https://arm-ai-optimization-challenge.devpost.com/rules
- Relevant conclusion: The challenge targets AI optimization on Arm-powered
  platforms across Physical AI, Cloud AI, and Mobile AI. Cloud AI covers Arm64
  cloud, inference performance, frameworks, agents, and production-ready
  developer workflows. The official rules list the submission period as
  2026-06-10 09:00 Pacific Time through 2026-08-14 16:00 Pacific Time.
- Implementation impact: ArmTune should target Cloud AI and prioritize
  measurable Arm64 inference performance, developer workflow, reproducibility,
  public source, setup instructions, and a short demo.
- Unresolved uncertainty: Track naming on the overview says Mobile AI, while
  the rules prize table also mentions Edge AI in one section. The submission
  should follow the active Devpost category UI when entered.

## Arm Performix

- Organization: Arm
- URL: https://developer.arm.com/servers-and-cloud-computing/arm-performix
- Relevant conclusion: Arm Performix is a performance analysis toolkit for
  server and cloud developers on Arm-based infrastructure. It provides
  function-level insights, visualizations, guided recommendations, and
  automation-ready output for Arm Neoverse systems.
- Implementation impact: Performix is a valuable optional profiling evidence
  layer, but ArmTune must still benchmark without it. Integration belongs in
  Phase 5 after the benchmark and passport vertical slice is stable.
- Unresolved uncertainty: The exact CLI output formats and licensing terms
  should be verified from installed tooling or official user guide before
  parser implementation.

## llama.cpp

- Organization: ggml-org
- URL: https://github.com/ggml-org/llama.cpp
- Relevant conclusion: llama.cpp provides local/cloud LLM inference with
  binaries such as `llama-cli` and `llama-server`, and supports multiple
  hardware backends.
- Implementation impact: ArmTune should inspect actual binaries and runtime
  output rather than assuming feature usage from host capability.
- Unresolved uncertainty: Arm-specific build flags and backend evidence should
  be verified against the exact llama.cpp commit used in a run.

## llama-bench

- Organization: ggml-org
- URL: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- Relevant conclusion: `llama-bench` supports multiple output formats,
  including Markdown and CSV. CSV includes fields for build, backend, batch,
  thread, timing, and throughput data.
- Implementation impact: Prefer machine-readable output when available and
  preserve raw output for audit.
- Unresolved uncertainty: Parser tests should cover the exact output generated
  by the runtime version selected for the demo target.

## KleidiAI

- Organization: Arm
- URL: https://github.com/ARM-software/kleidiai
- Relevant conclusion: KleidiAI provides optimized performance-critical
  micro-kernels for AI workloads on Arm CPUs, including specialized kernels for
  Arm CPU architectural features such as DotProd and I8MM.
- Implementation impact: KleidiAI should be treated as workload execution
  evidence only when build or runtime artifacts explicitly prove that an
  integrated path was used.
- Unresolved uncertainty: The initial llama.cpp vertical slice must verify
  whether and how the chosen llama.cpp version exposes KleidiAI evidence.

## Native Arm64 CI Validation

- Source: GitHub Actions
- Workflow: `ArmTune Native Arm64 Smoke`
- Workflow file: `.github/workflows/armtune-arm64-smoke.yml`
- Run URL: https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566
- Run ID: `29270842566`
- Job URL: https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566/job/86887839990
- Job ID: `86887839990`
- Event: `push`
- Conclusion: `success`
- Tested commit: `2074dfac80e699ba22178774b28c66e79afae475`
- Runner evidence from downloaded artifact:
  - OS: `Linux`
  - architecture: `aarch64`
  - kernel: `6.17.0-1018-azure`
- Uploaded artifact name: `armtune-arm64-smoke-artifacts`
- Artifact run ID: `20260713_173254_650848`
- Benchmark result origin: `test_fixture`
- Benchmark parse status: `parsed`
- Downloaded artifact digest validation: `passed`; 13 artifacts checked.
- Focused ArmTune suite at the tested commit: 94 tests.
- Relevant conclusion: ArmTune's controller, worker, provenance, parser, and
  artifact-validation pipeline was validated natively on a GitHub-hosted Linux
  Arm64 runner.
- Claim boundary: the parsed fixture values `321.5 tokens/s` and
  `87.25 tokens/s` are fake workload constants, not Arm performance results.
- Implementation impact: Phase 2C can move to a real SSH Linux Arm64 target,
  real `llama-bench` binary, and user-supplied small GGUF model for a
  preliminary measured baseline.
- Unresolved uncertainty: real llama.cpp, model execution, optimized-path
  evidence, quality validation, Performix, and search remain unvalidated.

## Phase 2C Real SSH Arm64 Baseline

- Source: user-provisioned AWS T4g Linux Arm64 target
- Evidence file: `docs/armtune/results/PHASE_2C_AWS_GRAVITON2.md`
- Run ID: `20260716_231915_147773`
- Target type: `ssh`
- Target architecture: `aarch64`
- Target OS: `Ubuntu 24.04.4 LTS`
- Kernel: `6.17.0-1017-aws`
- CPU model: `Neoverse-N1`
- PerfRunbench commit: `41b57b70b318a94977a5dd822314782e855295f6`
- llama.cpp commit: `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- llama-bench SHA-256:
  `990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8`
- Model SHA-256:
  `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`
- Result origin: `measured`
- Validation status: `passed`
- Execution status: `completed`
- Parse status: `parsed`
- Benchmark status: `passed`
- Artifact validation: `passed`; 13 artifacts checked.
- Fake executable marker: `false`
- Preliminary parsed throughput:
  - prefill: `43.8 tokens/s`
  - decode: `30.13 tokens/s`
- Relevant conclusion: ArmTune's Windows controller, system OpenSSH target
  path, remote worker protocol, real `llama-bench` execution, parser, and
  downloaded artifact digest validation worked on a real Linux Arm64 target.
- Claim boundary: this is a preliminary measured baseline used to validate the
  real SSH Arm64 execution pipeline on a burstable AWS T4g instance. It is not
  evidence of optimization, KleidiAI execution, build-profile improvement,
  thread-search improvement, quality preservation, stable production
  performance, or formal competition benchmark results.
- Implementation impact: Phase 3A can add repeated measurement and Arm-aware
  candidate search using fake tests first, without running a real AWS search as
  part of implementation.
- Unresolved uncertainty: optimized-path evidence, quality validation,
  Performix, Performance Passport generation, and stable production
  performance remain unvalidated.

## Phase 3A.1 Real Search Validation

- Source: user-provisioned AWS Graviton2 Linux Arm64 development target
- Evidence file: `docs/armtune/results/PHASE_3A_REAL_SEARCH_VALIDATION.md`
- Run ID: `20260718_001757_718789`
- Target type: `ssh`
- Target architecture: `aarch64`
- Target OS: `Ubuntu 24.04.4 LTS`
- Kernel: `6.17.0-1017-aws`
- CPU model: `Neoverse-N1`
- Logical CPUs: `2`
- PerfRunbench commit: `ee4cbb305d788e8405e65ab839b1848cb743a2eb`
- llama.cpp commit: `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- llama-bench SHA-256:
  `990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8`
- Model SHA-256:
  `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`
- Search protocol: `warmup_runs=1`, `search_runs=3`,
  `confirmation_runs=5`, `cooldown_seconds=2.0`, `random_seed=42`,
  `insert_paired_baseline_every=2`, `noise_band_percent=5.0`.
- Candidate plan: `baseline`, `threads-1`, `affinity-compact-0_1`,
  `affinity-spread-0_1`, `affinity-reserve-one-core-0`.
- `llama-bench` help supported `--batch-size` and `--ubatch-size`; it did not
  expose `--threads-batch`.
- Result origin: `measured`
- Search status: `completed`
- Parse statuses: `parsed=34`
- Benchmark statuses: `passed=34`
- Failed repetitions: `0`
- Artifact validation: `passed`; 34 repetition manifests and 442 artifact
  records checked.
- Selected candidate: `affinity-spread-0_1`
- Recommendation status: `inconclusive`
- Noise-band classification: `inconclusive`
- Delta percent: `0.116029`
- Quality gate status: `not_evaluated`
- Relevant conclusion: ArmTune's repeated benchmark, staged search,
  statistics, conservative recommendation, SSH worker, provenance, parser, and
  artifact-validation pipeline worked against a real `llama-bench` workload on
  Linux Arm64.
- Claim boundary: this is a pipeline validation run, not final optimization,
  stable production performance, KleidiAI evidence, quality validation,
  build-profile comparison, Performix evidence, Performance Passport evidence,
  or a formal competition benchmark result.
- Implementation impact: Phase 3 hardening should add a configurable candidate
  cap or affinity-mode allowlist for small real-target smoke profiles.
- Unresolved uncertainty: optimized-path evidence, semantic quality,
  build-profile comparisons, Performix, Performance Passport generation, and
  stable production performance remain unvalidated.

## Phase 3A.2 Search Guardrail Smoke Validation

- Source: user-provisioned AWS Graviton2 Linux Arm64 development target.
- Run ID: `20260718_154049_739709`
- PerfRunbench commit:
  `77ed7fb23a4ff8660f57fac857341b705a585050`
- Command shape:
  `autotuneai arm optimize --config <private-config> --warmup-runs 1 --search-runs 1 --confirmation-runs 2 --cooldown-seconds 0 --max-candidates 3 --affinity-modes scheduler-default,compact`
- Search protocol override: `warmup_runs=1`, `search_runs=1`,
  `confirmation_runs=2`, `cooldown_seconds=0.0`, `random_seed=42`,
  `insert_paired_baseline_every=2`, `noise_band_percent=5.0`,
  `max_candidates=3`, `affinity_modes=[scheduler-default, compact]`.
- Candidate plan: `baseline`, `threads-1`, `affinity-compact-0_1`.
- Unique candidate count: `3`.
- Process execution count: `9`; this confirms `max_candidates` limited unique
  candidate configurations, not warm-up, paired-baseline, confirmation, or
  total process executions.
- Effective config artifact: `optimize_config.json` preserved the CLI override
  values for `max_candidates` and `affinity_modes`.
- Search status: `completed`
- Result origin: `measured`
- Quality gate status: `not_evaluated`
- Recommendation status: `inconclusive`
- Artifact validation: `passed`; 9 repetition manifests and 117 artifact
  records checked after download.
- Relevant conclusion: the Windows controller config merge, SSH protocol,
  remote worker execution, bounded search plan, effective-config artifact, and
  downloaded artifact digest validation all worked for the Phase 3A.2 guardrail
  smoke profile.
- Claim boundary: this is a functional validation of candidate-count and
  affinity-mode guardrails on a real Linux Arm64 target. It is not a new
  optimization result, not stable production performance, not KleidiAI
  evidence, not `llama-server` evidence, not TTFT or end-to-end latency
  evidence, and not deterministic quality-gate evidence.

## Phase 3B.1-R Real llama-server Validation

- Source: user-provisioned AWS T4g / Graviton2 Linux Arm64 development target.
- Evidence file:
  `docs/armtune/results/PHASE_3B1_REAL_SERVER_VALIDATION.md`
- Run ID: `20260719_004451_487119`
- Target type: `ssh`
- Target architecture: `aarch64`
- Target OS: `Ubuntu 24.04.4 LTS`
- Kernel: `6.17.0-1019-aws`
- CPU model: `Neoverse-N1`
- Logical CPUs: `2`
- PerfRunbench commit:
  `1b70660812cee0da2ddf40eb1a0bc1af4a51d4c9`
- llama.cpp commit:
  `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- llama-server SHA-256:
  `6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e`
- Model SHA-256:
  `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`
- Protocol: loopback-only `llama-server`, target-local HTTP client,
  OpenAI-compatible SSE, 3 synthetic prompts, 3 warm-up requests, and 6
  measured requests.
- Result origin: `measured`
- Inspection status: `passed`
- Startup status: `ready`
- Readiness status: `ready`
- Request status: `completed`
- Stream parse status: `parsed`
- Benchmark status: `passed`
- Cleanup status: `passed`
- Failed requests: `0`
- Quality gate status: `not_evaluated`
- Artifact validation: `passed`; 48 artifact records checked.
- TTFT distribution: valid `6`, failed `0`, median `41.105 ms`, p95
  `45.24175 ms`.
- End-to-end latency distribution: valid `6`, failed `0`, median
  `762.659 ms`, p95 `908.2335 ms`.
- Real SSE compatibility: role-only metadata deltas were ignored for TTFT,
  generated content arrived through `choices[0].delta.content`, runtime
  timing metadata was preserved separately, and the stream ended with
  `data: [DONE]`.
- Relevant conclusion: ArmTune's `llama-server` lifecycle, target-local
  loopback HTTP client, streaming TTFT and end-to-end latency measurement,
  generated-output preservation, cleanup, SSH worker, provenance, and
  artifact-validation pipeline worked against a real `llama-server` workload
  on Linux Arm64.
- Claim boundary: this is a bounded server-pipeline validation run. It is not
  final optimization, stable production performance, KleidiAI evidence,
  quality validation, build-profile comparison, Performix evidence,
  Performance Passport evidence, or a formal competition benchmark result.

## Phase 4-L Local Fake Quality-Aware Server Search

- Source: local fake `llama-server` fixture and public synthetic BootTrace
  quality suite.
- Quality suite: `configs/armtune/quality/boottrace_suite.json`
- CLI shape:
  `autotuneai arm optimize --config <local-fake-quality-config>`
- Result origin: `test_fixture`
- Validated behavior: server optimize config parsing, derived prompt-suite
  snapshot, repeated `benchmark.llama_server` candidate execution,
  deterministic quality validation, faster quality-failing candidate rejection,
  quality-passing candidate eligibility, conservative recommendation state,
  cleanup evidence, and root artifact digest validation.
- Claim boundary: fake TTFT, end-to-end latency, generated outputs, and quality
  responses are fixtures. They prove pipeline behavior only and must not be
  reported as Arm performance, model quality, or optimized Arm execution
  evidence.

## Phase 4-R Real Quality-Aware Rejection Validation

- Source: user-provisioned AWS T4g / Graviton2 Linux Arm64 development target.
- Evidence file:
  `docs/armtune/results/PHASE_4_REAL_QUALITY_AWARE_TUNING.md`
- Run ID: `20260719_035956_457526`
- Target type: `ssh`
- Target architecture: `aarch64`
- Target OS: `Ubuntu 24.04.4 LTS`
- Kernel: `6.17.0-1019-aws`
- CPU model: `Neoverse-N1`
- Logical CPUs: `2`
- PerfRunbench commit:
  `33504d731bdb11316d439609cb3bf3145401b9e8`
- llama.cpp commit:
  `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- llama-server SHA-256:
  `6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e`
- Model SHA-256:
  `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55`
- Quality suite ID: `synthetic_boottrace_quality_v1`
- Quality suite digest:
  `42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169`
- Protocol: loopback-only `llama-server`, target-local HTTP client, five
  synthetic BootTrace quality cases, `warmup_runs=1`, `search_runs=2`,
  `confirmation_runs=2`, `max_candidates=3`, and
  `affinity_modes=[scheduler-default, compact]`.
- Candidate plan: `baseline`, `server-batch256-ubatch128`, `threads-1`.
- Result origin: `measured`
- Candidate count: `3`
- Eligible candidate count: `0`
- Failed candidate count: `3`
- Recommendation state: `no_eligible_candidate`
- Quality gate status: `failed`
- Required quality cases passed: `0/5`
- Artifact validation: `passed`; 224 artifact records checked.
- CLI exit code: `1`
- Cleanup evidence: every candidate reported `cleanup_status: passed`; a
  post-run process check found no residual `llama-server` or ArmTune worker
  processes.
- Performance pipeline evidence: all candidates completed real measured server
  requests with `request_status: completed`, `stream_parse_status: parsed`,
  `benchmark_status: passed`, and `failed_request_count: 0`.
- Quality rejection evidence: all candidates were marked `rejected_quality`
  because generated outputs were not raw JSON and failure cases were repeatedly
  mislabeled as `success`.
- Claim boundary: this validates real Arm execution, streaming latency
  measurement, deterministic quality evaluation, quality-aware rejection,
  cleanup, and artifact validation. It does not validate a quality-preserving
  recommendation, final optimization, stable production performance, KleidiAI,
  build-profile improvement, Performix, Performance Passport output, or formal
  competition benchmark results.

## Phase 6B-R Qwen2.5-7B Qualification Rejection

- Source: user-provisioned AWS C7g / Graviton3 Linux Arm64 development target.
- Evidence file:
  `docs/armtune/results/PHASE_6_QWEN25_PROFILE_COMPARISON.md`
- Run ID: `20260719_152540_406176`
- Target type: `ssh`
- Target architecture: `aarch64`
- Target OS: `Ubuntu 24.04.4 LTS`
- Kernel: `6.17.0-1019-aws`
- CPU model: `Neoverse-V1`
- Logical CPUs: `8`
- PerfRunbench remote worker commit:
  `ff6a31531db48f423915ae37875afa90c949a0c4`
- llama.cpp source commit:
  `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
- llama-server SHA-256:
  `6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e`
- Model repository: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Model repository revision:
  `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
- Merged model SHA-256:
  `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49`
- Quality suite ID: `synthetic_boottrace_quality_v1`
- Quality suite digest:
  `42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169`
- Prompt contract SHA-256:
  `8c4af720b8e5da01d506c4aeba0dde3b588d21693e04eae59c2230ab0a80e01b`
- Protocol: loopback-only `llama-server`, target-local HTTP client, five
  synthetic BootTrace quality cases, `threads=8`, `context_size=4096`,
  `temperature=0.0`, `seed=42`, strict normalization disabled, and one
  measured request per case.
- Result origin: `measured`
- Request success count: `5/5`
- Raw JSON valid count: `5/5`
- Required quality cases passed: `0/5`
- Critical validator failures: `22`
- Failure categories: `missing_required_evidence`,
  `model_capability_failure`, `response_format_violation`,
  `wrong_root_cause`, and `wrong_status`.
- Artifact validation: `passed`; 40 quality-calibration artifact records and
  32 nested benchmark artifact records checked.
- CLI exit code: `1`
- Cleanup evidence: benchmark reported `cleanup_status: passed`; a post-run
  process check found no residual `llama-server` or ArmTune worker processes.
- Qualification outcome: `model_profile_suitability: unsuitable`.
- Phase 6C outcome: build-profile comparison and Performance Passport
  generation were not executed because the qualification gate did not pass.
- Claim boundary: this validates real Arm execution, Qwen2.5 split GGUF
  acquisition and merge, deterministic quality rejection, cleanup, and
  artifact validation. It does not validate a suitable model/profile,
  quality-preserving recommendation, final optimization, stable production
  performance, KleidiAI, build-profile improvement, Performix, Performance
  Passport output, or formal competition benchmark results.

## Phase 7-L BootTrace Contract Calibration Foundation

- Evidence file:
  `docs/armtune/results/PHASE_7_BOOTTRACE_CONTRACT_CALIBRATION.md`
- Held-out evaluation suite:
  `configs/armtune/quality/boottrace_suite.json`
- Development calibration suite:
  `configs/armtune/quality/boottrace_calibration_suite.json`
- Held-out case audit:
  `configs/armtune/quality/boottrace_case_audit.json`
- Prompt contract versions:
  `current-zero-shot`, `taxonomy-and-decision-rules`, and
  `taxonomy-rules-and-disjoint-few-shot`.
- Implementation boundary: local code/tests and public documentation only.
  No paid cloud resource was started and no real Qwen2.5 Phase 7 rerun was
  executed in this foundation step.
- Validation: local focused tests covered audit coverage, calibration suite
  disjointness, prompt contract digest stability, distinct rendered prompts,
  prompt-contract artifact recording, request JSON preservation, and SSH
  prompt-suite upload command shape.
- Claim boundary: this validates prompt-contract infrastructure and evidence
  recording. It does not validate Qwen2.5 suitability, a stronger model,
  profile comparison, Performance Passport output, Arm performance
  improvement, KleidiAI, Performix, or formal competition benchmark results.
