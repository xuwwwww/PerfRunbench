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
