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
