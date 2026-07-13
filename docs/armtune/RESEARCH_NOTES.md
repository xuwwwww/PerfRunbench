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
