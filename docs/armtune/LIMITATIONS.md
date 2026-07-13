# ArmTune Limitations

## Current Phase 0 Limitations

- No ArmTune production code exists yet.
- No `autotuneai arm` commands exist yet.
- No local or SSH Arm target has been inspected by ArmTune yet.
- No llama.cpp binary has been inspected by ArmTune yet.
- No ArmTune benchmark has been executed.
- No measured Arm performance result exists in this repository.
- No Arm Performance Passport has been generated.
- No Performix integration exists.

## MVP Out of Scope

- Automatic cloud instance provisioning.
- Kubernetes orchestration.
- Hosted SaaS dashboards.
- User accounts.
- Custom Neon or SVE assembly.
- A new inference engine.
- Model training.
- Automatic quantization.
- ONNX Runtime, ExecuTorch, or vLLM adapters before the llama.cpp vertical
  slice is stable.
- Power measurement without reliable hardware support.
- Cost estimates without verified current pricing.
- x86-vs-Arm comparisons as the primary product result.

## Result Claim Limitations

Existing PerfRunbench benchmark results are useful prior work but are not
ArmTune production Arm results. They must not be used as evidence that an Arm
target, Arm optimized path, KleidiAI, SVE, I8MM, BF16, SME, or Performix was
used.

Any future result missing raw artifacts, provenance, or comparability checks
should be labeled unverified.

## Phase 2A Limitations

- Phase 2A is a minimal llama-bench execution slice, not an optimization
  system.
- Phase 2A runs one fixed configured invocation only.
- Phase 2A does not implement candidate search, warmup logic, paired
  baselines, confirmation runs, or noise-band decisions.
- Phase 2A does not implement a quality gate.
- Phase 2A does not verify KleidiAI or any other optimized Arm path.
- Phase 2A parser support is limited to explicit llama-bench table/text
  strategies covered by fixtures. Unknown formats are reported as
  `unsupported_format` rather than converted into metrics.
- Phase 2A fake executable tests are `test_fixture` results, not measured Arm
  results.
- Real production Arm benchmark validation remains out of scope for Phase 2A.

## Phase 2B Limitations

- GitHub native Arm64 CI proves packaging, CLI, host inspection, focused tests,
  and fixture benchmark plumbing on an Arm64 Linux runner only.
- The completed GitHub run
  `https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566` validated
  commit `2074dfac80e699ba22178774b28c66e79afae475` on Linux `aarch64`
  kernel `6.17.0-1018-azure`.
- Downloaded artifact validation passed for artifact
  `armtune-arm64-smoke-artifacts`; 13 artifacts were checked and
  `result_origin` remained `test_fixture`.
- GitHub native Arm64 CI does not prove real model performance, inference
  optimization, KleidiAI usage, Performix support, or competition benchmark
  results.
- Workflow timing is not benchmark data.
- Fixture values such as `321.5 tokens/s` prefill and `87.25 tokens/s` decode
  are fake workload constants and must not be reported as Arm performance.
- Fake llama-bench output remains `test_fixture` on real Arm hardware.
- Phase 2B does not use AWS, external SSH servers, cloud provisioning, real
  models, `llama-server`, time to first token, candidate search, quality gates,
  build profiles, Performix, or Performance Passport generation.
- Real-Arm smoke scaffolding is skipped by default and requires explicit
  user-supplied binary and model paths plus acknowledgement.
- Phase 2C must provide a real model, real target, and explicit measured-data
  handling before any preliminary measured result is run.
