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
