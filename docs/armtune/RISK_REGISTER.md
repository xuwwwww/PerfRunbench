# ArmTune Risk Register

| Risk | Impact | Mitigation | Status |
| --- | --- | --- | --- |
| x86 controller measurements are mistaken for Arm results. | Invalid competition claims. | Production benchmark commands must reject non-Linux-Arm64 targets. | Open |
| Host features are treated as runtime execution evidence. | False optimized-path claims. | Store host capabilities and workload evidence separately. | Open |
| Incompatible experiments are compared. | Misleading speedups or regressions. | Passport comparison must validate model, prompt, runtime, target, and protocol compatibility. | Open |
| SSH command construction leaks secrets or executes unsafe input. | Security and privacy issue. | Use fixed worker entry point and versioned JSON job specs. | Open |
| Arm-specific logic spreads into generic PerfRunbench modules. | Maintenance and backward-compatibility risk. | Keep Arm logic under `autotune.arm_tune`; add generic abstractions only when justified. | Open |
| Parser failures become zero metrics. | False successful trials. | Failed parsing must produce trial failure records. | Open |
| Raw artifacts are not preserved. | Results cannot be audited. | Store raw stdout/stderr and artifact digests under run directories. | Open |
| Temporary tuning is not restored after failure. | Target state pollution. | Reuse existing lifecycle patterns and test SIGINT/SIGTERM cleanup. | Open |
| Performix becomes required for benchmarking. | MVP becomes harder to run. | Keep Performix optional and Phase 5 only. | Open |
| Model files or private logs are committed. | Privacy/license violation. | Do not store model weights in run artifacts; document public/synthetic data rules. | Open |
| Documentation implies measured Arm results before Phase 6. | Integrity violation. | Label all templates and examples clearly. | Open |
| Challenge before/after claim includes pre-challenge work. | Misleading submission. | Maintain `BEFORE_AFTER.md` and cite the pre-challenge baseline commit. | Open |
