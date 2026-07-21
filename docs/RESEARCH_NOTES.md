# PerfRunbench Research Notes

## Phase 15C - Windows x86 PyTorch Validation

Phase 15C validated PerfRunbench's golden-path CLI and optional
`pytorch-training` adapter on a local Windows x86_64 controller while AWS
remained out of scope.

Key observations:

- The local target satisfied the preflight guard for disk, memory, Python 3.11,
  clean checkout, and CPU-only PyTorch installation.
- Windows command handling preserved argv entries containing spaces, Unicode,
  quoted values, and flags after `--`.
- The public synthetic PyTorch workload completed a CPU-only smoke test with
  finite loss.
- `run --best` refused an empty cache without retuning or launching the
  workload.
- `run --auto` on cache miss tuned a bounded candidate set, produced an eligible
  RecommendationBundle, ran canary validation, and launched the full workload.
- A second `run --auto` and subsequent `run --best` reused the exact cache entry
  without creating a new tuning result.
- C7g and T4g Arm RecommendationBundles hard-mismatched on Windows x86_64 and
  dry-run apply failed without launching a Windows workload.

Phase 15C found and fixed one correctness defect: Windows-style hex CPU
affinity masks were not normalized as equivalent to explicit CPU lists. The fix
is recorded as:

```text
5bf2a0d fix: normalize hex CPU affinity masks
```

See `docs/results/PHASE_15C_WINDOWS_X86_PYTORCH_VALIDATION.md` for the
sanitized evidence summary.

## Phase 19 - CPU-Aware Native Build Tuning

Phase 19 shifts ArmTune build tuning from named profiles such as
`arm-native` toward explicit CPU-aware build evidence:

- one pinned upstream `llama.cpp` source commit per Phase 19 run
- upstream default Release build as the reasonable baseline
- portable reference build only when supported by the pinned source
- `GGML_NATIVE=ON` native build with compile-command evidence
- explicit CPU target only after a compiler probe succeeds
- KleidiAI build profiles only when build and runtime backend evidence both
  support the claim
- optional LTO only after toolchain probing

The design intentionally keeps historical Phase 9 C7g evidence separate from
Phase 19 T4g evidence. A new `llama.cpp` source lock changes the benchmark
protocol, so old and new percentages must not be combined.

The selected public wording remains:

> Best observed eligible configuration under the tested CPU, model, runtime
> source commit, compiler, and measurement protocol.
