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
