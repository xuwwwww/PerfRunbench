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

Real T4g follow-up:

- pinned `llama.cpp` source: `b9999` /
  `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`
- `gcc -mcpu=neoverse-n1` probe: supported, produced binary ran
- `upstream-default` and `native-current-cpu` both built `llama-bench`
- both builds were marked `built_runtime_unverified` because no suitable
  existing small GGUF model was present on the target
- artifact manifests validated remotely and after download
- no benchmark, runtime search, Passport, or RecommendationBundle was produced

This is an `insufficient_evidence` guard result, not a speedup claim.

## Phase 20 - C7g CPU-Aware Build Backend And Runtime Tuning

Phase 20 returned to a real C7g / Neoverse-V1 target with the existing
Qwen2.5-7B Q4_K_M model and pinned `llama.cpp` source
`47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.

Key observations:

- `explicit-neoverse-v1` was added as an explicit profile and built
  successfully.
- `portable-reference` failed to build under the pinned source/build policy and
  was preserved as rejected evidence.
- `native-kleidiai` and `native-kleidiai-lto` built and measured, but were not
  eligible because KleidiAI runtime backend usage was not evidenced.
- Fixed build-profile comparison found `explicit-neoverse-v1` at
  `17.78 tok/s` decode median versus `upstream-default` at `17.71 tok/s`;
  the `0.395257%` delta stayed inside the 5% noise band.
- Bounded runtime search tested six unique candidates and selected the
  baseline; complete confirmation median decode was `17.72 tok/s`.
- Secondary `llama-server` measurement completed 3/3 measured requests with
  median TTFT `61.470 ms` and median E2E latency `911.976 ms`; quality was
  `not_evaluated`.
- Performance Passport generation and verification passed, but the claim state
  was `ineligible_provenance` because the measured comparison did not produce
  an eligible performance claim.
- Artifact manifests validated remotely and after download: `77/77`.

Conclusion:

```text
phase20_c7g_build_backend_runtime_tuning: passed
performance_improvement_verified: false
```

## Phase 21A - Host System And Kernel Profile Framework

Phase 21A adds a local fake-root framework for Linux system and kernel profile
planning. It deliberately avoids AWS access and real host mutation.

Key decisions:

- Add `autotune.system_profiles` instead of extending the legacy
  `autotune.system_tuner` sysctl helper in place.
- Treat unavailable host interfaces as `not_available`, not false or zero.
- Generate profiles from discovered capabilities instead of assuming CPUFreq,
  cpuidle, cgroup v2 partitions, IRQ affinity, THP sizes, HugeTLB, RPS/XPS,
  GRUB, or 64K kernels exist.
- Keep system profiles protocol-specific across steady-state decode, cold
  model load, warm first request, interactive server, and concurrent server.
- Refuse real-system writes in Phase 21A unless a future phase adds explicit
  opt-in.
- Bind non-read-only changes to a lease and require reverse-order rollback.
- Mark dirty host state as recommendation-ineligible.

Phase 21A does not overturn Phase 20. Phase 20 showed no build/simple-runtime
candidate outside the 5% noise band. System and kernel mechanisms remain
unevaluated until Phase 21B/21C.
