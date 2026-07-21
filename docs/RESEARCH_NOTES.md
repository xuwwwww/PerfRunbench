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

## Phase 21B - C7g Reversible System Validation

Phase 21B validated bounded reversible host-system tuning on the same C7g /
Neoverse-V1 class target and fixed Qwen2.5-7B Q4_K_M `llama-bench` protocol.

Key observations:

- The target exposed no CPUFreq policy, so governor tuning was correctly
  `unsupported`.
- `thp-never`, `thp-always`, and `vm-system-conservative` were applied under
  lease, measured, rolled back, and verified against the original normalized
  snapshot.
- `vm-system-conservative` temporarily changed only `vm.stat_interval` and
  `kernel.sched_autogroup_enabled`, then restored both.
- `affinity-cpus-1-7` regressed by `-10.493%` because it reduced the workload
  CPU set and thread count.
- Positive candidates stayed inside the 5% noise band; no recommendation was
  created.
- The final normalized snapshot matched the original, and remote/local artifact
  validation checked `712/712` artifacts.

Phase 21B found two real-host correctness issues and fixed them:

```text
891a3c4 fix: tolerate volatile system queue masks
f723384 fix: normalize bracketed sysfs mode verification
```

Conclusion:

```text
phase21b_c7g_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
```

## Phase 21B2 - Remaining C7g Reversible System Validation

Phase 21B2 exhausted the remaining safe reversible C7g system families without
entering Phase 21C boot/kernel changes.

Key observations:

- `no-mmap` reached `16.75 tok/s` decode median versus the `16.26 tok/s`
  steady-state baseline, a `+3.014%` delta inside the 5% noise band.
- `thp-always-mmap` proved actual THP use with
  `max_anon_huge_pages_kb: 4483072`, but its `+1.630%` delta also stayed
  inside the guard.
- `cpuset-1h7w`, `cpuset-2h6w`, and `cpuset-1h7w-irq-workqueue` sharply
  regressed this workload and were rejected.
- Storage read-ahead and block scheduler measurements were preserved as
  `cold_model_load` evidence and were not ranked against steady-state decode.
- `prefault-warm` was preserved as `warm_page_cache_decode` evidence and was
  not ranked against steady-state decode.
- `jemalloc` and `mimalloc` were unavailable through system libraries and
  non-persistent extraction.
- HugeTLB was not qualified because no verified workload consumer or
  `hugetlbfs` mapping path existed in the run.
- Non-root `perf` remained denied by host security policy; root `perf stat`
  was available without weakening security and a baseline perf collection
  passed.
- Secondary `llama-server` smoke completed, but its timing remained secondary
  runtime evidence and did not affect eligibility.

Phase 21B2 found one real-host correctness issue and fixed it:

```text
88a2cee fix: normalize bracketed block scheduler mode
```

The downloaded artifact revalidation also caught a private harness
manifest-ordering issue before documentation. The finalizer was rerun with a
stable final summary before manifest generation, and both remote and local
manifest validation passed.

Conclusion:

```text
phase21b2_remaining_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
all_reversible_families_exhausted: true
ready_for_phase21c_boot_kernel_validation: false
```

## Phase 21C0 - C7g Boot And Kernel Validation Readiness

Phase 21C0 translates the Phase 21B2 boot-readiness blocker into a concrete,
versioned, read-only planning model. It is a local implementation and fixture
validation phase only. AWS is not accessed and no kernel, GRUB, EFI,
initramfs, kernel command line, or reboot state is modified.

Key decisions:

- Keep boot/kernel plans separate from reversible runtime system leases.
- Require a complete recovery capability model before any future reboot.
- Treat missing kernel config, Secure Boot, lockdown, bootloader, initramfs,
  package metadata, or fallback evidence as unknown or blocked, never guessed.
- Use one-time boot selection only; never change the permanent default before
  a successful validation cycle.
- Return to the original 4K known-good state between every experimental
  profile.
- Keep steady-state decode, cold model load, warm first request, and stability
  protocols separate.

The local fixture suite covers 4K and 64K inventory, fallback kernel
qualification, duplicate command-line parameter refusal, one-time boot plan
verification, recovery plan generation, post-boot health failure
classification, and original-state fixture verification.

Remote C7g read-only audit observations:

- The running target was Linux `aarch64` with 8 logical CPUs, GRUB UEFI, AWS
  4K kernel `6.17.0-1019-aws`, and base page size `4096`.
- The audit verified a known-good fallback kernel and stable read-only
  inventory comparison.
- Official AWS ARM64 64K package metadata was found for
  `linux-image-6.14.0-1018-aws-64k` version `6.14.0-1018.18~24.04.1`.
- The 64K candidate state is `metadata_qualified_not_installed`; no package was
  installed and no 64K kernel was booted.
- Recovery snapshot and out-of-band recovery inputs were not verified, so every
  modified boot profile remained blocked.
- Remote and downloaded artifact manifests both validated; downloaded
  validation checked `599/599` artifacts.
- No audit, boot, or autotune process remained after the run.

The remote audit required one focused parser fix:

```text
0280f6d fix: discover AWS 64K kernel metadata from apt
```

Conclusion:

```text
phase21c0_remote_readonly_audit: passed
ready_for_phase21c_real_boot_validation: false
kernel_or_boot_state_modified: false
```

Phase 21C real boot changes remain future work.

## Phase 21C1 - Same-Kernel Boot Rehearsal

Phase 21C1 used the validated recovery path to perform exactly one same-kernel
guest reboot through a one-time GRUB entry. It did not install kernels, boot a
64K kernel, alter EFI BootOrder, change the permanent GRUB default, or run a
benchmark workload.

Key observations:

- Recovery snapshot, Serial Console login, Serial Console sudo, fallback
  kernel, and one-time boot selection were verified before the reboot.
- The host returned with the same 4K AWS kernel, same base page size, and same
  command line.
- GRUB left an empty `next_entry=` key after consuming the one-time value; the
  run cleared it and verified final `grubenv` matched the pre-arm state.
- Downloaded artifact validation checked `30/30` artifacts.

Conclusion:

```text
phase21c1_one_time_boot_rehearsal: passed
ready_for_phase21c_real_boot_validation: true
performance_improvement_verified: false
```

## Phase 21C2A - 4K Boot Isolation Validation

Phase 21C2A executed the first real 4K boot-parameter experiment on the C7g /
Neoverse-V1 target using the installed AWS 4K kernel. It remained a 4K-only
experiment; 64K kernel installation and validation remain out of scope until a
separate Phase 21C2B decision.

Key decisions:

- Limit this phase to the installed known-good AWS 4K kernel.
- Keep 64K kernel installation and validation out of scope until Phase 21C2B.
- Generate isolated temporary GRUB custom entries instead of editing the
  generated Ubuntu menu entry.
- Preserve all original kernel arguments and append only allowlisted
  parameters.
- Require `0-1` housekeeping CPUs, `2-7` workload CPUs, supported kernel config
  evidence, and a one-reboot transaction returning to G0.
- Separate E1 full-capacity baseline from E2 matched causal baseline and E3
  experimental profile.

The real run exposed a pre-measurement temporary-entry path defect on the first
G1 attempt. Serial Console recovery selected the original Ubuntu entry, cleanup
returned the host to verified G0, and the corrected harness preserved the
original GRUB kernel/initrd paths. Corrected G1/G2/G3 runs then completed.

Result:

```text
G0_E2_decode_median: 13.52 tok/s
G1_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G2_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G3_E3_decode_median: 12.59 tok/s, -6.88%, regressed
remote_artifact_revalidation: 410/410 passed
downloaded_artifact_revalidation: 410/410 passed
original_4k_state_restored: true
permanent_default_unchanged: true
performance_improvement_verified: false
latency_or_noise_improvement_verified: false
```
