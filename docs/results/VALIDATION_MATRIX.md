# PerfRunbench Validation Matrix

This matrix separates portability and guard evidence from performance claims.
Numbers from different targets or protocols must not be combined.

| Phase | Target | Workload | Purpose | Result boundary | Status |
| --- | --- | --- | --- | --- | --- |
| 14B | Linux Arm64 C7g / Neoverse-V1 | Public synthetic PyTorch training | Real Arm training-adapter workflow and guarded apply | Single Arm target, synthetic workload | passed |
| 15A | Linux Arm64 T4g.small / Neoverse-N1 class | Dependency-free fixture, PyTorch skipped when unsafe | Low-resource guard and portability evidence | No performance claim | passed |
| 15B | Linux Arm64 T4g.small / Neoverse-N1 class | Public synthetic PyTorch training | Real low-resource Arm PyTorch workflow | Do not compare with C7g | passed |
| 15C | Windows x86_64 controller | Public synthetic PyTorch training | Local CLI, cache, drift, apply, and portability validation | Do not compare with Arm | passed |
| 16 | Windows local clean venv plus GitHub Actions matrix | Dependency-free public fixtures | Release-candidate installed-wheel usability | Packaging and CLI usability, not performance | pending final CI |
| 19 | Linux Arm64 T4g.small / Neoverse-N1 class | No suitable existing model available; build evidence only | CPU-aware native build tuning guard validation | Build artifacts collected, but no benchmark or recommendation claim | insufficient evidence |
| 20 | Linux Arm64 C7g / Neoverse-V1 | Qwen2.5-7B Q4_K_M with pinned llama.cpp | CPU-aware build/backend evidence, bounded runtime search, server timing, Passport, and artifact validation | Real measured workflow validation; build and runtime improvements stayed inside the 5% noise band | passed |
| 21A | Local fake Linux roots | Synthetic `/proc`, `/sys`, cgroup, IRQ, block, and boot fixtures | Host system and kernel profile framework, leases, rollback, and boot-plan safety | Planning and fake E2E only; no AWS and no real host mutation | passed |

## Current Evidence Boundaries

- Arm RecommendationBundles do not apply on Windows x86_64 because architecture
  and CPU-count hard fingerprints differ.
- Windows RecommendationBundles are bound to the tested Windows x86_64
  environment and are not portable to Arm targets.
- Fixture, smoke, and synthetic-training measurements are reproducibility and
  guard evidence unless a document explicitly marks them as a target-specific
  measured result.
- `run --best` never retunes. It requires an existing eligible cache entry and
  refuses to launch when no usable recommendation is available.
- `run --auto` may retune on cache miss or hard drift according to policy; exact
  cache reuse does not create a new tuning result.
- CPU-specific binary reuse requires native-binary portability evidence in
  addition to workload and environment identity. Neoverse-N1 and Neoverse-V1
  CPU-model-bound binaries are not interchangeable by default.

## Phase 15C Summary

```text
source_commit: 5bf2a0d217283e8da2ed4954a9be65a4be4ed954
target: Windows x86_64 / 8 logical CPUs
torch_version: 2.13.0+cpu
candidate_count_total_including_baseline: 6
recommendation_bundle_id: c61f327d271288d01437d435
bounded_slice_status: recommended
artifact_manifest_revalidation_status: passed
cleanup_status: passed
```

Phase 15C also validated the CPU-affinity mask canonicalization fix for
Windows-style hex masks such as `0xff`.

## Phase 16 Release Candidate Scope

Phase 16 validates that a fresh user can install the built distribution,
initialize a config, check readiness with `autotuneai doctor`, run bounded
tuning through `run --auto`, reuse recommendations, and inspect the cache
without knowing internal Passport, RecommendationBundle, or cache paths.

Required CI platforms:

```text
windows-latest
ubuntu-latest
ubuntu-24.04-arm
```

The required CI matrix uses dependency-free fixtures and must not install
PyTorch, download models/datasets, access AWS, or turn fixture evidence into
measured performance evidence.

## Phase 20 C7g Build/Backend Summary

Phase 20 returned to a user-provisioned C7g / Neoverse-V1 target with the
existing Qwen2.5-7B Q4_K_M model and pinned `llama.cpp` source commit
`47c786924ad1ab7e91da2cdc72fcdb563780c2bd`.

The validation completed build-profile comparison, bounded runtime search,
secondary `llama-server` measurement, Performance Passport generation and
verification, remote artifact revalidation, and downloaded artifact
revalidation.

Outcome:

```text
build_profile_recommendation: inconclusive
runtime_search_recommendation: inconclusive
selected_runtime_candidate: baseline
KleidiAI_status: requested profiles rejected unless runtime backend evidence exists
Performance_Passport: verification passed, claim_eligibility_state ineligible_provenance
artifact_revalidation: 77/77 passed remotely and after download
performance_improvement_verified: false
```

## Phase 21A System/Kernel Framework Summary

Phase 21A adds a guarded local framework for future Linux system and kernel
autotuning:

```text
AWS_access: false
real_host_mutation: false
snapshot_model: HostStateSnapshot
profile_catalog: capability-driven safe-system
reversible_apply: fake-root only
rollback: reverse-order exact restoration
boot_profiles: planning only, no GRUB writes
protocol_separation: enforced
dirty_host_recommendation: ineligible
```

This phase does not provide a performance result. It prepares future C7g
validation for reversible system profiles and reboot-required kernel profiles.
