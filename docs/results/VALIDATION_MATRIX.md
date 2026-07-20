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
