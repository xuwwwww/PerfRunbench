# Phase 19 T4g CPU-Aware Build Tuning

Status: pending real T4g validation.

Phase 19 validates CPU-aware native build tuning on a constrained Arm target.
It is a bounded low-resource validation, not a global Arm performance claim and
not comparable to the historical C7g Phase 9 numbers.

## Scope

Target class:

```text
Linux aarch64
AWS Graviton2 / Neoverse-N1 class
2 logical CPUs
approximately 2 GiB memory
burstable CPU behavior
```

Policies:

- use only the existing user-started T4g.small target
- do not access the stopped paid C7g target
- do not create, start, stop, resize, terminate, or otherwise control AWS
- do not modify EBS
- do not add swap
- do not download new models or datasets
- stop after this single bounded Phase 19 validation

## Phase 19 Source Lock

To be filled after real validation:

```text
upstream_repository: pending
tag: pending
source_commit: pending
source_dirty_state: pending
cmake_version: pending
compiler_version: pending
linker_version: pending
```

Historical Phase 9 `llama.cpp` evidence remains separate and is not
reinterpreted as Phase 19 evidence.

## Build Candidates

Planned candidate cap: at most six unique candidates.

| Candidate | Profile ID | Planned status |
| --- | --- | --- |
| A | `upstream-default` | required reasonable baseline |
| B | `portable-reference` | use only when supported |
| C | `native-current-cpu` | required when `GGML_NATIVE=ON` is supported |
| D | `explicit-current-cpu` | use only after compiler probe succeeds |
| E | `native-kleidiai` | eligible only with build and runtime evidence |
| F | `native-kleidiai-lto` | attempt only after KleidiAI and LTO probes succeed |

## Real Validation Results

To be filled after real validation:

```text
target_fingerprint: pending
detected_cpu: pending
detected_features: pending
model_identity: pending
model_sha256: pending
build_candidates_requested: pending
build_candidates_executed: pending
verified_kleidiai_state: pending
lto_state: pending
optional_blas_state: pending
fixed_build_profile_comparison: pending
runtime_search_result: pending
complete_configuration_result: pending
hardware_counters: pending
native_binary_portability_scope: pending
passport_verification: pending
recommendation_cache_apply_behavior: pending
artifact_validation: pending
cleanup: pending
```

## Claim Boundary

Allowed wording after successful measured validation:

> Best observed eligible configuration under the tested CPU, model, runtime
> source commit, compiler, and measurement protocol.

Disallowed:

- global Arm speedup
- direct comparison with Phase 9 C7g numbers
- combined build-profile plus runtime-search percentage
- KleidiAI claim from requested CMake flags alone
- portable reuse of CPU-specific binaries without native-binary guard evidence

