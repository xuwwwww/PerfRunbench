# Phase 19 T4g CPU-Aware Build Tuning

Status: insufficient evidence.

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

```text
upstream_repository: https://github.com/ggml-org/llama.cpp
tag: b9999
source_commit: 47c786924ad1ab7e91da2cdc72fcdb563780c2bd
source_dirty_state: clean
source_archive_digest: 564fd25c25dc9b746cb64656565857433dfa74a18b6e3d7a98098346303996f9
cmake_version: 4.4.0
compiler_version: gcc 13.3.0
linker_version: GNU ld 2.42
```

Historical Phase 9 `llama.cpp` evidence remains separate and is not
reinterpreted as Phase 19 evidence.

The older `master-fff0e0e` tag was checked first and rejected for Phase 19
because it did not contain the current GGML CPU build infrastructure required
for this validation.

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

```text
controller_source_commit_for_real_run: 6cb5591d362de63084f73483df6031597aa0d3ed
remote_source_commit_for_real_run: 6cb5591d362de63084f73483df6031597aa0d3ed
target_class: Linux aarch64 / T4g.small / Neoverse-N1 class / 2 logical CPUs
detected_features: fp, asimd, aes, pmull, sha1, sha2, crc32, atomics, fphp,
  asimdhp, cpuid, asimdrdm, lrcpc, dcpop, asimddp
available_memory_before_build: approximately 1.5 GiB
free_root_space_before_build: approximately 25 GiB
swap: none
model_identity: not available
model_sha256: not available
build_run_id: 20260721_024332_303417
build_candidates_requested: upstream-default, native-current-cpu
build_candidates_executed: 2
artifact_validation_remote: passed, 26 artifacts checked
artifact_validation_after_download: passed, 26 artifacts checked
cleanup: no residual listener, worker, active lock, or active lease observed
```

The target did not contain an existing suitable small GGUF model. Only
`llama.cpp` vocabulary fixture GGUF files were present under the source tree.
Per Phase 19 model policy, PerfRunbench did not download a replacement model
and did not treat vocabulary fixtures as benchmark models.

## Build Evidence Summary

| Profile | Build status | Eligibility | `GGML_NATIVE` | Effective CPU flag | Binary SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `upstream-default` | `built_runtime_unverified` | `built_runtime_unverified` | `ON` | `-mcpu=neoverse-n1+crc+crypto+dotprod+noi8mm+nosve` | `d224776aa9554ca3b44dc1975b14429265ed85324cfef3110e1cc65d82a970b3` |
| `native-current-cpu` | `built_runtime_unverified` | `built_runtime_unverified` | `ON` | `-mcpu=neoverse-n1+crc+crypto+dotprod+noi8mm+nosve` | `a06d32a79769d867b722291c928ac6633233730481907ac5d9c518ae2ccb3d24` |

The upstream default build already resolved to `GGML_NATIVE=ON` on this source
and target. The native-current-cpu profile produced the same effective CPU flag
shape, so it must not be presented as a distinct performance improvement.

KleidiAI, LTO, and optional BLAS profiles were not attempted because the run
already lacked the model-load smoke requirement needed to rank verified build
profiles.

## Compiler Probe

```text
probe: gcc -mcpu=neoverse-n1
supported: true
produced_binary_ran: true
```

The explicit CPU target probe passed, but it was not promoted into a separate
candidate because the built upstream-default and native-current-cpu profiles
already showed the same effective CPU flag shape.

## Missing Evidence

The following Phase 19 outputs were intentionally not produced:

- fixed build-profile performance comparison
- runtime candidate search
- fresh complete-configuration confirmation
- hardware counter comparison
- Performance Passport
- RecommendationBundle/cache/apply validation

Reason:

```text
No suitable existing small GGUF model was available on the T4g target, and the
phase policy prohibited downloading another model.
```

Outcome:

```text
insufficient_evidence
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
