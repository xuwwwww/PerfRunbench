# CPU-Aware Build Tuning

PerfRunbench treats CPU-specific compilation as a first-class tuning dimension
for Arm inference workloads. Runtime search alone is not enough when the
compiled binary controls backend selection, CPU feature use, and portability.

This document defines the Phase 19 build-profile model. It does not modify the
v0.2.0 release tag or reinterpret historical Phase 9 evidence.

## Default Flow

```text
inspect Arm target
-> lock a pinned runtime source commit
-> inspect compiler and CMake support
-> create allowlisted build candidates
-> capture build evidence
-> verify runtime backend evidence
-> compare build profiles with one fixed runtime protocol
-> run bounded runtime search on the selected build
-> confirm the complete configuration
-> generate a Performance Passport
-> guard recommendation reuse and apply
```

## Source Lock

Each CPU-aware build tuning run uses one exact upstream `llama.cpp` source
commit for every candidate. Historical `llama.cpp` commits remain historical
evidence and must not be mixed with Phase 19 results as the same protocol.

The source lock records:

- upstream repository identity
- tag when available
- full source commit
- source dirty state
- source checkout digest when captured
- CMake, compiler, and linker versions
- submodule state when applicable

## Build Candidates

Phase 19 supports at most six unique build candidates:

| Candidate | Profile ID | Role |
| --- | --- | --- |
| A | `upstream-default` | Reasonable upstream Release baseline with no PerfRunbench architecture override. |
| B | `portable-reference` | Portability reference when upstream supports multi-variant CPU builds. |
| C | `native-current-cpu` | `GGML_NATIVE=ON` with compile-command evidence. |
| D | `explicit-current-cpu` | Explicit CPU target only after compiler probe succeeds. |
| E | `native-kleidiai` | Native KleidiAI request; eligible only with build and runtime backend evidence. |
| F | `native-kleidiai-lto` | Optional KleidiAI + LTO experiment after LTO probe succeeds. |

Unsupported, duplicate, runtime-unverified, or failed candidates are preserved
as evidence but are not ranked as eligible recommendations.

## Allowlisted Build Model

Public configuration fields describe intent, not arbitrary shell commands.
PerfRunbench rejects untrusted compiler strings and unsupported CMake
definitions. Current allowlisted CPU build policies are:

```text
cpu_target_policy:
  portable
  upstream_default
  native_current_cpu
  explicit_cpu_target

backend_policy:
  default_cpu
  kleidiai
  openblas_prefill_only

optimization_policy:
  standard_release
  lto
```

Compiler flags must come from allowlisted profile fields and compiler probes.
`shell=True` is not required for profile validation or compiler probing.

## Build Evidence

Each build profile preserves:

- source commit and dirty state
- compiler, linker, CMake, and generator evidence
- requested CMake options
- effective CMake cache
- compile commands and effective CPU flags
- linked-library evidence
- runtime backend evidence
- binary SHA-256 and size
- build stdout/stderr summaries or captured artifacts
- build duration
- build status and eligibility state

KleidiAI can be `enabled_and_evidenced` only when build evidence and runtime
backend evidence both confirm it. A requested flag alone is reported as
`requested_but_unverified` or `requested_but_runtime_unverified`.

## Comparison Protocols

Build-profile comparison and runtime search are separate protocols.

The fixed build-profile protocol compares every verified build with the same:

- target
- model digest
- prompt and generation counts
- batch and ubatch
- thread count
- affinity policy
- cooldown
- benchmark binary interface
- repetition count

Primary objective:

```text
decode_tokens_per_second
```

Secondary objective:

```text
prefill_tokens_per_second
```

Prefill and decode are never combined into one score. Build-profile
percentage, runtime-search percentage, and complete-configuration percentage
must remain separately labeled.

## Claim Boundary

Allowed wording:

> Best observed eligible configuration under the tested CPU, model, runtime
> source commit, compiler, and measurement protocol.

Disallowed wording:

- global Arm speedup
- universal Neoverse speedup
- KleidiAI enabled solely from a requested CMake flag
- combined improvement across different protocols
- portability of a CPU-specific binary without native-binary guard evidence

