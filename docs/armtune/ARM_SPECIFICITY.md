# Arm Specificity

## Why ArmTune Is Arm-Specific

ArmTune is specific to Arm because it validates and optimizes AI inference on
real Linux Arm64 targets, with evidence tied to Arm CPU identity, topology,
features, build choices, runtime behavior, and profiling artifacts.

Generic benchmark wrappers can measure a command. ArmTune must answer whether
an Arm-specific execution path was available, whether it was actually used by
the workload, and whether any measured speedup is reproducible and larger than
noise.

## Features With Arm-Specific Meaning

These concepts have Arm-specific meaning and should not be translated into
generic x86 labels:

- Arm architecture level and Linux `aarch64` validation.
- Neon and Advanced SIMD capability.
- DotProd, I8MM, BF16, SVE, SVE2, SME, and SME2 host features.
- Neoverse CPU identity and topology.
- Arm-specific CPU placement and NUMA topology interpretation.
- KleidiAI or other Arm CPU micro-kernel evidence.
- Arm Performix profiling evidence.

## Capability Detection

Host capability detection may use:

- `uname -m`
- Python `platform.machine()`
- `/proc/cpuinfo`
- `lscpu --json`
- OS, kernel, cgroup, and NUMA metadata

Capability detection proves only what the host exposes. It does not prove that
the inference workload used an optimized path.

## Workload Execution Evidence

Workload execution evidence must be recorded separately from host capability.
Examples:

- Runtime version output.
- Build flags and compile definitions.
- Binary hashes.
- Runtime logs indicating selected backend or kernels.
- Parsed `llama-bench` rows.
- Raw stdout/stderr artifacts.
- Optional profiling artifacts.

If the runtime does not provide evidence for an optimized path, the result must
be labeled `unknown` or `unverified`.

## Topology-Aware Search

ArmTune search should use topology only after target inspection confirms it.
Potential strategies:

- scheduler default
- compact placement
- spread placement
- reserve-one-core placement
- NUMA-aware candidate subsets when NUMA is present

Any topology decision must be recorded in the candidate config and passport.

## Build Profile Comparison

Build profiles should be comparable only when the model, prompt suite, token
counts, runtime source commit, binary provenance, CPU allocation, and quality
thresholds are controlled.

Phase 5 profile IDs:

- `portable-baseline`
- `arm-native`
- `arm-optimized`

KleidiAI or another optimized path must not be labeled enabled without
explicit compile, link, runtime, or binary evidence. Host feature support and
requested CMake settings are necessary context but not workload execution
evidence.

## Claims Requiring A Real Arm Host

These claims require Phase 6 or another real Linux Arm64 target run:

- Arm throughput or latency results.
- Arm optimized path used by the workload.
- KleidiAI enabled in the measured workload.
- Neoverse-specific performance conclusions.
- Performix-derived bottleneck conclusions.
- Regression gate thresholds backed by Arm measurements.

Phase 5 fixture passports are useful for validating the evidence pipeline, but
they are not Arm-specific performance claims unless backed by a real Linux
Arm64 target run with `result_origin: measured`.
