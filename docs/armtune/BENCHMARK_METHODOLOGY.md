# ArmTune Benchmark Methodology

## Scope

ArmTune measures AI inference performance on real Linux Arm64 targets. The
default methodology covers:

```text
inspect Arm target
-> inspect workload runtime
-> benchmark a portable baseline
-> build Arm profiles
-> search bounded runtime candidates
-> confirm the selected candidate
-> generate and verify a Performance Passport
-> validate a guarded recommendation
```

`llama.cpp` is the first workload adapter. The product claim is not tied to one
model or one benchmark binary.

## Evidence Boundary

Production Arm performance evidence requires:

- Linux Arm64 target provenance
- host identity, topology, and environment evidence
- real executable and real model file digests
- raw stdout/stderr preservation
- parser status separate from execution status
- artifact manifests with SHA-256 digests
- independent artifact validation after download
- cleanup and port-release evidence when a server process is used

Fixture, mocked, synthetic, CI-runner, or fake-executable outputs must be
labeled `test_fixture` and must not be promoted to measured Arm performance.

## Comparability

Profile and runtime comparisons are valid only when the following remain
compatible:

- target fingerprint
- model digest
- executable/source commit identity
- prompt and generation counts
- CPU allocation and affinity policy
- workload adapter and protocol
- repetitions, warm-up handling, cooldown, and timeout settings
- parser and artifact schema versions

Results from different protocols must not be combined into one aggregate
speedup.

## Metrics

`llama-bench` primary metrics:

- prefill throughput
- decode throughput
- median
- p95
- min/max
- variability
- failed trials

`llama-server` secondary metrics:

- request completion
- nonempty output
- stream parsing
- TTFT
- end-to-end latency
- cleanup and port release

## Search Discipline

Runtime search uses:

- warm-up exclusion
- deterministic randomized order
- paired baseline insertion
- repeated measured runs
- confirmation runs separate from search samples
- cooldown
- noise-band classification
- bounded candidate counts rather than full Cartesian products

Candidate eligibility requires successful execution and parsing, valid build
evidence, matching target/model/protocol digests, cleanup, sufficient samples,
and a difference outside the configured noise band.

## Build Evidence

Every build profile should preserve:

- source commit and dirty state
- compiler and CMake versions
- exact build command when available
- effective CMake cache
- compile flags
- linked-library evidence
- binary SHA-256 and size
- raw build stdout/stderr when an executing build flow supplies them

Optimized-path claims such as KleidiAI require evidence beyond a requested
flag. Unsupported or unverified paths must be recorded as such.

## Performance Passport

The Performance Passport binds the measured result to the target, model,
runtime, protocol, build evidence, statistical distributions, artifact
digests, eligibility state, and limitations. Passport verification must fail
on schema, digest, artifact, or claim-consistency problems.

## External Workload Correctness

PerfRunbench keeps workload-neutral correctness hooks and guarded
recommendation contracts. Workload-specific semantic evaluation projects, such
as firmware-log capability tests, should run externally and exchange evidence
through JSON or a generic command workload. They are not part of default
ArmTune performance recommendation eligibility.

The `v0.1.0-armtune` tag preserves the earlier integrated experimental state.
