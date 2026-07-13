# Benchmark Methodology

## Integrity Rules

ArmTune must never invent benchmark numbers, test results, or optimized-path
evidence. Failed parser output must not become zero-valued successful metrics.

Results must be labeled as one of:

- `measured`
- `derived`
- `fixture`
- `synthetic`
- `example`
- `unverified`
- `not_available`

## Production Benchmark Requirements

Production benchmark, optimize, and profile operations require:

- Real Linux Arm64 target.
- Controlled model identity.
- Controlled prompt suite identity.
- Controlled token counts and sampling settings.
- Runtime and binary provenance.
- Raw stdout/stderr preservation.
- Repeated measurements.
- Explicit failed trial records.
- Quality validation when output quality matters.

Controller-side x86 measurements must never be reported as Arm results.

## Comparability Checks

Experiments are comparable only when these fields are compatible:

- model basename, size, hash, and quantization metadata when available
- prompt-suite hash
- runtime source commit and binary hashes
- benchmark protocol
- token counts
- context size
- CPU allocation and affinity strategy
- target CPU identity and topology
- quality threshold

Incompatible experiments should be rejected or labeled incomparable.

## Noise and Paired Baselines

Existing PerfRunbench logic already uses paired baselines and a default
`+/-2%` noise band for performance recommendation decisions. ArmTune should
reuse the same concept but record:

- repeated baseline distribution
- repeated candidate distribution
- median, p95, and spread
- paired normalized ratios when thermal or clock drift is plausible
- confirmation runs for baseline and recommendation

## Metrics

Minimum inference metrics:

- prompt processing throughput
- token generation throughput
- latency distribution
- time to first token when server mode is used
- peak RSS or cgroup memory when available
- failure reason and raw output path for failed trials

## Quality Validation

The BootTrace reference validator should be deterministic and should not require
network access or a paid model. It should check structured output properties
rather than exact natural-language text.

Synthetic and fixture quality cases must be labeled explicitly.

## Result Publication

Before public publication, every measured result should point to:

- run ID
- passport path
- raw artifact digest
- reproducibility command
- limitations
- whether Performix was used

No current repository benchmark result is an ArmTune production Arm result.
