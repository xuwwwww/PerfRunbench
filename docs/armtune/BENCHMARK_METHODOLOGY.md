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

## Phase 2A llama-bench Method

Phase 2A runs one fixed llama-bench invocation from an existing binary and an
existing model file. It does not download models, build llama.cpp, search
candidates, run quality gates, use Arm Performix, or generate performance
passport claims.

The configured benchmark controls are:

- thread count
- prompt token count
- generation token count
- repetitions passed to llama-bench
- bounded process timeout

If repetitions are implemented through llama-bench itself, ArmTune records that
fact as a single invocation. ArmTune does not add warmup, search, or
confirmation loops in Phase 2A.

Before execution, ArmTune inspects the binary using `--version` and `--help`.
The adapter uses only supported command-line flags. If a required flag is not
present, the benchmark fails with a structured validation error rather than
silently changing the experiment.

The parser records explicit parse outcomes:

- `parsed`
- `partially_parsed`
- `unsupported_format`
- `malformed_output`
- `no_metrics`
- `ambiguous_output`

Missing metrics are reported as `Not available`, never as zero. A process that
exits successfully but prints no parseable required metrics is not a successful
benchmark result.

Duplicate or multiple rows for the same required metric are ambiguous. ArmTune
keeps the rows as audit evidence and fails the parse as `ambiguous_output`
rather than choosing the fastest or largest value.

Phase 2A does not claim KleidiAI, Neon, DotProd, I8MM, BF16, SVE, SVE2, SME, or
another optimized path. Optimized-path evidence requires later phases with
workload execution proof.

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

Real production validation remains a later phase. Phase 2C may run preliminary
real-Arm smoke validation only with a real target and user-supplied model.

## Phase 2B Native Arm64 CI Method

The GitHub-hosted Arm64 workflow runs fixture validation on a native Linux
Arm64 runner. It checks package installation, CLI availability, local
production-target readiness, host inspection serialization, focused ArmTune
tests, and a fake llama-bench public CLI E2E.

This CI smoke test is not an AI inference optimization result. It uses a fake
executable and a synthetic model placeholder, so its benchmark output remains
`result_origin: test_fixture` even on native Arm64 hardware.

Measured results require all of the following:

- real Linux Arm64 target
- real executable
- real model
- real benchmark execution
- no test-only host override
- no fake executable marker
- validated artifact manifest and digests

CI timing, workflow duration, fake fixture throughput, and parser fixture
values must not be used as competition benchmark data.

Real measured smoke execution is prepared for Phase 2C through an explicit
opt-in script that requires user-supplied binary and model paths. It must not
download models automatically.

## Phase 3A Repeated llama-bench Search Method

Phase 3A turns the single fixed benchmark invocation into a repeated
measurement protocol for performance-only search. It still uses
`llama-bench`, not `llama-server`, and it does not measure TTFT, streaming
latency, semantic quality, or build-profile differences.

Default controls are configurable:

```text
warmup_runs: 2
search_runs: 5
confirmation_runs: 10
cooldown_seconds: 2
random_seed: 42
insert_paired_baseline_every: 4
noise_band_percent: 2.0
```

Warm-up repetitions are preserved as artifacts but excluded from performance
distributions. Measured repetitions retain candidate ID, repetition index,
start/end UTC, raw stdout/stderr, execution status, parse status, parsed
metrics, host fingerprint, allowed CPU mask, target load when available, and
failure reason.

Failed trials are counted separately and are never aggregated as zero.
Distribution summaries include:

- valid count
- failed count
- median
- p95 when sample count supports it
- min
- max
- IQR or standard deviation

ArmTune emits a warning when sample count is too small for a meaningful
percentile.

Candidate generation is topology-aware and conservative. Thread candidates are
generated from visible and allowed CPUs, deduplicated, and validated. Optional
`llama-bench` flags are generated only when the inspected binary help output
proves support for those flags. Unsupported flags fail validation rather than
being used silently.

Phase 3B adds optional search-space controls for bounded validation profiles:

- `search.max_candidates` caps the total candidate plan, including the
  baseline.
- `search.affinity_modes` allowlists generated affinity candidates.
- Defaults preserve the Phase 3A candidate generation behavior when these
  fields are omitted.

The cap is applied in deterministic stage order. This keeps smoke runs small
without changing the benchmark operation, parser, artifact preservation,
statistics, or recommendation semantics.

The search is staged:

1. thread count
2. threads-batch and batch parameters
3. affinity
4. confirmation of baseline and selected candidate

Phase 3A recommendations are performance-only and include
`quality_gate_status: not_evaluated`. ArmTune must not call a candidate
generally optimal. The allowed wording is:

> Best observed performance candidate under the tested llama-bench
> configuration.

Candidates that failed required trials, missed required metrics, are within the
configured noise band, or differ in model, binary, target, token counts, or
protocol are not recommended as improvements.
