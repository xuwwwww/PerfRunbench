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

Phase 3A.2 adds optional search-space controls for bounded validation
profiles:

- `search.max_candidates` caps unique tested candidate configurations,
  including the retained baseline.
- `search.affinity_modes` allowlists generated affinity candidates.
- Defaults preserve the Phase 3A candidate generation behavior when these
  fields are omitted.

The cap is applied in deterministic stage order and does not count warm-up
runs, paired-baseline insertions, confirmation runs, or total process execution
count. Stages that still had raw candidates but could not be evaluated because
the cap was exhausted are recorded as
`skipped_due_to_candidate_limit`.

Unknown affinity modes are configuration errors. Repeated affinity modes are
deduplicated. `numa-local` is generated only when topology makes it meaningful,
and `reserve-one-core` is not generated when only one visible CPU exists. CLI
overrides take precedence over config-file values, and the final effective
config is preserved with the run artifacts. This keeps smoke runs small without
changing the benchmark operation, parser, artifact preservation, statistics, or
quality-gate boundary.

The search is staged:

1. thread count
2. threads-batch and batch parameters
3. affinity
4. confirmation of baseline and selected candidate

Phase 3A recommendations are performance-only and include
`quality_gate_status: not_evaluated`. ArmTune must not call a candidate
generally optimal. The allowed wording is bounded by the actual tested
configuration count, for example:

> Best observed performance candidate among 4 tested configurations.

Candidates that failed required trials, missed required metrics, are within the
configured noise band, or differ in model, binary, target, token counts, or
protocol are not recommended as improvements.

Phase 3B is reserved for `llama-server` lifecycle validation, local-on-target
HTTP measurement, streaming time-to-first-token, end-to-end latency, structured
BootTrace output, deterministic quality validation, and combined performance
plus quality eligibility. Phase 3A.2 candidate controls are not Phase 3B
quality evidence.

## Phase 3B.1 llama-server Method

Phase 3B.1 uses `llama-server` to validate server lifecycle, target-local HTTP
request timing, streaming parsing, generated-output preservation, and cleanup.
It does not run candidate search or deterministic quality scoring.

The server is inspected before startup. ArmTune preserves raw `--version` and
`--help` stdout/stderr, records binary SHA-256 and size, extracts supported
flags from help output, and records unknown or unverified runtime/build fields
as `unknown` rather than guessing.

The server binds only to loopback. Public bind addresses are configuration
errors in Phase 3B.1. The HTTP client runs on the target and connects only to:

```text
http://127.0.0.1:<selected-port>
```

TTFT is defined as:

```text
first valid generated content timestamp - request start timestamp
```

The following do not count as first generated content:

- HTTP response headers
- empty SSE events
- role-only deltas
- metadata-only events
- keepalive events
- malformed events

End-to-end latency is defined as:

```text
response completion timestamp - request start timestamp
```

Runtime timing fields, when present, are preserved separately from
client-observed timings. ArmTune does not estimate unavailable token counts
from character count.

The default Phase 3B.1 request protocol is sequential:

```text
warmup_requests: 1 per prompt
measured repetitions: 3 per prompt
concurrency: unsupported
```

Warm-up request artifacts are preserved but excluded from measured latency
distributions. Failed requests remain failed request records; they are not
converted into zero latency.

The prompt suite is public and synthetic. Phase 3B.1 validates only basic
execution properties: request completed, generated content is nonempty,
streaming parsed, and no server error occurred. It records
`quality_gate_status: not_evaluated` and must not call nonempty output a
quality pass.

## Phase 4 Quality-Aware Server Search Method

Phase 4 reuses the Phase 3B.1 `benchmark.llama_server` worker path for each
candidate and adds controller-side deterministic quality validation. The
worker still receives only a versioned job spec path and a run directory, and
the target still executes only allowlisted worker operations.

The primary Phase 4 performance metric is median client-observed TTFT in
milliseconds. Lower is better. Median end-to-end latency is recorded as a
secondary metric, but ArmTune does not combine TTFT and end-to-end latency into
a composite score.

Candidate eligibility requires:

- `result_origin` is not `unverified`.
- binary and model validation passed.
- server startup and readiness passed.
- measured requests completed.
- SSE parsing passed.
- cleanup passed or was forced with port release.
- downloaded artifact digest validation passed.
- the quality suite digest, prompt-suite snapshot, and protocol settings are
preserved.
- all required deterministic quality cases pass.

A candidate with better TTFT but failed required quality validation is rejected
with `rejected_quality` and is not eligible for recommendation. A candidate
with execution, lifecycle, cleanup, provenance, or artifact-validation failure
is rejected with `rejected_execution`.

The recommendation text must remain bounded to the tested candidate count, for
example:

> Best observed quality-passing performance candidate among 3 tested
> configurations.

Phase 4 results are still not final competition benchmark conclusions and do
not prove KleidiAI or any optimized Arm execution path.

## Phase 5 Competition Evidence Method

Phase 5 treats the Phase 4 real `no_eligible_candidate` result as useful
evidence: the real target, server lifecycle, streaming parser, cleanup, and
artifact validation worked, while deterministic quality validation rejected
unsuitable generated output. Phase 5 must recover quality through an auditable
contract and evidence pipeline, not by weakening canonical expected answers.

### Quality Diagnostics

Every quality case records final failure categories, including:

- `response_format_violation`
- `markdown_fence_present`
- `invalid_json`
- `missing_required_field`
- `wrong_status`
- `wrong_failure_stage`
- `wrong_root_cause`
- `missing_required_evidence`
- `invented_evidence`
- `truncated_output`
- `request_failure`
- `stream_parse_failure`
- `unsupported_structured_output`
- `model_capability_failure`

These categories are diagnostic labels. They do not replace the existing
validator evidence, parser status, raw output, or normalized output.

### Output Normalization

Normalization is optional and versioned. The initial safe operations are:

- trim leading and trailing whitespace
- remove a UTF-8 BOM
- normalize line endings
- extract exactly one complete Markdown fenced JSON block when the policy
  permits it
- reject multiple fenced blocks
- reject prose outside a fenced block unless the policy explicitly permits it

Normalization must not repair malformed JSON, add missing fields, change
values, infer evidence IDs, change semantic classification, or replace invalid
answers with canonical answers.

Quality artifacts must record:

- normalization policy version
- whether normalization was applied
- normalization steps
- raw output SHA-256
- normalized output SHA-256 when applicable
- whether validation used raw or normalized output

Strict mode disables normalization and validates only raw output.

### Structured Generation

Structured generation options are used only when support is evidenced by the
inspected binary help output or by a bounded probe. Capability states are:

- `supported`
- `unsupported`
- `unverified`
- `probe_failed`

ArmTune must not rely on source documentation alone. It must not accept
arbitrary grammar text. Any schema or grammar used for BootTrace output is
derived from the validated public quality-suite schema and recorded with a
digest.

### Quality Calibration

`autotuneai arm quality-calibrate --config <path>` runs one fixed
runtime/profile against the quality suite before performance search. It
reports suite case count, request success count, raw JSON validity, normalized
JSON validity, required-case pass count, critical failure count,
failure-category histogram, `quality_gate_status`, and model/profile
suitability.

Suitability states are:

- `suitable`
- `unsuitable`
- `inconclusive`
- `execution_failed`

Output normalization alone does not make a model/profile suitable. The
semantic validators must still pass.

### Build Profiles

Build profiles are evidence records, not assumptions. The initial profile IDs
are:

- `portable-baseline`
- `arm-native`
- `arm-optimized`

Each profile records source commit, dirty-state status, compiler and CMake
versions, requested and effective CMake definitions, target architecture,
binary digests, binary size, linked-library evidence, build-output evidence
when available, build status, and evidence completeness.

`arm-optimized` may report KleidiAI only with build and runtime evidence.
Requested build flags or CMake cache entries without compile, link, runtime,
or binary evidence remain
`requested_but_unverified`.

### Profile Comparison

Profile comparison runs only after quality calibration. Eligible profiles must
share the same model, prompt suite, normalization policy, generation settings,
target, and CPU allocation.

Primary metric:

```text
median client-observed TTFT
```

Secondary metrics include median end-to-end latency, p95 TTFT, p95
end-to-end latency, runtime-reported timing when available, failure rate, and
variability. Metrics are not combined into an undocumented score, and failed
trials are never aggregated as zero.

Recommendation states are:

- `recommended`
- `inconclusive`
- `rejected_quality`
- `rejected_build_evidence`
- `rejected_execution`
- `insufficient_evidence`
- `no_eligible_profile`

Recommended wording must remain bounded:

> Best observed eligible build profile under the tested Arm target, model,
> server workload, and deterministic quality protocol.

### Performance Passport

The Performance Passport binds evidence into:

```text
performance_passport.json
performance_passport.md
```

The passport includes repository commit, target fingerprint, source commit,
build evidence, compiler and CMake versions, binary/model digests, quality
suite, prompt, and normalization digests, benchmark protocol, profile results,
quality results, rejected results, selected result, statistical distributions,
noise-band decision, artifact-manifest digest, claim eligibility, limitations,
and redacted reproducibility commands.

Claim eligibility states are:

- `eligible_preliminary`
- `ineligible_quality`
- `ineligible_provenance`
- `ineligible_statistics`
- `ineligible_cleanup`
- `fixture_only`

Passport verification checks schema, internal digest references, artifact
manifest binding, and claim consistency. It must never upgrade fixture,
quality-failing, provenance-incomplete, statistics-incomplete, or cleanup-
incomplete evidence into a public performance claim.

Phase 5 local tests cover strict-output failure, fenced-JSON normalization,
fixture build-profile evidence, optimized-profile evidence rejection, fixture
profile comparison, Performance Passport generation, and passport tamper
detection. These tests validate the evidence pipeline and claim gates; they do
not provide measured Arm performance.
