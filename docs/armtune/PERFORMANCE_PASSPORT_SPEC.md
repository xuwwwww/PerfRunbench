# Arm Performance Passport Specification

## Purpose

The Arm Performance Passport is the primary reproducibility artifact for an
ArmTune run. It must be readable and comparable without access to the original
machine.

Phase 5 MVP files:

```text
.autotuneai/runs/<run_id>/armtune/passport/performance_passport.json
.autotuneai/runs/<run_id>/armtune/passport/performance_passport.md
.autotuneai/runs/<run_id>/armtune/passport/performance_passport_verification.json
```

## Top-Level Requirements

The Phase 5 MVP passport includes:

- `schema_version`
- `run_id`
- `claim_eligibility_state`
- `claim_summary`
- `result_origin`
- `comparison_result_path`
- `comparison_result_sha256`
- `artifact_manifest_path`
- `artifact_manifest_sha256`
- `benchmark_claims`
- `provenance_summary`
- `limitations`
- `created_at_utc`

## Required Evidence Groups

### Repository

- git commit
- dirty status summary
- source package version

### Target

- target type
- OS and kernel
- architecture validation
- cgroup limits
- CPU identity
- CPU topology
- host capabilities

### Runtime and Build

- runtime adapter
- runtime version
- source commit when available
- binary paths sanitized for public summaries
- binary hashes
- build profile
- build evidence
- optimized-path evidence

### Workload

- model basename
- model SHA-256 when configured
- model file size
- model license metadata
- prompt-suite hash
- inference parameters

### Benchmark

- warmup runs
- search runs
- confirmation runs
- cooldown policy
- candidate order policy
- failed candidates
- raw artifact digests

### Results

- baseline distributions
- recommended distributions
- percentage deltas
- quality results
- Pareto candidates
- recommendation rationale

### Profiling

- Performix availability
- Performix version when used
- raw artifact references
- parsed summaries only when supported by raw evidence

## Compatibility and Migration

Passport readers should reject unsupported major schema versions and preserve
unknown fields when safe. Migrations should be explicit and tested.

## Public Summary Rules

Public summaries must not include:

- private absolute controller paths
- SSH private key paths when they reveal sensitive directory structure
- credentials
- model weights
- private logs

Public summaries must distinguish measured, synthetic, fixture, example, and
unverified data.

## Claim Eligibility States

The passport must use one of these claim states:

- `eligible_preliminary`
- `ineligible_quality`
- `ineligible_provenance`
- `ineligible_statistics`
- `ineligible_cleanup`
- `fixture_only`

`fixture_only` passports can demonstrate pipeline behavior, but they are not
measured Arm performance claims.

## Verification

`autotuneai arm verify-passport --passport <path>` checks:

- passport schema version
- supported claim eligibility state
- fixture result-origin consistency
- comparison-result digest
- artifact-manifest digest when present
- versioned benchmark-claim structure

Verification failure means the passport must not be used as a public
performance claim.
