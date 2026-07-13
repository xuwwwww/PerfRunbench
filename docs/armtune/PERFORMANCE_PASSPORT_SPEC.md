# Arm Performance Passport Specification

## Purpose

The Arm Performance Passport is the primary reproducibility artifact for an
ArmTune run. It must be readable and comparable without access to the original
machine.

Planned file:

```text
.autotuneai/runs/<run_id>/armtune/arm_performance_passport.json
```

## Top-Level Requirements

Every passport includes:

- `schema_version`
- `armtune_version`
- `perfrunbench_version`
- `repository`
- `target`
- `platform`
- `runtime`
- `model`
- `workload`
- `benchmark_protocol`
- `search_space`
- `baseline`
- `recommendation`
- `quality`
- `artifacts`
- `validity`
- `limitations`
- `reproducibility`
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
