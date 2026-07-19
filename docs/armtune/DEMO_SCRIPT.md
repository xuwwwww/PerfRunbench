# ArmTune Demo Script

This is a three-minute demo outline. Existing measured Arm runs are bounded
pipeline-validation evidence, not final competition performance results.

## Pre-Demo Requirements

- A real Linux Arm64 target is available locally or over SSH.
- llama.cpp binaries or source are present on the target.
- A compatible user-supplied GGUF model is present on the target.
- No model weights are committed to the repository.
- The target config does not contain credentials.

## Storyboard

### 0:00-0:30 - Product Setup

Show:

```text
autotuneai arm doctor --target-config configs/armtune/targets/demo.json
```

Explain that the controller can run on x86, but production benchmark
execution is validated on a real Linux Arm64 target.

### 0:30-1:00 - Arm Evidence

Show:

```text
autotuneai arm inspect-host --target-config configs/armtune/targets/demo.json
```

Highlight CPU identity, topology, host capabilities, and the label that host
capability is not the same as workload execution evidence.

### 1:00-1:30 - Quality Calibration

Show:

```text
autotuneai arm quality-calibrate --config configs/armtune/quality_calibration.example.json
```

Highlight strict raw-output validation, optional audited normalization,
failure-category histogram, prompt-contract digest, and model/profile
suitability.

### 1:30-2:10 - Build Profiles and Comparison

Show:

```text
autotuneai arm build-plan --config configs/armtune/build_profiles.example.json
autotuneai arm build-profile --config configs/armtune/build_profiles.example.json
autotuneai arm profile-compare --config configs/armtune/profile_comparison.example.json
```

Highlight that `build-plan` does not execute a build, `build-profile` records
existing build evidence, `arm-optimized` stays `requested_but_unverified`
without real KleidiAI evidence, and profile comparison rejects incomplete or
quality-failing profiles.

### 2:10-2:40 - Passport and Report

Show:

```text
autotuneai arm passport --comparison-result .autotuneai/runs/<run_id>/armtune/profile_comparison/profile_comparison_result.json
autotuneai arm verify-passport --passport .autotuneai/runs/<run_id>/armtune/passport/performance_passport.json
```

Open:

```text
.autotuneai/runs/<run_id>/armtune/passport/performance_passport.md
```

Highlight claim eligibility, limitations, result origin, comparison-result
digest, and artifact-manifest digest.

### 2:40-3:00 - CI Gate

Show:

```text
autotuneai arm ci-check --baseline passports/reference.json --candidate passports/current.json
```

Explain pass, regression, invalid/incomparable, and runtime-error exit codes.

## Demo Integrity Checklist

- Do not show fixture or synthetic values as measured results.
- Do not claim KleidiAI, SVE, I8MM, BF16, SME, or Performix usage without
  explicit evidence.
- Do not hide failed candidates.
- Do not compare incompatible model, prompt, runtime, or target configurations.
