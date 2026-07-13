# ArmTune Demo Script

This is a planned three-minute demo outline. It contains no measured Arm
result.

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

### 1:00-1:30 - Baseline

Show:

```text
autotuneai arm benchmark --config configs/armtune/llama_cpp_boottrace.json
```

Highlight raw artifacts, repeated measurements, and quality validation.

### 1:30-2:10 - Optimization

Show:

```text
autotuneai arm optimize --config configs/armtune/llama_cpp_boottrace.json
```

Highlight candidate search, paired baseline controls, failed trial visibility,
and recommendation rationale.

### 2:10-2:40 - Passport and Report

Show:

```text
autotuneai arm passport --run-id <run_id>
```

Open:

```text
.autotuneai/runs/<run_id>/armtune/report.html
```

Highlight reproducibility command, limitations, and raw artifact digests.

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
