# Phase 2C AWS Graviton2 Baseline Evidence

## Scope

This document records the first ArmTune preliminary measured baseline used to
validate the real SSH Arm64 execution pipeline on a burstable AWS T4g instance.

This is not a formal competition benchmark result and must not be used as
evidence of optimization, KleidiAI execution, build-profile improvement,
thread-search improvement, quality preservation, stable production
performance, or a generally optimal ArmTune configuration.

## Public Evidence

```text
run_id: 20260716_231915_147773
target_type: ssh
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1017-aws
cpu_model: Neoverse-N1
result_origin: measured
validation_status: passed
execution_status: completed
parse_status: parsed
benchmark_status: passed
artifact_validation: passed
checked_artifacts: 13
fake_executable_marker: false
prefill_throughput_tokens_per_second: 43.8
decode_throughput_tokens_per_second: 30.13
PerfRunbench_commit: 41b57b70b318a94977a5dd822314782e855295f6
llama_cpp_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama_bench_sha256: 990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8
model_sha256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
```

## Claim Boundary

This evidence supports only the following claim:

> ArmTune executed a real `llama-bench` binary with a real GGUF model over SSH
> on a Linux Arm64 target, preserved artifacts, parsed required throughput
> metrics, and validated downloaded artifact digests.

The measured values are preliminary pipeline-validation data from a burstable
cloud VM. They are not stable production performance numbers.

This evidence does not show:

- candidate search
- performance optimization
- KleidiAI usage
- build-profile comparison
- quality-gate validation
- llama-server behavior
- TTFT or streaming latency
- Arm Performix evidence
- Performance Passport generation
- formal competition benchmark results

## Privacy Boundary

This public evidence intentionally omits:

- public IP address
- SSH key path
- private controller configuration paths
- credentials or identity-file contents
- model file contents
- raw artifacts containing private absolute paths
