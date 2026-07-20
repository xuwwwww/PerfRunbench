# Phase 3B.1 Real llama-server Validation Evidence

## Scope

This document records a bounded real-target validation of the Phase 3B.1
`llama-server` benchmark foundation on a user-provisioned AWS T4g / Graviton2
development instance.

This run validates server lifecycle, target-local loopback HTTP measurement,
OpenAI-compatible SSE parsing, generated-output preservation, cleanup, and
downloaded artifact digest validation against a real `llama-server` binary and
a real GGUF model on Linux Arm64.

This is not a formal competition benchmark result and must not be used as
evidence of final optimization, stable production performance, KleidiAI usage,
quality preservation, build-profile improvement, Performix support, or an
optimal ArmTune configuration.

## Public Evidence

```text
run_id: 20260719_004451_487119
target_type: ssh
target_family: AWS T4g / Graviton2
target_architecture: aarch64
target_os: Ubuntu 24.04.4 LTS
kernel: 6.17.0-1019-aws
cpu_model: Neoverse-N1
logical_cpus: 2
online_cpus: 0,1
PerfRunbench_commit: 1b70660812cee0da2ddf40eb1a0bc1af4a51d4c9
llama_cpp_commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama_server_version: version 10050 (b15ca938a)
llama_server_build: GNU 13.3.0 for Linux aarch64
llama_server_sha256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
model_sha256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
result_origin: measured
validation_status: passed
inspection_status: passed
startup_status: ready
readiness_status: ready
request_status: completed
stream_parse_status: parsed
benchmark_status: passed
cleanup_status: passed
port_released: true
quality_gate_status: not_evaluated
artifact_validation: passed
checked_artifacts: 48
failed_requests: 0
```

The public CLI command shape was:

```text
autotuneai arm benchmark --config <redacted-private-server-config>
```

The private target and server configuration file, public IP address, SSH key
path, identity-file contents, model file contents, and private absolute
configuration paths are intentionally omitted.

## Server Protocol

```text
adapter: llama_server
workload_mode: server
server_bind: 127.0.0.1
port: auto loopback port
concurrency: 1
prompt_count: 3
warmup_requests: 1 per prompt
measured_repetitions: 2 per prompt
total_warmup_requests: 3
total_measured_requests: 6
predict_tokens: 32
context_size: 512
threads: 2
batch_size: 128
seed: 42
temperature: 0.0
streaming: true
```

The inspected real `llama-server --help` output supported the required Phase
3B.1 flags:

```text
--model
--host
--port
--ctx-size
--threads
--batch-size
--ubatch-size
--seed
--temp / --temperature
--predict / --n-predict
--no-webui
```

ArmTune used only a bounded argument list and did not use `shell=True` for the
worker benchmark execution. The HTTP client ran on the target and connected
only to loopback.

## Latency Distributions

Measured requests only; warm-up requests are preserved as artifacts but
excluded from distributions.

TTFT is measured from request start to the first valid generated content
delta.

| metric | valid | failed | min ms | median ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ttft_ms | 6 | 0 | 40.645 | 41.105 | 45.24175 | 45.387 |
| end_to_end_latency_ms | 6 | 0 | 383.894 | 762.659 | 908.2335 | 910.594 |

These latency values are preliminary pipeline-validation measurements from a
burstable development instance. They are not stable production performance
numbers and are not an optimization result.

## Real SSE Compatibility

The real server emitted OpenAI-compatible SSE records that matched the Phase
3B.1 parser boundary:

- The first event was a role-only assistant metadata delta and was not counted
  as TTFT.
- Blank SSE separator lines were ignored.
- Generated content arrived through `choices[0].delta.content`.
- Runtime timing metadata was preserved as metadata and was not used as a
  substitute for client-observed TTFT or end-to-end latency.
- The stream ended with `data: [DONE]`.

Generated output was preserved for every warm-up and measured request. The
Phase 3B.1 run did not evaluate semantic quality; nonempty generated output is
not a quality pass.

## Lifecycle And Cleanup

The worker started `llama-server`, waited for the loopback health endpoint to
return HTTP 200, executed all configured requests, terminated the process, and
verified that the selected loopback port was released.

Independent post-run checks found no residual `llama-server` process and no
listener on the selected loopback port.

The raw server stderr contained a llama.cpp CORS warning because no API key was
configured. The server was bound only to `127.0.0.1`, no external port was
opened, and no cloud security-group change was made by ArmTune.

## Artifact Integrity

The controller downloaded the completed `server_benchmark` artifact directory
and independently re-ran the repository artifact-manifest validator. Validation
passed with 48 checked artifact records and no issues.

The artifact set includes:

```text
artifact_manifest.json
benchmark_result.json
executable_manifest.json
inspection.json
lifecycle.json
model_manifest.json
server_configuration.json
server_job.json
raw/inspection_version_stdout.txt
raw/inspection_version_stderr.txt
raw/inspection_help_stdout.txt
raw/inspection_help_stderr.txt
raw/server_stdout.txt
raw/server_stderr.txt
requests/<prompt-id>/<phase>/<index>/request.json
requests/<prompt-id>/<phase>/<index>/stream.jsonl
requests/<prompt-id>/<phase>/<index>/output.txt
requests/<prompt-id>/<phase>/<index>/result.json
```

## Claim Boundary

This evidence supports only the following claim:

> ArmTune's `llama-server` lifecycle, target-local loopback HTTP client,
> streaming TTFT and end-to-end latency measurement, generated-output
> preservation, cleanup, SSH worker, provenance, and artifact-validation
> pipeline was validated against a real `llama-server` workload on a Linux
> Arm64 target.

This evidence does not show:

- final ArmTune optimization
- stable production performance
- KleidiAI execution
- Arm optimized-path usage beyond host capability evidence
- deterministic quality-gate validation
- workload-specific semantic correctness
- build-profile comparison
- llama-server candidate search
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
