# Remote Targets

## Supported Targets

ArmTune supports:

- `local`: production benchmark execution only when the local machine is Linux
  Arm64.
- `ssh`: remote Linux Arm64 target over the system OpenSSH client.

Controller-only operations such as configuration validation, report
generation, Passport verification, and fixture-based tests may run on x86_64.

## SSH Target Rules

`SshArmTarget` must:

- use the system OpenSSH client for the MVP
- use strict host-key verification by default
- never log private key contents
- never store credentials in run artifacts
- validate Linux Arm64 target identity before production benchmark commands
- upload versioned JSON job specifications
- invoke a fixed worker entry point
- download versioned result manifests and artifacts
- validate schema versions and artifact digests after download
- avoid `shell=True`
- avoid arbitrary remote shell strings

Supported target configuration fields include host, user, port, identity file,
SSH executable, SCP executable, remote Python executable, remote module, remote
work directory, timeouts, and remote artifact retention.

## Versioned Worker Protocol

Controller flow:

1. Validate configuration.
2. Create a versioned JSON job specification.
3. Upload only required files.
4. Invoke a fixed remote ArmTune worker command.
5. Download result manifests and artifacts.
6. Validate schema versions and SHA-256 digests.

Remote worker flow:

1. Receive only a job-spec path and a run directory.
2. Validate every field in the job spec.
3. Execute an allowlisted operation.
4. Write versioned structured results.
5. Preserve raw stdout and stderr.
6. Perform cleanup.

The internal worker command shape is fixed:

```text
python3 -m autotune.cli arm worker --job-spec <remote-json-path> --run-dir <remote-run-dir>
```

User-provided benchmark parameters are represented inside the uploaded JSON job
spec and validated by the worker. They are not interpolated into arbitrary
remote shell strings.

## Allowlisted Operations

Current operations include:

- target prerequisite checks
- host inspection
- `benchmark.llama_bench`
- `benchmark.llama_server`
- build-profile inspection and evidence collection
- artifact manifest validation

Search orchestration remains controller-side. The controller creates candidate
plans and invokes repeated versioned worker jobs rather than sending a remote
script.

## Model Handling

The MVP does not upload model weights by default. When a model already exists
on the target, ArmTune validates configured basename, size, and SHA-256 when
those constraints are supplied.

Model weights, credentials, private target configs, endpoint addresses, and
identity-file paths must not be committed or copied into public artifacts.

## Server Benchmark Rules

`llama-server` benchmarks bind only to loopback:

```text
127.0.0.1
```

The target-local HTTP client connects only to:

```text
http://127.0.0.1:<selected-port>
```

Public bind addresses are configuration errors. ArmTune does not open cloud
firewall ports or modify security groups.

The server protocol records request completion, nonempty output, stream
parsing, TTFT, end-to-end latency, cleanup, and port release. Workload-specific
semantic validation can be supplied as an optional external correctness
artifact, but it is not part of the default remote worker contract.

## Artifact Trust Boundary

Remote result JSON is not trusted merely because the worker returned it. After
download, the controller validates `artifact_manifest.json` against files under
the local run directory.

The controller rejects:

- digest mismatches
- missing artifacts
- duplicate manifest relative paths
- path traversal attempts
- manifest paths outside the run directory
- symlink escapes
- unsupported artifact manifest schema versions

Failed artifact validation prevents a successful benchmark status while
preserving downloaded files for debugging.

## Recommendation Operations

`autotuneai arm passport` and `verify-passport` are controller-side evidence
operations. RecommendationBundle validation and process-scoped guarded apply
reuse PerfRunbench's generic recommendation safety model. They do not provision
cloud resources, change AWS state, or require workload-specific semantic
research.
