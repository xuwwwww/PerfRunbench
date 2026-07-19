# Remote Targets

## Supported Initial Targets

ArmTune should initially support:

- `local`: production benchmark only when the local machine is Linux Arm64.
- `ssh`: remote Linux Arm64 target over the system OpenSSH client.

Controller-only operations such as config validation, report generation, and
fixture-based tests may run on x86_64.

## Local Target Rules

`LocalArmTarget` must:

- Accept production benchmark execution only on Linux Arm64.
- Reject Windows, macOS, WSL x86_64, and Linux x86_64 production benchmark
  execution.
- Permit fixture and report operations on the controller.
- Preserve the distinction between controller inspection and production target
  inspection.

## SSH Target Rules

`SshArmTarget` must:

- Use the system OpenSSH client for the MVP.
- Use strict host-key verification by default.
- Support host, user, port, identity file, SSH executable, SCP executable,
  remote Python executable, remote Python module, remote work directory,
  connect timeout, command timeout, and remote artifact retention.
- Never log private key contents.
- Never store credentials in run artifacts.
- Preserve stdout, stderr, exit status, and timeout information.
- Validate that the target identifies as Linux Arm64 before production
  benchmark commands run.
- Upload job specs and invoke a fixed worker command rather than composing
  large arbitrary shell scripts.
- Never use `shell=True`.
- Never accept arbitrary remote worker command strings.
- Not add Paramiko unless the system OpenSSH approach has a documented blocker.
- Clean temporary directories when safe, while allowing configured artifact
  retention for debugging.

## Versioned Worker Protocol

Controller flow:

1. Validate configuration.
2. Create a versioned JSON job specification.
3. Upload only required files.
4. Invoke a fixed remote ArmTune worker command.
5. Download versioned result manifests and artifacts.
6. Validate schema versions and artifact digests.

Remote worker flow:

1. Receive only a job-spec path and run directory.
2. Validate every field in the job spec.
3. Execute an allowlisted operation.
4. Write versioned structured results.
5. Preserve raw stdout and stderr.
6. Perform cleanup.

The protocol boundary is the only place where SSH command-line quoting should
be needed. The remote command shape is fixed; user-provided benchmark commands
must be represented inside the uploaded JSON job spec and validated by the
worker before execution.

## Remote Prerequisite Checks

Before production benchmark execution, the controller must check:

- Python executable availability through the configured `remote_python`.
- Worker module import availability through the configured `remote_module`.
- Worker protocol compatibility through a versioned `check_prerequisites`
  operation.
- Linux operating system.
- Arm64 architecture.
- Writable remote work directory.

The MVP may require PerfRunbench to be installed manually on the remote target.
ArmTune must not perform automatic remote installation or cloud provisioning.

## Model Handling

The MVP should not upload a model unless explicitly configured. When the model
already exists on the remote target, ArmTune should validate basename, size, and
expected SHA-256 when provided.

Model weights must not be committed or copied into run artifacts.

## Worker Entry Point

The internal worker command should be documented as internal:

```text
python3 -m autotune.cli arm worker --job-spec <remote-json-path> --run-dir <remote-run-dir>
```

The normal user workflow should remain controller-side commands such as:

```text
autotuneai arm doctor --target-config <path>
autotuneai arm inspect-host --target-config <path>
```

## Timeouts and Cleanup

All target operations should use bounded timeouts. Child processes must be
terminated and reaped on success, failure, timeout, SIGINT, and SIGTERM.

Temporary tuning must be restored after benchmark failure.

## Phase 2B Artifact Trust Boundary

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

The controller must not trust absolute paths supplied by a remote result JSON
when deciding which local files to validate. Artifact records should contain
run-directory-relative paths and SHA-256 digests.

Failed artifact validation prevents a successful benchmark status while
preserving the downloaded files for debugging.

The native Arm64 GitHub workflow does not exercise an external SSH server.
Remote-target production validation with a real SSH host remains a later
phase.

## Phase 3B.1 Server Benchmark Protocol

Phase 3B.1 adds the allowlisted worker operation:

```text
benchmark.llama_server
```

The controller uploads a versioned `server_job.json` and invokes the same fixed
worker entry point:

```text
python3 -m autotune.cli arm worker --job-spec <remote-json-path> --run-dir <remote-run-dir>
```

The job spec must describe executable, model, server, workload, and artifact
settings. It must not contain arbitrary remote shell strings or user-controlled
extra server flags.

The remote worker must bind `llama-server` only to loopback:

```text
127.0.0.1
```

Phase 3B.1 rejects public bind addresses such as `0.0.0.0`; it does not open
EC2 security-group ports and does not expose the server to external networks.

The HTTP benchmark client runs on the target and connects only to:

```text
http://127.0.0.1:<selected-port>
```

The controller downloads the completed `server_benchmark` artifact directory
and validates `artifact_manifest.json` using the same digest trust boundary as
the `llama-bench` benchmark path.

## Phase 4 Server Optimize Protocol

Phase 4 does not add a new remote worker operation for search. The controller
performs candidate planning locally and invokes repeated versioned
`benchmark.llama_server` jobs for each candidate.

The candidate-specific fields are serialized into the server job spec:

- `server.threads`
- `server.batch_size`
- `server.ubatch_size`
- `server.affinity_mode`
- `server.cpu_affinity`

The remote worker validates every field through the existing server benchmark
protocol allowlist. It still does not accept arbitrary remote shell strings,
free-form server flags, public bind addresses, or `shell=True` execution.

For SSH targets, `workload.prompt_suite_path` must be an absolute path on the
target. The controller writes a local derived `prompt_suite_snapshot.json` for
auditability, but the operator must upload the same prompt-suite payload to the
configured target path before a bounded real Phase 4 validation run.

The controller downloads each candidate's completed server benchmark artifact
directory, validates candidate artifact manifests, writes the
`server_optimize_result.json`, and validates the root `server_optimize`
artifact manifest. When an optional experimental quality gate is explicitly
enabled, quality evidence is included in that same result without changing the
fixed worker protocol. Credentials, identity-file paths, and private host
details must not be written to public artifacts.

## Phase 5 Evidence Commands on SSH Targets

Phase 5 does not loosen the remote execution boundary.

`autotuneai arm quality-calibrate --config <path>` still invokes the existing
allowlisted `benchmark.llama_server` worker operation for workload execution.
The controller adds local evidence artifacts around that run: quality-suite
manifest, quality-suite snapshot, normalization policy, prompt contract,
structured capability summary, quality calibration result, artifact manifest,
and artifact validation.

For SSH targets, the input config still carries an absolute
`workload.prompt_suite_path` for schema compatibility, but Phase 7
`quality-calibrate` renders the selected BootTrace prompt contract on the
controller, writes a local `prompt_suite_snapshot.json`, uploads that exact
versioned prompt suite into the remote run directory, and substitutes only that
remote prompt-suite path in the allowlisted worker job. It does not upload
models, credentials, arbitrary workload files, or private shell commands.

The downloaded request artifacts preserve the final JSON request body sent to
`llama-server`, so reviewers can verify that the selected prompt contract,
seed, temperature, and output token bound were applied.

`autotuneai arm build-plan` does not execute remote work. `build-profile` and
`inspect-build` record configured source and binary evidence and must not
accept arbitrary remote shell strings. A profile may be compared only after
quality calibration and artifact digest validation pass.

`autotuneai arm passport` and `verify-passport` are controller-side evidence
operations. They do not connect to the target and do not modify cloud
resources.
