# ArmTune Architecture

## Product Identity

ArmTune is an Arm-first AI inference optimization subsystem inside
PerfRunbench. It is not generic Arm compatibility, a generic benchmark wrapper,
or a dashboard around llama.cpp.

## Controller and Target

ArmTune splits orchestration from measurement:

- Controller: user-facing CLI, configuration validation, search orchestration,
  comparison, passport/report generation, and fixture-based tests.
- Target: real Linux Arm64 host inspection, build inspection, inference
  execution, CPU placement, process monitoring, optional profiling, and raw
  artifact generation.

The controller may run on x86_64 or Arm64. Production benchmarks must run on a
real Linux Arm64 target.

## Target Protocol

The target abstraction should conceptually support:

```text
inspect()
prepare_run()
upload_file()
upload_directory()
execute()
start_process()
terminate_process()
download_file()
download_directory()
collect_artifacts()
cleanup()
```

The remote protocol should avoid arbitrary shell command construction:

Controller responsibilities:

1. Validate configuration.
2. Create a versioned JSON job specification.
3. Upload only required files.
4. Invoke a fixed remote ArmTune worker command.
5. Download versioned result manifests and artifacts.
6. Validate schemas and artifact digests.

Remote worker responsibilities:

1. Receive only a job-spec path and run directory.
2. Validate every field.
3. Execute an allowlisted operation.
4. Write versioned structured results.
5. Preserve raw stdout/stderr.
6. Perform cleanup.

The initial SSH implementation must use the system OpenSSH client with strict
host-key checking by default. It must not use `shell=True`, must not be built
around arbitrary remote shell strings, and must not add Paramiko unless the
system OpenSSH approach has a documented blocker.

The SSH worker command is structured configuration, not a free-form remote
command string. The MVP fields are:

- `remote_python`
- `remote_module`
- `remote_workdir`

The fixed worker invocation shape is:

```text
python3 -m autotune.cli arm worker --job-spec <path> --run-dir <path>
```

The controller may configure a different Python executable or module path, but
it must still invoke the same argument shape. ArmTune must not accept an
arbitrary `remote_worker_command` field.

Before production benchmark commands, the SSH target must verify:

- Python executable availability.
- Module import availability.
- Worker protocol compatibility.
- Linux operating system.
- Arm64 architecture.
- Writable remote work directory.

## Package Layout

Initial target:

```text
autotune/
  arm_tune/
    __init__.py
    cli.py
    config.py
    schemas.py
    errors.py
    targets/
      __init__.py
      base.py
      local.py
      ssh.py
      worker.py
      protocol.py
    platform/
      __init__.py
      architecture.py
      cpu_identity.py
      cpu_features.py
      topology.py
      cloud_platform.py
      environment.py
      evidence.py
```

Later packages:

```text
builds/
adapters/
benchmark/
search/
quality/
profiling/
passport/
ci/
report/
```

## PerfRunbench Reuse

Reuse existing PerfRunbench infrastructure where sound:

- `.autotuneai/runs/<run_id>/` lifecycle and provenance.
- Resource budgets and CPU affinity primitives.
- Systemd, Docker, and local executor capability ideas.
- Signal handling and cleanup patterns.
- Paired baseline and noise-band decision logic.
- Generic JSON and HTML report helpers.

The concrete symbol-level reuse map is maintained in `REUSE_MAP.md`.

ArmTune-specific artifacts live under:

```text
.autotuneai/runs/<run_id>/armtune/
```

## CLI Shape

The top-level CLI remains `autotuneai`. ArmTune adds one isolated group:

```text
autotuneai arm doctor
autotuneai arm inspect-host
autotuneai arm inspect-build
autotuneai arm benchmark
autotuneai arm optimize
autotuneai arm profile
autotuneai arm passport
autotuneai arm compare
autotuneai arm ci-check
autotuneai arm demo
```

`autotune/cli.py` should only receive a thin dispatcher for the `arm` group.

## Doctor Status Semantics

`autotuneai arm doctor` reports diagnostics, not benchmark success. It may
return exit code 0 when diagnostics complete successfully even if no production
Arm target is ready.

Doctor output separates:

- `controller_status`: whether the controller-side diagnostics completed.
- `local_target_status`: whether the controller machine itself is a supported
  local production target.
- `configured_target_status`: whether a supplied target configuration is ready.
- `production_benchmark_ready`: whether a configured target is ready for real
  Linux Arm64 production benchmark execution.

A Windows or x86_64 controller can have `controller_status: ready` while
`local_target_status: unsupported` and `production_benchmark_ready: false`.
Controller measurements must not be represented as Arm benchmark results.

## Serialization

Every serialized top-level ArmTune artifact includes:

- `schema_version`
- UTC timestamps
- stable enum values
- provenance sufficient to audit parsed metrics

Unknown future fields may be preserved where safe. Unknown critical
configuration fields should be rejected with actionable errors.
