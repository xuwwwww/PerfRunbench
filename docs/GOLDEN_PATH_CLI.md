# Golden-Path CLI

PerfRunbench's default command workflow is:

```bash
autotuneai run --auto --config autotune.yaml -- python train.py
```

The command after `--` is preserved as an argv list. PerfRunbench does not
concatenate it into a shell string and does not use `shell=True` for the
golden-path launcher.

Executable paths are preserved for launch. This matters for Python virtualenvs:
an argv entry such as `.venv/bin/python` remains the virtualenv executable in
RecommendationBundle launch specs instead of being resolved to the underlying
system interpreter.

## Commands

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai run -- python train.py
autotuneai tune --config autotune.yaml -- python train.py
autotuneai run --best --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

- `init` writes a public adapter config and refuses to overwrite an existing
  output unless `--force` is provided.
- `run --` is the plain monitored wrapper. It launches the command with the
  existing resource runner and does not consult the recommendation cache unless
  an explicit recommendation mode is requested.
- `tune` validates the workload identity, runs bounded candidates, performs
  paired confirmation, writes a Performance Passport, writes a
  RecommendationBundle when the statistical guard passes, and inserts the
  eligible bundle into the recommendation cache.
- `run --best` requires an eligible cache entry. It validates the bundle and
  environment fingerprint, runs a canary, launches the workload with the
  selected process-scoped environment, and rolls back the apply lease.
- `run --auto` performs cache lookup and policy evaluation before choosing
  reuse, canary, or retune.

## Config

Top-level structures are versioned with `schema_version: "1.0"`. JSON is the
mandatory interchange format. YAML config files are supported when PyYAML is
available; the public YAML examples are placeholders for user-owned commands.

Required sections:

- `workload`: adapter name, timeout, optional identity files.
- `objective`: metric name and direction.
- `metrics`: allowlisted metric source.
- `tuning`: preset, bounded candidate list, repetitions, noise band.
- `cache`: cache root, TTL, retune policy, canary policy.
- `apply`: canary and lease settings.

Supported metric sources:

- `json_file`
- `jsonl_file`
- `stdout_json_prefix`
- `adapter_result_file`

All metric sources must produce finite numeric values for the configured
objective metric before a candidate can be eligible.

## Training Adapter

`workload.adapter: pytorch-training` routes `tune`, cache-miss `run --auto`,
and `run --best` through the safe training adapter. The adapter requires a
bounded slice for tuning, records a slice launch spec for canary validation,
and records a separate full-run launch spec without the slice control for the
post-apply workload launch.

The adapter supports only allowlisted CLI, environment, boolean flag, and
JSON/YAML overlay bindings. It does not rewrite Python source files, execute
parser commands, or require PyTorch as a core dependency.

Real Arm validation evidence for the training adapter is recorded in
`docs/results/PHASE_14B_REAL_ARM_PYTORCH_VALIDATION.md`.

## Safety

Golden-path apply is process-scoped by default. Cache entries store
fingerprints, digests, bundle IDs, and redacted argv data. Credential-like env
keys in launch specs are rejected by the existing RecommendationBundle schema.

