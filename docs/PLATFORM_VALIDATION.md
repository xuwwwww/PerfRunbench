# Portable Platform Validation

Run one bounded, non-installing validation:

```bash
autotuneai platform validate --preset quick --output validation.json
```

The report records sanitized OS, architecture, logical CPU count, memory,
NVIDIA GPU detection, package/source identity, and compatibility checks for
already-installed runtimes. Missing PyTorch, CUDA, or ONNX Runtime is reported
as `unavailable`; PerfRunbench never installs them during validation.

## Preset Budgets

| Preset | Hard wall-clock budget | Candidate cap | Confirmation |
| --- | ---: | ---: | --- |
| `quick` | 5 minutes | 3 | minimal |
| `standard` | 20 minutes | 8 | normal |
| `deep` | 60 minutes | 16 | stronger |

`--budget-minutes N` overrides the search wall-clock limit. Presets control
search cost only; correctness, cleanup, drift, and failure guards remain
unchanged.

## Evidence Boundary

`compatibility_verified` means a lightweight bounded smoke completed.
`performance_improvement_verified` requires repeated baseline/candidate
evidence that passes the existing statistical guard. A compatibility smoke
alone never creates a performance claim. Fixture results remain
`fixture_only`; missing runtimes are `unavailable`; execution or import errors
are `failed`.

## Collect Another Machine

Install the same PerfRunbench source commit or wheel, then run:

```bash
autotuneai platform validate --preset quick --output validation.json
```

Collect the resulting JSON file. It excludes usernames, hostnames, machine
identifiers, and absolute paths, but review it before public sharing as part of
normal evidence handling.
