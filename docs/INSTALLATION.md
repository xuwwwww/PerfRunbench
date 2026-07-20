# Installation

PerfRunbench currently supports Python 3.10 or newer.

## Source Install

```bash
python -m pip install .
autotuneai --help
```

## Editable Development Install

```bash
python -m pip install -e .
python scripts/run_tests.py --fast
```

## Local Wheel Install

PerfRunbench is not published to PyPI in this repository state. Build and
install a local wheel:

```bash
python -m pip install build
python -m build
python -m pip install dist/perfrunbench-<version>-py3-none-any.whl
```

Validate the installed console script:

```bash
autotuneai --help
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
```

## Dependencies

Base dependency:

- `psutil`

Base installation does not install PyTorch, CUDA, ONNX, model files, datasets,
SSH keys, or cloud tooling.

Optional benchmark/development extras:

```bash
python -m pip install -e ".[benchmark]"
```

The optional `benchmark` extra may install large ML packages such as PyTorch,
TorchVision, ONNX, and ONNX Runtime. Install it only in an environment where
those packages are needed.

## Console Script

The package installs:

```text
autotuneai = autotune.cli:main
```

Use `autotuneai --help` to confirm the installed command is visible on `PATH`.

## Fresh-Install Validation

The release smoke script validates a built wheel in a fresh virtual
environment:

```bash
python scripts/release_smoke.py --wheel dist/perfrunbench-<version>-py3-none-any.whl
```

It uses only public synthetic fixtures. It does not use AWS, PyTorch, models,
datasets, private configs, or network access after the wheel is installed.
