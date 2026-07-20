# Release Checklist

This checklist records the v0.2 release-candidate validation boundary.

## Source Audit

- source branch: `release/v0.2-usability`
- source base requested by Phase 16: `origin/main`
- observed `origin/main` at Phase 16 start:
  `97380574c4271da3ae160b2a3cf95a62133ef674`
- prerequisite validation commits integrated because they were still present
  only on `origin/validation/phase15c-windows-x86`:
  - `40aee3c docs: record real T4g PyTorch validation`
  - `5bf2a0d fix: normalize hex CPU affinity masks`
  - `72bd656 docs: record Windows x86 PyTorch validation`
- package version before release-candidate validation: `0.1.0`
- target package version after validation: `0.2.0`
- Python support metadata: `>=3.10`
- console script: `autotuneai = autotune.cli:main`
- base dependency: `psutil`
- optional extras: `benchmark`, `dev`
- package data policy: Python packages under `autotune*`; public docs,
  tests, models, datasets, private artifacts, and `.autotuneai/private` are
  not packaged in the wheel.
- README metadata: `README.md`
- license metadata: not declared in this repository state

## Required Local Validation

```bash
python -m unittest tests.test_pytorch_training_adapter -v
python -m unittest tests.test_golden_path_cli tests.test_golden_path_doctor tests.test_recommendation_safety tests.test_workload_runner -v
python -m unittest discover -s tests -p "test_arm_tune*.py" -v
python scripts/run_tests.py --fast
python -m compileall autotune
git diff --check
```

## Build Validation

```bash
python -m build
python -m twine check dist/*
python scripts/release_smoke.py --wheel dist/perfrunbench-0.2.0-py3-none-any.whl
```

Record:

- wheel filename and SHA-256:
  `dist-v0.2/perfrunbench-0.2.0-py3-none-any.whl`
  `034963a60008a96f33ed5763b2481f7caf757bdeb3a06e7e3e05b4858de8205b`
- sdist filename and SHA-256:
  `dist-v0.2/perfrunbench-0.2.0.tar.gz`
  `bc59f63e7dfb35a9b4a86f3222a56cf921b813dec9af8ba445e90d8fe59fcd2e`
- fresh Windows install result: passed
- optional PyTorch smoke result: skipped; no additional PyTorch wheel was
  downloaded solely for release-candidate validation

## Fresh-User Smoke Coverage

The release smoke must run in a temporary directory outside the source
worktree and validate:

- installed console script
- no import from source checkout
- `init`
- `doctor`
- `tune`
- `run --auto` cache miss
- `run --auto` cache hit
- `run --best`
- cache list/show/invalidate/gc
- command arguments with spaces
- Unicode command argument
- working directory with spaces
- JSON output
- actionable missing-cache error
- ArmTune fake benchmark `result_origin: test_fixture`
- no active lease after completion

## CI Matrix

Required installed-wheel matrix:

- Windows x86_64: `windows-latest`
- Linux x86_64: `ubuntu-latest`
- Linux Arm64: `ubuntu-24.04-arm`

Each required job builds the distribution, installs the wheel in a clean
environment, runs release smoke, runs focused unit tests, and runs
`compileall`.

No required CI job installs PyTorch, downloads models/datasets, accesses AWS,
or caches recommendation/user artifacts between runs.

## Safety Audit

Scan tracked files and built distributions for:

- credentials
- tokens
- AWS addresses
- PEM paths
- user-specific absolute paths
- private artifact paths
- model files
- datasets
- RecommendationBundles containing private target data
- BootTrace-specific production code
- unsafe shell execution examples

Generated `dist/`, `build/`, `.egg-info`, `.autotuneai/`, and temporary venvs
must not be committed.

## Release Boundary

Do not create a tag, publish to PyPI, provision cloud resources, add new
models, add benchmark backends, or restart BootTrace prompt/model research for
this release candidate.
