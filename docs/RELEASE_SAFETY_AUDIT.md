# Release Safety Audit

Phase 13 public safety audit scope:

- tracked source files
- tracked tests
- tracked public configs
- tracked public documentation

Checks:

- no IP addresses
- no PEM paths
- no credentials or tokens
- no private target config paths
- no model files
- no private absolute paths
- no dependency on ignored private artifacts

Generated `.autotuneai/` artifacts remain ignored and are not required for
public quick-start execution. Real benchmark artifacts must remain sanitized
before they are committed to documentation.

Latest Phase 13 local scan:

```text
checked_files: 311
violations: 0
```

Phase 14A adds the `pytorch-training` adapter and keeps the same public safety
boundary:

- generated configs are public placeholders
- no AWS lifecycle control is performed
- no model files or datasets are downloaded
- PyTorch is not a mandatory dependency
- candidate overlays are copied into artifact directories
- Python source files are not rewritten
- arbitrary parser commands and shell strings are not supported

Phase 16 release-candidate boundary:

- required validation uses local clean environments and GitHub Actions only
- AWS targets are not accessed or controlled
- generated wheel/sdist must not include `.autotuneai/private`, ignored run
  artifacts, model files, datasets, PEM files, or local absolute paths
- the installed-wheel smoke test creates all fixture files in a temporary
  directory outside the source worktree
- the release smoke verifies `autotuneai` console-script installation, `init`,
  `doctor`, `tune`, `run --auto`, `run --best`, cache operations, and ArmTune
  fake benchmark provenance
- ArmTune fake benchmark evidence must remain `result_origin: test_fixture`
- PyTorch remains optional and is not installed by required release smoke jobs
- BootTrace production code remains outside the default PerfRunbench workflow

Latest Phase 16 local scan:

```text
checked_source_files: 340
checked_untracked_source_files: 9
checked_distributions:
  - dist-v0.2/perfrunbench-0.2.0-py3-none-any.whl (159 entries)
  - dist-v0.2/perfrunbench-0.2.0.tar.gz (242 entries)
violations: 0
```

Literal policy/test strings such as `.pem` or `.autotuneai/private` are allowed
only when they appear as negative-test assertions or documented forbidden
patterns, not as real private paths.
