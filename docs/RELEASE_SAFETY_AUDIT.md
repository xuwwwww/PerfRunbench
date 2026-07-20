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
