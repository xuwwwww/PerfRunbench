# Submission Safety Review

Review date: 2026-07-21

Scope:

- `docs/submission/`
- `docs/submission/assets/`
- `scripts/submission/`

No AWS instance was accessed, started, stopped, resized, or otherwise
controlled for this review. No benchmark workload was rerun.

## Baseline Checks

| Check | Result |
| --- | --- |
| Latest worktree branch | `docs/competition-submission` |
| Base commit | `80569b477d57c0f5489a0a5159ec3a914d3c5f2d` |
| `v0.2.0` tag resolves to base commit | passed |
| Package metadata version in `pyproject.toml` | `0.2.0` |
| Release documentation exists | passed |
| Validation matrix exists | passed |
| Phase 9 evidence exists | passed |
| Phase 10 evidence exists | passed |
| Phase 14B evidence exists | passed |
| Phase 15A evidence exists | passed |
| Phase 15B evidence exists | passed |
| Phase 15C evidence exists | passed |

## Public Link Validation

The public links returned HTTP 200 during validation:

- `https://github.com/xuwwwww/PerfRunbench`
- `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.2.0`
- `https://github.com/xuwwwww/BootTraceBench`

## Submission File Scan

The safety scan checked text assets for:

- private absolute paths;
- IPv4 addresses;
- SSH key-file extensions;
- private config or artifact paths;
- model weight file extensions;
- inline secret assignments;
- universal-best wording;
- cross-protocol uplift wording across incompatible protocols.

Result:

```text
checked_text_files: 12
checked_png_files: 4
violations: 0
```

## Claim Boundary Review

Passed:

- C7g llama.cpp build-profile results remain separate from runtime-search
  results.
- C7g llama-server TTFT is documented as a secondary latency protocol.
- C7g, T4g, and Windows PyTorch validations are not ranked against each other.
- Synthetic PyTorch results are labeled as synthetic guard and workflow
  evidence, not production training performance.
- Fixture and CI smoke evidence are not described as production Arm
  performance.
- Recommendation wording is bounded to the tested target, workload, runtime,
  and measurement protocol.

## Product Narrative Review

Passed:

- PerfRunbench is the main product identity.
- ArmTune is described as the Arm inference subsystem inside PerfRunbench.
- The default workflow is `init -> doctor -> run --auto`.
- The architecture diagram omits external experimental workloads.
- BootTraceBench appears only as a related external project or as a safety
  checklist boundary.

## Asset Review

Generated assets:

- `docs/submission/assets/c7g_llama_runtime_before_after.svg`
- `docs/submission/assets/c7g_llama_runtime_before_after.png`
- `docs/submission/assets/c7g_pytorch_before_after.svg`
- `docs/submission/assets/c7g_pytorch_before_after.png`
- `docs/submission/assets/multitarget_validation_summary.svg`
- `docs/submission/assets/multitarget_validation_summary.png`
- `docs/submission/assets/perfrunbench_architecture.svg`
- `docs/submission/assets/perfrunbench_architecture.png`

The chart data comes from fixed public constants in
`scripts/submission/render_submission_charts.py`. The script does not read
private artifacts, access the network, or run benchmarks.

## Remaining Manual Checks

- Capture final screenshots and redact browser/account chrome if needed.
- Record the demo video without showing local private directories or cloud
  console pages.
- Fill Devpost fields from `docs/submission/DEVPOST_FINAL.md`.
- Upload the GitHub Release link and repository link.
