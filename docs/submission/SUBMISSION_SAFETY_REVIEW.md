# Submission Safety Review

Review date: 2026-07-28

Scope:

- root `LICENSE` and README submission references
- `docs/submission/`
- `docs/releases/v0.3.1.md`
- generated screenshot and deck assets

No AWS resource is required or accessed for submission packaging. No model is
downloaded. The only executed workload is a dependency-free public fixture
used to capture the config-free CLI experience; it is explicitly excluded
from performance evidence.

## Baseline Checks

| Check | Result |
| --- | --- |
| Source base | latest `origin/main` |
| Public version | `0.3.1` |
| Public release | GitHub Release v0.3.1 |
| License | MIT |
| Phase 21D2B evidence | present |
| Phase 21D3 productization evidence | present |
| v0.3.1 release documentation | present |
| Validation matrix | present |
| T4g visual training evidence | present |

## Public Links

- `https://github.com/xuwwwww/PerfRunbench`
- `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1`
- `https://github.com/xuwwwww/PerfRunbench/actions/runs/30234620475`
- `https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md`

## Claim Boundary Review

Passed:

- The 69.375%-86.490% throughput range and 91.128%-98.001% p95 TTFT
  reduction range are limited to the tested C7g server, T4g remote client,
  model, pinned runtime, direct private path, concurrency 4, and workload
  protocols.
- C7g build-profile, runtime-search, secondary server-latency, and remote
  serving protocols remain separate.
- Fixture CLI output is labeled `test_fixture` and is not performance evidence.
- Installed-wheel CI is compatibility evidence, not performance evidence.
- Synthetic PyTorch output is guard and workflow evidence, not production
  training evidence.
- The T4g visual result is limited to one synthetic CPU PyTorch protocol and
  discloses its burstable CPU-credit state.
- Compiler, kernel, 64K-page, and network experiments are not claimed as
  improvements.
- No global-best or universal-speedup wording is used.

## Privacy And Repository Review

The final scan covers text and generated assets for:

- credentials and tokens;
- private/public AWS addresses;
- SSH key and PEM paths;
- model paths or model files;
- private artifact paths;
- local absolute paths;
- account-only browser data; and
- unsupported performance claims.

Final scan result:

```text
scanned_text_files: 36
screenshots_validated: 7
submission_failures: 0
npm production vulnerabilities: 0
```

## Asset Review

- The deck is generated from checked-in Slidev source.
- Visual QA renders every page at 1366x768 and 1920x1080.
- The exported PDF page count and SHA-256 are recorded after final export.
- Devpost screenshots are copied from the validated 1920x1080 page renders.
- The CLI screenshot is generated from preserved public fixture JSON and a
  deterministic local HTML renderer.
- GitHub release and Actions screenshots show only public pages.
- No demo video or placeholder video URL is included.

## Remaining Browser Checks

- Save and re-open the Devpost project fields.
- Confirm Cloud AI track selection.
- Confirm signed-in account/team and Arm Developer Program registration.
- Upload and inspect the seven final screenshots.
