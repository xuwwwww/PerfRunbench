# Screenshot Shot List

Use screenshots that are readable on Devpost and in a 16:9 demo video. Redact
or crop anything private before uploading.

## 1. GitHub Repository Hero

- Visible: repository name, README title, quick start, release badge or link.
- Crop: browser toolbar can remain, but no other personal tabs.
- Redact: account-only notifications if visible.
- Aspect ratio: 16:9.
- Caption: `PerfRunbench repository and quick-start workflow.`

## 2. Public Quick Start

- Visible: `init`, `doctor`, and `run --auto` commands.
- Crop: keep only README quick-start section.
- Redact: none expected.
- Aspect ratio: 16:9 or 4:3.
- Caption: `One-command guarded autotuning workflow.`

## 3. Doctor Output

- Visible: readiness checks, config accepted, no workload launch.
- Crop: terminal only.
- Redact: local absolute paths.
- Aspect ratio: 16:9.
- Caption: `Doctor validates workload readiness before tuning.`

## 4. First Auto Run Cache Miss

- Visible: cache miss, bounded tuning start, candidate count.
- Crop: terminal only.
- Redact: local absolute paths.
- Aspect ratio: 16:9.
- Caption: `First run performs bounded tuning on cache miss.`

## 5. Selected Candidate

- Visible: selected candidate or explicit inconclusive status, confirmation
  summary, noise-band decision.
- Crop: terminal or result document only.
- Redact: private paths.
- Aspect ratio: 16:9.
- Caption: `PerfRunbench recommends only eligible measured candidates.`

## 6. Second Auto Run Cache Hit

- Visible: cache hit, no retuning, environment validation.
- Crop: terminal only.
- Redact: local absolute paths.
- Aspect ratio: 16:9.
- Caption: `Exact cache reuse avoids unnecessary retuning.`

## 7. Performance Passport Verification

- Visible: Passport status, verification passed, artifact manifest.
- Crop: public evidence document.
- Redact: any private local path if displayed.
- Aspect ratio: 16:9.
- Caption: `Performance Passport keeps recommendation evidence auditable.`

## 8. Environment Hard-Drift Refusal

- Visible: hard drift or incompatible environment, no `run --best` launch.
- Crop: terminal or validation matrix.
- Redact: private host details.
- Aspect ratio: 16:9.
- Caption: `Recommendations are refused on incompatible environments.`

## 9. Architecture Diagram

- Visible: full diagram from command to cleanup or rollback.
- Crop: generated diagram only.
- Redact: none expected.
- Aspect ratio: 16:9.
- Caption: `PerfRunbench guarded autotuning architecture.`

## 10. C7g llama Before/After Chart

- Visible: baseline, selected, `+28.25% observed decode throughput`.
- Crop: chart only.
- Redact: none expected.
- Aspect ratio: 16:9.
- Caption: `Protocol-specific Arm inference runtime tuning result.`

## 11. C7g PyTorch Before/After Chart

- Visible: baseline, selected, synthetic CPU-only workload footnote.
- Crop: chart only.
- Redact: none expected.
- Aspect ratio: 16:9.
- Caption: `Synthetic PyTorch guard workflow on C7g Arm64.`

## 12. Validation Matrix

- Visible: C7g, T4g, Windows rows and pass states.
- Crop: chart only.
- Redact: none expected.
- Aspect ratio: 16:9.
- Caption: `Multi-target validation matrix without cross-machine ranking.`

## 13. GitHub v0.2.0 Release

- Visible: title, tag, assets, release state.
- Crop: release page content only.
- Redact: account-only controls if visible.
- Aspect ratio: 16:9.
- Caption: `Published GitHub Release with verified wheel and sdist.`

## 14. Test And CI Success

- Visible: GitHub Actions release validation success across Windows, Linux, and
  Arm64.
- Crop: action run summary or job matrix.
- Redact: none expected.
- Aspect ratio: 16:9.
- Caption: `Installed-wheel validation across Windows, Linux, and Arm64 CI.`

## Never Show

- AWS IP addresses.
- PEM paths.
- Private cache paths.
- Absolute user paths.
- Hidden artifact directories.
- Keys, credentials, or tokens.
- Model weight paths.

