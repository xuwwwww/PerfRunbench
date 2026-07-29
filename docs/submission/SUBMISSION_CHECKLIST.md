# Submission Checklist

The checked repository items are reproducible from public files. Devpost form
items must be checked only after the saved submission is inspected in the
browser.

## Repository And Release

- [x] `origin/main` contains the public v0.3.1 release state.
- [x] `v0.3.1` tag exists.
- [x] GitHub Release v0.3.1 exists and is public.
- [x] Wheel and sdist are downloadable.
- [x] Downloaded release SHA-256 digests match `PROJECT_FACTS.md`.
- [x] README presents PerfRunbench as the product.
- [x] README presents ArmTune as the Arm inference subsystem.
- [x] BootTrace is optional experimental work outside the default path.
- [x] Root MIT `LICENSE` is present.
- [x] Public submission files contain no private connection material.
- [x] Source deck and exported PDF are included.

## Devpost Copy

- [x] English title is ready.
- [x] English tagline is ready.
- [x] English short description is ready.
- [x] English full description is ready.
- [x] Cloud AI track wording is explicit.
- [x] Built-with list is ready.
- [x] Repository URL is ready.
- [x] v0.3.1 release URL is ready.
- [x] Limitations and claim boundaries are included.
- [x] Significant updates since June 10, 2026 are documented.
- [x] Project fields saved and re-opened in Devpost.
- [x] Cloud AI track explicitly identified in the Devpost story.
- [x] Repository and release links saved in Devpost.
- [x] Signed-in Devpost account and individual team entry confirmed.
- [ ] Arm Developer Program registration must be confirmed in the entrant's
  separate Arm account.

## Screenshots And Slides

- [x] Actual public fixture `run --auto` output captured.
- [x] Fixture screenshot explicitly says it is not performance evidence.
- [x] Real C7g/T4g result slide is protocol-scoped.
- [x] Guard and inconclusive behavior are shown.
- [x] GitHub Actions compatibility screenshot is included.
- [x] GitHub v0.3.1 release screenshot is included.
- [x] Submission deck exports to an eleven-page PDF.
- [x] Desktop visual QA reports no overflow.
- [x] Final seven-image screenshot set uploaded to Devpost.
- [x] C7g.4xlarge evidence overview is prepared as the eighth sanitized image.
- [ ] The entrant has reviewed and uploaded the eighth screenshot to Devpost.
- [x] Same-protocol C7g size-scaling images are prepared as the ninth and tenth
  sanitized images.
- [ ] The entrant has reviewed and uploaded the ninth and tenth screenshots to
  Devpost.

## Submission Status

- [x] Official Rules and Devpost Terms accepted by the entrant.
- [x] A withdrawal/disregard request was sent to the challenge manager after a
  premature submission.
- [ ] The challenge manager has confirmed withdrawal or disregard.
- [ ] The entrant has completed a final review of all submission fields,
  screenshots, links, and claims.
- [ ] The demo video has been recorded and reviewed.
- [ ] The entrant has explicitly approved final submission after review.
- [ ] The final project has been submitted to the challenge.

## Evidence Integrity

- [x] Measured, synthetic, fixture, and compatibility evidence are separated.
- [x] Remote serving, build profile, runtime search, and secondary latency
  protocols are not combined.
- [x] `tokens/s`, requests/s, and `samples/s` are not compared directly.
- [x] C7g, T4g, and Windows are not ranked using unrelated protocols.
- [x] No global-best or universal-speedup claim is made.
- [x] CI compatibility is not presented as measured Arm performance.
- [x] Performance Passport verification is documented.
- [x] Guarded apply and rollback are documented.
- [x] Cross-target hard-drift refusal is documented.
- [x] Unsupported network, kernel, compiler, and 64K-page experiments are not
  presented as eligible improvements.

## Public Safety

- [x] No credentials or tokens.
- [x] No private or public AWS addresses.
- [x] No PEM paths or SSH fingerprints.
- [x] No model files or private model paths.
- [x] No private artifact paths or local absolute paths in tracked evidence.
- [x] No PyPI publication is claimed.
- [x] Submission work requires no running AWS resources.
- [x] Safety scan and `git diff --check` pass.

## Optional Video

- [x] Video is treated as optional under the competition rules.
- [x] No placeholder, fake URL, or pending demo QR is saved.
- [ ] The final demo video has been recorded and reviewed.
- [ ] Any entrant-approved video URL has been added to Devpost.
