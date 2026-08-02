# Asset Status

The deck has no required placeholder assets.

## Actual CLI Capture

- Slide: 3
- Commands:

  ```bash
  autotuneai guide
  autotuneai init --adapter llama-bench --target ssh --preset quick
  autotuneai guide --project .autotuneai/perfrunbench-project
  autotuneai explain decode-heavy
  ```

- Asset: `public/assets/armtune-guided-cli.png`
- Provenance: generated locally by
  `scripts/render-armtune-guided-screenshot.mjs` from the installed source tree
- Classification: public configuration-only UX demonstration
- Claim boundary: no SSH, network, workload, AWS, or measured result

## Generic Fixture CLI Capture

- Slide: appendix/reference only
- Command:

  ```bash
  autotuneai run --auto --preset quick -- python app.py
  ```

- Asset: `public/assets/cli-run-auto-fixture.png`
- Provenance: `../evidence/SCREENSHOT_PROVENANCE.md`
- Classification: public synthetic fixture UX demonstration
- Claim boundary: not Arm or production performance evidence

## Optional Demo Video

The optional video is intentionally omitted. The deck contains no pending demo
URL or placeholder demo QR.

## Already Sourced

### Server CPU Hero

- Slide: 1
- Asset: `public/assets/server-cpu-hero.png`
- Source: original unbranded image generated for this deck
- Claim boundary: generic hardware visual; measured C7g/T4g facts come from the
  linked public evidence, not from the image

### GitHub Actions Screenshot

- Slide: 10
- Source:
  `https://github.com/xuwwwww/PerfRunbench/actions/runs/30234620475`
- Must show the successful Windows x86-64, Ubuntu x86-64, and Ubuntu Arm64
  installed-wheel jobs.

### GitHub Release Screenshot

- Slide: 11
- Source:
  `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1`
- Must show the release title, published state, and assets without account-only
  controls.

### Repository QR Code

- Slide: 11
- Encodes:
  `https://github.com/xuwwwww/PerfRunbench`
- Asset:
  `public/assets/repository-qr.svg`

### C7g Same-Protocol Size-Scaling Charts

- Slides: 6, 7, and 8
- Assets:
  - `public/assets/c7g-size-thread-scaling.svg`
  - `public/assets/c7g-size-serving-before-after.svg`
  - `public/assets/c7g-size-workload-guard.svg`
- Generated from:
  `docs/results/c7g-size-scaling/c7g_size_scaling_validation.json`
- Source evidence:
  `docs/results/C7G_SIZE_SCALING_VALIDATION.md`
- Classification: measured Arm64 evidence
- Claim boundary: three C7g sizes with one pinned model/runtime identity and
  declared llama-bench and loopback protocols; hardware scaling, runtime
  tuning, and remote-client results remain separate

## Never Include

- AWS public or private IP addresses
- SSH key paths or host-key fingerprints
- Security Group IDs
- model paths or model weights
- local absolute paths
- cloud credentials or tokens
- private artifact directories
