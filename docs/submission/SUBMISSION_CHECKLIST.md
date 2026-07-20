# Submission Checklist

## Repository

- [ ] `main` contains the intended release state.
- [ ] `v0.2.0` tag exists.
- [ ] GitHub Release exists.
- [ ] Release assets are downloadable.
- [ ] Release asset SHA-256 digests match the verified final artifacts.
- [ ] README presents PerfRunbench as the main product.
- [ ] README presents ArmTune as the Arm inference subsystem.
- [ ] BootTraceBench link is correct.
- [ ] BootTrace is not in the default tuning workflow.
- [ ] No unrelated dirty files are included in public history.
- [ ] License file is present or the limitation is acknowledged.

## Devpost

- [ ] Title: `PerfRunbench`.
- [ ] Tagline ready.
- [ ] Full description ready.
- [ ] Short description ready.
- [ ] Challenge category selected.
- [ ] Technologies list ready.
- [ ] Screenshots uploaded.
- [ ] Demo video uploaded.
- [ ] Repository URL added.
- [ ] Release URL added.
- [ ] Team member information complete.
- [ ] Acknowledgements complete.
- [ ] Limitations included.

## Evidence

- [ ] Measured, synthetic, fixture, and CI smoke evidence are clearly labeled.
- [ ] C7g llama.cpp build-profile results are not combined with runtime-search
  results.
- [ ] `tokens/s` and `samples/s` are never compared directly.
- [ ] C7g, T4g, and Windows are not ranked by throughput.
- [ ] No universal-best claim is made.
- [ ] Synthetic PyTorch results are not described as production training
  performance.
- [ ] Arm relevance is clear.
- [ ] Performance Passport verification is shown.
- [ ] Guarded apply validation is shown.
- [ ] Cross-target hard-drift refusal is shown.

## Video

- [ ] Duration is within the competition limit.
- [ ] Audio is understandable.
- [ ] Commands are readable.
- [ ] No private data is visible.
- [ ] No long waiting periods are included.
- [ ] Captions or subtitles are included if useful.
- [ ] Final call to action shows repository and release links.

## Final Safety

- [ ] No credentials.
- [ ] No private IP addresses.
- [ ] No PEM paths.
- [ ] No model weights.
- [ ] No private artifact paths.
- [ ] No AWS instance is left running for submission work.
- [ ] No PyPI publication is claimed.
