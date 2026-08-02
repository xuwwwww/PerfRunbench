# PerfRunbench ArmTune Final Demo Script

Use `docs/armtune/DEMO_SCRIPT.md` as the timed narration source and
`docs/submission/VIDEO_RECORDING_GUIDE.md` as the capture checklist.

The final demo must center ArmTune:

```text
guided project initialization
-> workload definitions
-> Linux Arm64 target boundary
-> measured before/after evidence
-> workload-specific rejection
-> Performance Passport
-> guarded reuse
```

The live commands are local UX evidence only:

```powershell
autotuneai arm init --output .autotuneai/demo-armtune --preset quick
autotuneai arm guide --project .autotuneai/demo-armtune
autotuneai arm explain decode-heavy
```

The performance segment uses committed measured evidence and must display the
label:

```text
PREVIOUSLY MEASURED REAL ARM EVIDENCE
```

Do not access AWS or reveal private target details while recording. Do not
submit the video or competition form until the user has reviewed the final
render.
