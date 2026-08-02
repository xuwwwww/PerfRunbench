# ArmTune Competition Video Recording Guide

This guide produces a 3:00 to 3:30 competition demo without live AWS access.
The live segment demonstrates the product UX. Committed charts demonstrate
previously measured real Arm evidence.

## Recording Setup

- Record at 1920x1080, 30 fps.
- Use 125% to 150% terminal zoom and a font size that remains readable on a
  laptop screen.
- Close notifications, password managers, SSH agents, cloud consoles, and
  unrelated browser tabs.
- Use a clean working directory created only for the demo.
- Keep the challenge deck PDF, repository README, and measured result report
  open before recording.

Recommended tools:

- OBS Studio for screen and microphone capture;
- Windows Snipping Tool or Clipchamp only for trimming and captions;
- a wired or stable microphone at a fixed distance.

## Recording Method

Do not narrate while operating the CLI, and do not turn the entire demo into
an animation. Record silent, short product clips first, then record each
voice-over segment separately. Commands may be pasted before pressing Enter;
the evidence is the real tool behavior, not typing speed.

Use motion only for a short title, architecture arrows, before/after emphasis,
guard rejection, and section transitions. Keep the CLI, charts, Passport, and
repository views as real screen captures. This makes mistakes cheap to rerecord
and keeps the product demonstration credible.

## Prepare The Live Demo

Create an isolated environment from a clean source checkout or installed wheel:

```powershell
py -3.11 -m venv "$env:TEMP\perfrunbench-demo"
& "$env:TEMP\perfrunbench-demo\Scripts\python.exe" -m pip install <verified-wheel-path>
```

Open a new PowerShell window and activate it:

```powershell
& "$env:TEMP\perfrunbench-demo\Scripts\Activate.ps1"
autotuneai --help
```

Pretype these commands in a local text file so the recording does not include
typing mistakes:

```powershell
autotuneai guide
autotuneai init --adapter llama-bench --target ssh --output .autotuneai/demo-armtune --preset quick
autotuneai guide --project .autotuneai/demo-armtune
autotuneai explain interactive-short
autotuneai explain decode-heavy
autotuneai explain prefill-heavy
```

The generated starter contains placeholders and `result_origin: unverified`.
Do not replace them with private details for the recording.

## Capture Plan

Record short clips rather than one fragile continuous take:

| clip | duration | content | required label |
| --- | ---: | --- | --- |
| 1 | 20 s | Problem statement and ArmTune cover | `ArmTune: guarded Arm AI inference autotuning` |
| 2 | 30 s | unified `guide`, `init`, and project `guide` | `LIVE LOCAL UX - NO PERFORMANCE CLAIM` |
| 3 | 25 s | three top-level `explain` commands | `WORKLOAD DEFINITIONS` |
| 4 | 25 s | target inspection command shape | `PREPARED OUTPUT - PRIVATE DETAILS HIDDEN` |
| 5 | 35 s | measured cross-size before/after chart | `PREVIOUSLY MEASURED REAL ARM EVIDENCE` |
| 6 | 25 s | prefill-heavy rejection chart | `WORKLOAD-SPECIFIC GUARD` |
| 7 | 20 s | Passport and guarded apply flow | `AUDITABLE RECOMMENDATION` |
| 8 | 15 s | repository and release | `SOURCE, EVIDENCE, AND RELEASE` |

## Narration Rules

Say:

> Best observed eligible configuration under the tested Arm target, model,
> workload, runtime, concurrency, topology, and measurement protocol.

Do not say:

- "ArmTune always doubles Arm performance";
- "this is the globally best configuration";
- "prefill was optimized" when the shown candidate was rejected for
  prefill-heavy traffic;
- "the live local demo measured Arm performance";
- "KleidiAI was enabled" without the required evidence state.

## Privacy Review Before Export

Watch the complete video at 0.5x speed and check every frame for:

- IP addresses or hostnames;
- PEM or identity-file paths;
- usernames, tokens, account IDs, cloud billing pages, or browser profiles;
- private config and artifact paths;
- model storage paths or license-gated download pages;
- unrelated email, employer, school, or personal information.

Blurred secrets are still risky if earlier frames show them clearly. Re-record
the clip instead of relying on a small blur.

## Export And Final Review

- Export H.264 MP4 at 1080p, 30 fps, 10 to 16 Mbps.
- Keep the final duration below the competition limit shown on the active
  submission form.
- Verify audio on laptop speakers and headphones.
- Verify charts and terminal text at 100% playback size.
- Verify the repository URL is readable for at least five seconds.
- Upload only after the user reviews the final MP4 and the complete submission
  form.
