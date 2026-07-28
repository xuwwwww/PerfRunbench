# Screenshot Provenance

## Public Fixture CLI Capture

The CLI screenshot was generated from an actual isolated execution against
source commit:

```text
5ccab60b0db61d7d8473e2254eccd2c2457c5714
```

Command:

```bash
autotuneai run --auto --preset quick -- python app.py
```

The executable was installed into a fresh Python 3.11 virtual environment from
the clean `origin/main` worktree. `app.py` was copied from
`public_fixture_app.py`. The command exited with code 0.

Observed result:

```text
CLI status: completed
auto status: no_eligible_candidate
tuning status: inconclusive
classification: within_noise_band
unique search candidate limit: 3
candidate count including baseline: 4
failed trials: 0
residual fixture processes: 0
active lease or lock files: 0
```

The generic command runner measured real local process wall time, but the
workload itself is a synthetic fixture. Therefore this capture is classified
as `test_fixture_demo`, and `performance_evidence` is false. It must not be
used as Arm or production performance evidence.

The preserved raw CLI payload is `public_fixture_run_auto.json`. The rendering
script reads that payload and `public_fixture_run_auto_summary.json`; it does
not invent a recommendation or alter the observed status.

Final artifacts:

```text
public_fixture_run_auto.json
SHA-256 63383500bd1fa124e3ea839993ecca4cffcf43344d7fad2e6483ba3d7bc839d3

cli-run-auto-fixture.png
SHA-256 3d04aeb65bf246d1a5fdcbd6fedbf672c3fcba7452a20338327024377906d174

PerfRunbench-Arm-AI-Challenge.pdf
SHA-256 19b73708aa1d96b59f819421bfd48654744b39a487e34da43aa17b3006a3de78
pages 9
media box 735.12 x 414 points
```

Visual QA checked all nine slides at 1366x768 and 1920x1080. It reported zero
overflow failures.
