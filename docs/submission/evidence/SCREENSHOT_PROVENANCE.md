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
SHA-256 2d6b7361f62a2c2c94262ee65aa68e92680adfb2181470c8a924c1547fb831c4
pages 11
media box 735.12 x 414 points
```

Visual QA checked all eleven slides at 1366x768 and 1920x1080. It reported
zero overflow failures.

## T4g Training Visual Evidence

The seventh screenshot was rendered with headless Chrome at 1920x1080 from the
sanitized checked-in HTML report. Its source is clean measured run:

```text
run_id: 20260728_142936_906425
source_commit: 43781a0dcafa58c24778ad310920a5c2c6196839
result_origin: measured
paired_confirmations: 12
```

The HTML renderer displays only the source filename, not a local absolute
path. The screenshot contains no AWS address, resource ID, SSH material,
account identifier, model path, or private artifact path.

Final artifacts:

```text
07-t4g-training-tuning-evidence.png
SHA-256 4460f157a929ff0a6800a6e830dfc9a7f5fcaa69eb2347c03c118ddc14a2dd98

training_tuning_report.html
SHA-256 86d01dcee6ded74c6e13d85f43888a133b44469b6dcced7f2fd529cb823cac48

training_summary.json
SHA-256 53083edbee9e3d579c71ae4bbec3a72b0b98b332a0861a9d312535025cc91021

paired_confirmation.csv
SHA-256 0a4c0de57f462cbb0885cbf9c949b2105fe94c294438bcaa77567f1af9c69848
```

## C7g.4xlarge Realistic Arm Inference Evidence

The eighth screenshot was rendered with headless Chrome at 1920x1080 from the
sanitized checked-in HTML report. Its source is one measured C7g.4xlarge run:

```text
source_commit: fab3297a62e510cf33b4cf27771707514f8f8542
result_origin: measured
target: C7g.4xlarge / Linux aarch64 / Neoverse-V1
measured_requests: 769
server_startups: 17
final_artifacts_checked_after_download: 278
final_artifacts_failed_after_download: 0
```

The screenshot contains no AWS address, resource ID, SSH material, account
identifier, model path, or private artifact path. It states that hardware
scaling and runtime tuning use separate protocols and that loopback evidence
is not remote-network evidence.

Final artifacts:

```text
08-c7g4xlarge-workload-aware-evidence.png
SHA-256 02a59badd80433c51951c132570ef3cfcb773d4c84fec2e98be8246eea76321b

arm_inference_report.html
SHA-256 a66d6f1b5c20fbe3a934477f7e421df1666d799302aabec2d08224d61cb013d0

arm_inference_summary.json
SHA-256 9ba49955ed27eb84eea44e2c5885be007346f4753af9d98e94b209f534683ee1
```

## C7g Same-Protocol Size-Scaling Evidence

The ninth and tenth screenshots are the 1920x1080 headless-Chrome renders of
slides 5 and 6 from the checked-in Slidev deck. Those slides embed charts
generated from the sanitized C7g size-scaling report. The report combines
three independently preserved measured summaries only after validating their
shared model, source, binary, runtime-bundle, topology, protocol, cleanup, and
artifact identities. The complete deck passed viewport QA at 1366x768 and
1920x1080 with zero overflow failures.

```text
source_under_test: 93b179a1c8a2370c1d6d3042909054af7ccfe666
result_origin: measured
targets: c7g.xlarge, c7g.2xlarge, c7g.4xlarge
measured_http_requests: 1,047
fresh_server_starts: 30
downloaded_artifacts_revalidated: 514
downloaded_artifact_failures: 0
```

The screenshots contain no AWS address or resource ID, SSH material, account
identifier, model path, private artifact path, or local absolute path. The
thread-scaling chart and runtime-tuning chart are explicitly separate
protocols. The tuning percentages are loopback results and are not combined
with the Phase 21D2B remote-client result.

Final artifacts:

```text
09-c7g-size-thread-scaling.png
SHA-256 5ecb437962fa944f0ca6be8061b8d7565667ca9e1781b92b78b172220b31a173

10-c7g-size-before-after.png
SHA-256 3d5abc639d6d8f500ee908d03ed858397407f977a400ca2bcfc536b7e72cc026

c7g_size_scaling_report.html
SHA-256 d1e0dd61b69f1bd3873ca3d01dc4c1359a2dc30e9a5f70a6e0ab59572f08a2c2

c7g_size_scaling_validation.json
SHA-256 d5e5f352c0210b79172682aea6d9eb1069f1b0e6c639518cf504000c838a63e4
```
