# Phase 21C2B C7g 4K Versus 64K Kernel Validation

Phase 21C2B performed a guarded real C7g 4K-versus-64K kernel validation on
the existing Linux Arm64 / Neoverse-V1 target. The goal was safety and
measurement integrity: install the official Ubuntu AWS ARM64 64K kernel
transaction, boot bounded one-time profiles, measure the same pinned workload,
then restore the original 4K package and boot state.

No public connection details, SSH trust material, model paths, cloud resource
identifiers, or private recovery markers are recorded in this document.

## Source Boundary

```text
branch: validation/phase21c2b-c7g-64k-real
initial_source_commit: bc5d89204beaed72ca4f78a1a31857742536bee5
final_source_commit: f2175f69e37e2d826237d395a04b1c04db40e786
target_architecture: Linux aarch64
target_cpu_class: Neoverse-V1
logical_cpus: 8
original_kernel: 6.17.0-1019-aws
original_page_size: 4096
candidate_64k_kernel: 6.17.0-1019-aws-64k
candidate_64k_page_size: 65536
workload: pinned llama.cpp llama-bench and llama-server
model: existing Qwen2.5-7B Q4_K_M artifact from prior phases
```

Two focused source fixes were required before G5/G6/G7 could be safely booted:

- `10a2f02` preserved generated GRUB `initrdfail` conditionals when rendering
  temporary entries.
- `f2175f6` accepted tab-separated generated `linux` commands.

Both fixes were discovered before any affected experimental reboot occurred.

## Package Transaction

The qualified candidate was the official Ubuntu AWS ARM64 64K kernel package
set at version `6.17.0-1019.19~24.04.1`.

New packages installed:

```text
linux-aws-64k
linux-headers-6.17.0-1019-aws-64k
linux-headers-aws-64k
linux-image-6.17.0-1019-aws-64k
linux-image-aws-64k
linux-modules-6.17.0-1019-aws-64k
```

The exact apt simulation had:

```text
new_packages: 6
upgraded_packages: 0
removed_packages: 0
kernel_removals: 0
```

After installation, `CONFIG_ARM64_64K_PAGES=y`, expected `/boot` artifacts,
the modules directory, and generated non-recovery GRUB entries were verified.

## Default Safety

Installing the 64K kernel introduced a newer generated kernel entry. A
temporary G0 safety pin was applied so ordinary boots continued to resolve to
the original 4K kernel. The pin used generated GRUB entry evidence and was
restored after package cleanup.

```text
safety_pin_applied: true
ordinary_default_during_experiments: G0 original 4K
one_time_experimental_entries_only: true
permanent_default_restored: true
```

## Measurement Identities

| identity | profile | kernel | page size | CPUs | threads | role |
| --- | --- | --- | ---: | --- | ---: | --- |
| H1 | G0 | `6.17.0-1019-aws` | 4096 | `0-7` | 8 | 4K full-capacity baseline |
| H2 | G0 | `6.17.0-1019-aws` | 4096 | `2-7` | 6 | 4K matched six-CPU baseline |
| H3 | G4 | `6.17.0-1019-aws-64k` | 65536 | `0-7` | 8 | 64K full-capacity baseline |
| H4 | G4 | `6.17.0-1019-aws-64k` | 65536 | `2-7` | 6 | 64K matched six-CPU baseline |
| H5 | G5/G6/G7 | `6.17.0-1019-aws-64k` | 65536 | `2-7` | 6 | 64K isolated six-CPU profile |

## Llama-Bench Results

Each profile used two warm-ups, seven measured repetitions, three-second
cooldown, identical prompt/generation counts, identical model and binary
digests, and a 5% noise band.

| label | decode median tok/s | decode min | decode p95 | decode CV | prefill median tok/s | failed trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 4K full | 17.46 | 17.15 | 17.538 | 0.00725 | 43.26 | 0 |
| H2 4K matched | 13.63 | 13.62 | 13.637 | 0.00047 | 32.28 | 0 |
| H3 64K full | 17.55 | 17.51 | 17.570 | 0.00129 | 43.31 | 0 |
| H4 64K matched | 13.64 | 13.30 | 13.650 | 0.00982 | 32.28 | 0 |
| G5 H5 64K nohz/RCU/IRQ | 13.61 | 13.38 | 13.670 | 0.00691 | 32.45 | 0 |
| G6 H5 64K managed IRQ | 13.67 | 13.60 | 13.707 | 0.00246 | 32.45 | 0 |
| G7 H5 64K domain interaction | 12.63 | 12.52 | 12.667 | 0.00378 | 32.43 | 0 |

## Comparisons

| comparison | delta | classification |
| --- | ---: | --- |
| H1 versus H3 page-size full-capacity | +0.515% | inconclusive |
| H2 versus H4 page-size matched six-CPU | +0.073% | inconclusive |
| H4 versus G5 H5 64K isolation | -0.220% | inconclusive |
| H4 versus G6 H5 64K isolation | +0.220% | inconclusive |
| H4 versus G7 H5 domain interaction | -7.405% | regressed |

No page-size or 64K isolation improvement was verified outside the 5% noise
band. G7 reproduced the prior domain-isolation risk pattern as a regression.

## Server And Cold-Load Results

`llama-server` was used only as a secondary latency protocol.

| label | TTFT median ms | TTFT p95 ms | E2E median ms | E2E p95 ms | failures | port released |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H1 4K server | 57.850 | 58.851 | 912.096 | 915.625 | 0 | true |
| H3 64K server | 58.985 | 59.399 | 933.453 | 945.668 | 0 | true |

Cold-load used the same declared cache protocol for both page sizes.

| label | model-ready median ms | model-ready p95 ms | failed trials |
| --- | ---: | ---: | ---: |
| H1 4K cold-load | 40029.029 | 40061.963 | 0 |
| H3 64K cold-load | 39027.036 | 39060.165 | 0 |

The cold-load difference remained inside the 5% decision band and is reported
as inconclusive.

## Restoration And Artifacts

```text
remote_artifact_tar_sha256_check: passed
downloaded_artifact_manifest_revalidation: 621/621 passed
final_kernel: 6.17.0-1019-aws
final_page_size: 4096
temporary_grub_entries_removed: true
next_entry_removed: true
package_cleanup: passed
packages_removed: 6
residual_64k_artifacts: none
residual_processes: none
residual_ports: none
dpkg_audit: clean
original_4k_state_restored: true
package_state_restored: true
permanent_default_restored: true
```

## Outcome

```text
phase21c2b_64k_kernel_validation: passed
page_size_performance_improvement_verified: false
page_size_cold_load_improvement_verified: false
page_size_latency_improvement_verified: false
64k_isolation_improvement_verified: false
original_4k_state_restored: true
package_state_restored: true
permanent_default_restored: true
ready_for_phase21d_serving_validation: true
ready_to_stop_c7g: true
```

Phase 21C2B validated that PerfRunbench can safely install, boot, measure, and
remove the official AWS ARM64 64K kernel transaction on C7g while preserving
recoverability and artifact integrity. It did not find a verified performance,
cold-load, or latency improvement outside the configured noise band.
