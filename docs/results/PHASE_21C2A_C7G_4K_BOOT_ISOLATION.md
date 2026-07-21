# Phase 21C2A C7g 4K Boot Isolation

Phase 21C2A validates reversible one-time 4K boot-parameter isolation profiles
on the existing C7g / Neoverse-V1 target. This is real host-system and
workload evidence, but it did not find a performance improvement outside the
configured 5% noise band.

No public connection details, SSH key paths, host-key fingerprints, model
paths, or cloud resource identifiers are recorded in this document.

## Source Boundary

```text
branch: validation/phase21c2a-4k-boot-isolation
source_commit: 4e5a11bc10906649bbe756a3094488c7d30824b9
target_architecture: Linux aarch64
target_cpu_class: Neoverse-V1
logical_cpus: 8
kernel: 6.17.0-1019-aws
base_page_size: 4096
workload: pinned llama.cpp llama-bench and llama-server
model: existing Qwen2.5-7B Q4_K_M artifact from prior phases
```

The controller and remote checkout were both verified at the expected commit
before mutation. The remote source checkout was clean before measurement.

## Profiles

| profile | label | appended parameters |
| --- | --- | --- |
| G0 | `original-4k-default` | none |
| G1 | `4k-nohz-rcu-irq` | `nohz_full=2-7`, `rcu_nocbs=2-7`, `irqaffinity=0-1` |
| G2 | `4k-managed-irq-isolation` | G1 parameters plus `isolcpus=managed_irq,2-7` |
| G3 | `4k-domain-isolation` | G1 parameters plus `isolcpus=domain,managed_irq,2-7` |

Baseline identities remained separated:

| identity | kernel/profile | workload CPUs | threads | role |
| --- | --- | --- | --- | --- |
| E1 | G0 | `0-7` | 8 | full-capacity baseline |
| E2 | G0 | `2-7` | 6 | matched causal baseline |
| E3 | G1/G2/G3 | `2-7` | 6 | experimental six-CPU profile |

Only E2 versus E3 is a causal boot-isolation comparison. E1 is a full-capacity
reference and is not used as the percentage denominator for boot isolation.

## Recovery Event

The first G1 attempt exposed a harness defect before any experimental
measurement: the temporary GRUB entry generated a kernel path with a `/boot/`
prefix, while the target's original GRUB entry used the root-level
`/vmlinuz-*` and `/initrd.img-*` paths.

Observed failure class:

```text
failed_entry: temporary G1 GRUB entry
failure_mode: kernel image path not found
measurement_included: false
recovery_path: EC2 Serial Console, manual selection of the original Ubuntu entry
post_recovery_cleanup: passed
post_recovery_g0_validation: passed
```

The validation harness was corrected to preserve the original GRUB kernel and
initrd paths. Corrected G1, G2, and G3 transactions then started from verified
G0 and returned to verified G0 after each profile.

## Llama-Bench Results

Each benchmark used two warm-ups followed by seven measured repetitions.
Warm-ups were excluded. All reported values are same-target, same-model,
same-protocol measurements.

| label | role | decode median tok/s | decode min | decode p95 | decode CV | prefill median tok/s | failed trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0_E1 | full-capacity baseline | 17.43 | 17.41 | 17.46 | 0.0010 | 43.28 | 0 |
| G0_E2 | matched causal baseline | 13.52 | 13.47 | 13.58 | 0.0023 | 32.26 | 0 |
| G1_E3 | nohz/rcu/irq isolation | 13.63 | 13.54 | 13.65 | 0.0026 | 32.44 | 0 |
| G2_E3 | managed IRQ isolation | 13.63 | 13.61 | 13.68 | 0.0017 | 32.44 | 0 |
| G3_E3 | domain isolation | 12.59 | 12.58 | 12.61 | 0.0008 | 32.43 | 0 |

Causal comparison against G0_E2:

| profile | decode median tok/s | delta vs G0_E2 | classification |
| --- | ---: | ---: | --- |
| G1_E3 | 13.63 | +0.81% | inconclusive |
| G2_E3 | 13.63 | +0.81% | inconclusive |
| G3_E3 | 12.59 | -6.88% | regressed |

G1 and G2 stayed inside the configured 5% noise band. G3 regressed and is not
eligible as an optimization result.

## Llama-Server Secondary Results

The secondary server protocol checked request completion, nonempty streamed
output, TTFT, E2E latency, cleanup, and port release. It did not run BootTrace
semantic validation.

| label | TTFT median ms | TTFT p95 ms | E2E median ms | failures | port released |
| --- | ---: | ---: | ---: | ---: | --- |
| G0_E1 | 58.170 | 58.890 | 917.850 | 0 | true |
| G0_E2 | 74.169 | 76.886 | 1177.137 | 0 | true |
| G1_E3 | 73.506 | 73.904 | 1167.247 | 0 | true |
| G2_E3 | 73.463 | 73.934 | 1166.990 | 0 | true |
| G3_E3 | 198.948 | 202.046 | 3142.350 | 0 | true |

G1 and G2 did not provide a latency improvement outside the noise boundary.
G3 materially worsened latency.

## Restoration and Validation

```text
remote_artifact_revalidation: 410/410 passed
downloaded_artifact_revalidation: 410/410 passed
tarball_sha256_match: passed
residual_processes: none
residual_ports: none
temporary_grub_entries_removed: true
next_entry_removed: true
original_cmdline_restored: true
kernel_restored: 6.17.0-1019-aws
page_size_restored: 4096
permanent_default_unchanged: true
original_4k_state_restored: true
```

## Outcome

```text
phase21c2a_4k_boot_isolation: passed
performance_improvement_verified: false
latency_or_noise_improvement_verified: false
selected_profile: none
eligible_recommendation: none
ready_for_phase21c2b_64k_validation: false
ready_to_stop_c7g: true
```

Phase 21C2A successfully validated the 4K one-time boot isolation protocol,
rollback, cleanup, artifact preservation, and evidence boundaries. It did not
find an eligible 4K boot-isolation performance improvement.
