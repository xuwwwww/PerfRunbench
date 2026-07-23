# Phase 21C2B0 - Local Preparation For C7g 64K Kernel Validation

Phase 21C2B0 is implementation, fixture testing, package transaction planning,
boot-plan validation, recovery planning, and documentation only.

```text
aws_accessed: false
kernel_or_boot_state_modified: false
kernel_installed: false
grub_or_efi_modified: false
machine_rebooted: false
```

No public connection details, SSH key paths, host-key fingerprints, model paths,
cloud resource identifiers, or private recovery markers are recorded here.

## Git Source Record

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21c2a_commit: 621eca64f5d3406355d6b0aa312cbf5f22f5bd94
phase21c2b0_initial_commit: 621eca64f5d3406355d6b0aa312cbf5f22f5bd94
pr6_status: open draft
```

The branch is:

```text
validation/phase21c2b0-64k-local-preparation
```

## G1 Path-Failure Regression Guard

Phase 21C2A exposed a pre-measurement temporary-entry defect: the first G1
attempt generated a hardcoded `/boot/vmlinuz-*` path while the generated Ubuntu
GRUB menu entry used root-relative `/vmlinuz-*` and `/initrd.img-*` paths.

Phase 21C2B0 adds a hard guard:

- parse generated GRUB menu entries before creating any temporary entry
- preserve the exact `linux` path from the source generated entry
- preserve the exact `initrd` path when present
- preserve root/search/module preparation commands
- reject ambiguous duplicate non-recovery entries
- reject generated-path guessing
- mark every generated temporary entry with `path_source: generated_grub_menu_entry`
- require `path_guess_allowed: false`

A plan generated from a guessed kernel path is ineligible.

## Package Transaction Model

The new plan-only package transaction model records:

- distribution release and architecture
- configured repository evidence
- exact official AWS ARM64 64K candidate metadata
- apt simulation result
- newly installed, upgraded, removed, and held packages
- download and installed size evidence when available
- `/boot` and root free-space evidence
- expected kernel artifacts
- exact install command with the qualified version
- exact package rollback set
- package-cache cleanup policy
- final original-state verification requirements

The model rejects plans that include kernel removals, unrelated upgrades,
bootloader replacement, architecture changes, unsigned packages, dependency
conflicts, or insufficient disk evidence.

Phase 21C2B0 does not run `apt`, install a kernel, regenerate initramfs, or
write GRUB state.

## Permanent Default Safety

Phase 21C2B0 distinguishes:

```text
configuration_text_unchanged
resolved_default_boot_target_unchanged
```

A future 64K package installation may make an ordinary boot resolve to the new
64K kernel even when the original GRUB configuration text was not manually
edited. The new `grub-default-plan` command therefore plans a guarded G0 safety
pin when needed.

The safety pin remains plan-only in this phase. A future real phase must back
up changed files byte-for-byte, pin only a verified stable G0 entry ID, verify
ordinary boot resolution before any reboot, and restore the original semantics
after validation.

If neither the existing default nor the safety pin can guarantee ordinary G0
boot, real 64K validation remains blocked with `safe_default_unavailable`.

## 64K Profiles

Prepared profiles:

| label | profile | page size | appended parameters |
| --- | --- | --- | --- |
| G0 | `original-4k-default` | 4096 | none |
| G4 | `64k-default` | 65536 | none |
| G5 | `64k-nohz-rcu-irq` | 65536 | `nohz_full=2-7`, `rcu_nocbs=2-7`, `irqaffinity=0-1` |
| G6 | `64k-managed-irq-isolation` | 65536 | G5 parameters plus `isolcpus=managed_irq,2-7` |
| G7 | `64k-domain-isolation-interaction` | 65536 | G5 parameters plus `isolcpus=domain,managed_irq,2-7` |

G7 is marked:

```text
advanced_interaction_candidate: true
prior_4k_domain_isolation_regressed: true
```

G7 requires separate explicit authorization and cannot be auto-recommended
merely because exhaustive testing was requested.

## Comparison Identities

Fresh real Phase 21C2B measurements must create fresh baselines in the same
session. Old Phase 21C2A measurements cannot be reused as denominators.

| identity | purpose | kernel | PAGE_SIZE | CPUs | threads |
| --- | --- | --- | --- | --- | --- |
| H1 | 4K full-capacity baseline | G0 | 4096 | `0-7` | 8 |
| H2 | 4K matched six-CPU baseline | G0 | 4096 | `2-7` | 6 |
| H3 | 64K full-capacity baseline | G4 | 65536 | `0-7` | 8 |
| H4 | 64K matched six-CPU baseline | G4 | 65536 | `2-7` | 6 |
| H5 | 64K isolated six-CPU profile | G5/G6/G7 | 65536 | `2-7` | 6 |

Causal comparisons:

```text
page-size effect: H1 versus H3
page-size matched six-CPU effect: H2 versus H4
64K isolation effect: H4 versus H5
```

Eight-core baselines must not be used to claim six-core isolation effects.

## Authorization Boundary

Future real execution requires all of:

```text
--allow-kernel-package-install
--allow-temporary-grub-default-safety-pin
--allow-guest-reboot
--recovery-marker <private marker>
--serial-recovery-marker <private marker>
--maximum-reboots <bounded value>
```

G7 additionally requires:

```text
--allow-domain-isolation-interaction
```

Default behavior remains plan-only.

## Completion Flags

```text
phase21c2b0_64k_local_preparation: passed
g1_grub_path_regression_guarded: true
package_transaction_plan_ready: true
safe_g0_default_plan_ready: true
package_cleanup_plan_ready: true
ready_for_manual_c7g_start: true
kernel_or_boot_state_modified: false
aws_accessed: false
```
