# Kernel Profile Autotuning

Phase 21A implements planning for reboot-required Linux kernel profiles. It
does not modify GRUB, reboot a host, install a kernel, or make any boot profile
permanent.

## Boot Profiles

Implemented planning IDs:

| profile | purpose |
| --- | --- |
| `aws-4k-default` | Standard Ubuntu AWS ARM64 4K kernel without isolation parameters. |
| `aws-4k-nohz-rcu-irq` | One-time plan for `nohz_full=2-7`, `rcu_nocbs=2-7`, `irqaffinity=0-1`. |
| `aws-4k-managed-irq-isolation` | One-time plan for `isolcpus=managed_irq,2-7`. |
| `aws-4k-domain-isolation` | Advanced less-flexible plan for `isolcpus=domain,managed_irq,2-7`, `nohz_full`, `rcu_nocbs`, and IRQ affinity. |
| `aws-64k-default` | Official Ubuntu AWS ARM64 64K kernel flavor without isolation parameters. |
| `aws-64k-nohz-rcu-irq` | 64K flavor with the same verified isolation policy as the 4K plan. |

## Required Evidence

Every boot plan records kernel inventory, generated command line, fallback
kernel, user snapshot marker state, one-time boot requirement, post-reboot
health checks, SSH availability, workload availability, rollback plan, and
restoration verification.

The planner checks kernel configuration evidence for `CONFIG_NO_HZ_FULL` and
`CONFIG_RCU_NOCB_CPU` before enabling `nohz_full` and `rcu_nocbs` plans.

## Safety Rules

Phase 21A never performs a real bootloader write. Future real validation must
require all of:

```text
explicit --allow-boot-change
user snapshot marker present
fallback kernel present
generated config validated
recovery instructions generated
system idle
no active workload lease
```

The known-good kernel must not be removed. A boot profile must not become
permanent until a one-time boot has passed and rollback has been verified.

`aws-4k-domain-isolation` is marked:

```text
less_flexible_boot_isolation: true
```

## Phase 21B2 Readiness Audit

Phase 21B2 performed only a read-only boot-validation readiness audit on the
C7g target. It did not begin Phase 21C and did not modify GRUB, install a
kernel package, reboot, or write a bootloader configuration.

The audit result remained:

```text
status: read_only_completed
ready_for_phase21c_boot_kernel_validation: false
```

The blocking condition was lack of an explicit Phase 21C boot-change approval
and out-of-band recovery path for that run. This is intentional; reversible
runtime system validation must not silently become boot/kernel validation.

## Phase 21C0 Readiness Model

Phase 21C0 adds an explicit read-only transaction model for future one-time
kernel boot experiments. It is still planning and fixture validation only:

```text
kernel_or_boot_state_modified: false
real_reboot_performed: false
kernel_installed: false
```

New planning profile IDs:

| profile | purpose |
| --- | --- |
| `original-4k-default` | Current known-good AWS 4K kernel with the original command line. |
| `4k-nohz-rcu-irq` | One-time 4K plan for `nohz_full`, `rcu_nocbs`, and IRQ affinity. |
| `4k-managed-irq-isolation` | One-time 4K managed IRQ isolation without domain isolation. |
| `4k-domain-isolation` | Advanced less-flexible 4K domain isolation plan. |
| `64k-default` | Official AWS ARM64 64K kernel metadata plan with no isolation additions. |
| `64k-nohz-rcu-irq` | 64K plan with bounded nohz/RCU/IRQ isolation. |
| `64k-managed-irq-isolation` | 64K managed IRQ isolation gated by earlier 64K boot evidence. |

The read-only CLI surface is:

```text
autotuneai system boot-doctor --json
autotuneai system kernel-inventory --json
autotuneai system boot-plan --profile <profile> --inventory <inventory.json>
autotuneai system boot-plan verify --plan <plan.json>
autotuneai system recovery-plan --inventory <inventory.json>
```

Phase 21C0 includes no real reboot command and no bootloader write command.
Unknown bootloader, Secure Boot, lockdown, kernel config, 64K package metadata,
initramfs generation, fallback kernel, snapshot, Serial Console, SSH health
check, rollback command, or user authorization evidence remains an explicit
blocker.

## Phase 21C0 C7g Remote Read-Only Audit

Phase 21C0 was also validated against a user-started C7g / Neoverse-V1 target
using read-only inventory commands only. The audit identified the running AWS
4K kernel as the known-good baseline:

```text
current_kernel_release: 6.17.0-1019-aws
current_base_page_size: 4096
bootloader: grub_uefi
fallback_kernel_verified: true
```

It also identified official AWS ARM64 64K package metadata without installing
or booting that kernel:

```text
package: linux-image-6.14.0-1018-aws-64k
version: 6.14.0-1018.18~24.04.1
qualification_state: metadata_qualified_not_installed
source_state: repository_metadata_only
```

All modified boot profiles remained blocked because recovery prerequisites and
one-time boot verification were not present in the private evidence. The audit
confirmed:

```text
phase21c0_remote_readonly_audit: passed
ready_for_phase21c_real_boot_validation: false
kernel_or_boot_state_modified: false
```

No Phase 21C reboot, bootloader write, kernel installation, initramfs
generation, or permanent default change was performed.

## Phase 21C2A Real 4K Boot Isolation Validation

Phase 21C2A executed the real 4K boot-isolation experiment on the already
installed known-good AWS 4K kernel. It did not install or remove kernel
packages, did not use a 64K kernel, did not alter EFI BootOrder, and did not
change the permanent GRUB default.

Executed 4K profile order:

| label | profile | appended parameters |
| --- | --- | --- |
| G0 | `original-4k-default` | none |
| G1 | `4k-nohz-rcu-irq` | `nohz_full=2-7`, `rcu_nocbs=2-7`, `irqaffinity=0-1` |
| G2 | `4k-managed-irq-isolation` | G1 parameters plus `isolcpus=managed_irq,2-7` |
| G3 | `4k-domain-isolation` | G1 parameters plus `isolcpus=domain,managed_irq,2-7` |

Experimental profiles used isolated temporary GRUB custom entries. The
original generated Ubuntu menu entry was not edited, `grub.cfg` was regenerated
through normal GRUB tooling rather than hand-edited, and the permanent default
remained unchanged. Each measured profile started from verified G0 and returned
to verified G0 before another experimental profile was attempted.

Measured result:

```text
G0_E2_decode_median: 13.52 tok/s
G1_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G2_E3_decode_median: 13.63 tok/s, +0.81%, inconclusive
G3_E3_decode_median: 12.59 tok/s, -6.88%, regressed
remote_artifact_revalidation: 410/410 passed
downloaded_artifact_revalidation: 410/410 passed
original_4k_state_restored: true
performance_improvement_verified: false
```

G1 and G2 remained inside the configured 5% noise band. G3 regressed and is not
eligible as an optimization result.

## Phase 21C2B0 64K Local Preparation

Phase 21C2B0 prepares future official Ubuntu AWS ARM64 64K validation without
accessing AWS, installing packages, editing GRUB, changing EFI state,
generating initramfs, or rebooting a machine.

New plan-only commands:

```text
autotuneai system kernel-package-plan --profile aws-arm64-64k --inventory <inventory.json>
autotuneai system kernel-package-plan verify --plan <plan.json>
autotuneai system grub-default-plan --inventory <inventory.json> --safe-entry <entry-id>
autotuneai system kernel-experiment-plan --profile <G4|G5|G6|G7> --inventory <inventory.json>
autotuneai system kernel-experiment-plan verify --plan <plan.json>
autotuneai system kernel-cleanup-plan --before <before.json> --after <after.json>
```

The 64K package transaction model requires an apt simulation and rejects kernel
removal, unrelated upgrades, bootloader replacement, unsigned packages,
architecture changes, dependency conflicts, and insufficient disk evidence. The
future real install command must use the exact package version qualified during
preflight.

Generated GRUB entries are now parsed as source evidence. Temporary entries
must preserve the exact parsed `linux` path, `initrd` path, and root/search
preparation lines. Hardcoded guesses such as constructing `/boot/vmlinuz-*`
from a kernel release are ineligible.

Installing a newer kernel can change the resolved ordinary boot target without
changing the original GRUB configuration text. Phase 21C2B0 therefore plans a
guarded G0 safety pin, but does not apply it. Real 64K validation remains
blocked unless either the existing default or the safety pin guarantees ordinary
boot into the verified G0 kernel.

G7 is an advanced interaction candidate because Phase 21C2A showed 4K domain
isolation regressed. It requires separate explicit authorization and can run
only after G4, G5, and G6 each boot successfully and return to exact G0.

## Phase 21C2B Real 64K Validation

Phase 21C2B executed the prepared flow on the C7g target using the official
Ubuntu AWS ARM64 64K package transaction. The run installed only the qualified
64K package set, used a temporary G0 safety pin for ordinary boots, booted
G4/G5/G6/G7 through one-time GRUB entries, and then removed exactly the
introduced packages.

```text
original_kernel: 6.17.0-1019-aws
candidate_64k_kernel: 6.17.0-1019-aws-64k
package_transaction_new_packages: 6
package_transaction_upgrades: 0
package_transaction_removals: 0
final_kernel: 6.17.0-1019-aws
final_page_size: 4096
package_cleanup: passed
```

The temporary-entry renderer was fixed during this phase to preserve generated
Ubuntu GRUB `initrdfail` conditionals and tab-separated `linux` commands. Both
defects were detected before any affected experimental reboot occurred.

The 64K page-size comparisons were inconclusive within the configured 5% noise
band, while G7 domain-isolation interaction regressed and remains ineligible.
