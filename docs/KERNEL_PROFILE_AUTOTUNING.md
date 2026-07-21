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
