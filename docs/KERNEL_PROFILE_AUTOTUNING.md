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
