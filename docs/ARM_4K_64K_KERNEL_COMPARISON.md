# Arm 4K And 64K Kernel Comparison Plan

This document defines a future comparison boundary for Ubuntu AWS ARM64 4K and
64K kernels. Phase 21A only generates plans and fake validation artifacts.

Candidate families:

```text
aws-4k-default
aws-4k-nohz-rcu-irq
aws-4k-managed-irq-isolation
aws-4k-domain-isolation
aws-64k-default
aws-64k-nohz-rcu-irq
```

Phase 21C0 refines the future comparison into explicit boot profile labels:

```text
G0 original-4k-default
G1 4k-nohz-rcu-irq
G2 4k-managed-irq-isolation
G3 4k-domain-isolation
G4 64k-default
G5 64k-nohz-rcu-irq
G6 64k-managed-irq-isolation
```

`G0` remains the headline baseline. Every modified profile must be selected as
a one-time next boot first and must return to `G0` before another experimental
profile is attempted.

Future real validation must keep target instance identity, model digest,
workload binary digest, protocol ID, CPU allocation, sample counts, cooldown,
noise band, and artifact validation constant.

4K and 64K kernel measurements must be labeled by protocol. A lower TTFT or
faster model load must not be presented as steady-state decode improvement
unless that protocol separately passes its statistical guard.

Phase 21B2 did not run this comparison. It performed only a read-only
boot-validation readiness audit and left:

```text
ready_for_phase21c_boot_kernel_validation: false
```

No 4K/64K kernel claim exists until a future phase explicitly authorizes
one-time boot changes, reboot recovery, and protocol-matched measurements.

Phase 21C0 does not install or boot a 64K kernel. It can only qualify official
AWS ARM64 64K package metadata and candidate config evidence when the distro
repository exposes it. If metadata is unknown, the result remains blocked
rather than guessed.
