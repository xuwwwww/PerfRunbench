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

Future real validation must keep target instance identity, model digest,
workload binary digest, protocol ID, CPU allocation, sample counts, cooldown,
noise band, and artifact validation constant.

4K and 64K kernel measurements must be labeled by protocol. A lower TTFT or
faster model load must not be presented as steady-state decode improvement
unless that protocol separately passes its statistical guard.
