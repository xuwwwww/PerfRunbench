# Phase 21C C7g Kernel Boot Plan

Phase 21C is a future real validation phase for reboot-required kernel
profiles. Phase 21A only creates fake/test plans.

Prerequisites:

```text
explicit user approval for boot changes
manual snapshot/backup marker
known-good fallback kernel present
generated boot command line validated
recovery instructions generated
system idle
no active workload lease
SSH health check plan
workload model availability check
```

Rules:

- use one-time boot first
- never remove the running or fallback kernel
- never make a boot profile permanent before one-time validation passes
- do not rank a boot profile unless workload evidence and rollback evidence
  are both valid
- do not prefer domain isolation over runtime cpuset isolation without measured
  evidence
