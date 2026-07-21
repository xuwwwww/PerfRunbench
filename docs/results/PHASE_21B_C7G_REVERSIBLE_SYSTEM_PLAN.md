# Phase 21B C7g Reversible System Plan

Phase 21B has now been executed. The real validation record is:

```text
docs/results/PHASE_21B_C7G_REVERSIBLE_SYSTEM_VALIDATION.md
```

This plan remains as the pre-run design boundary for reversible non-boot
system profiles.

Prerequisites:

```text
AWS instance manually started by user
latest branch checked out cleanly
model and workload digests unchanged
host snapshot captured
snapshot marker supplied by user
root free space and memory headroom checked
no active workload lease
```

Candidate stages:

```text
baseline-current
cpufreq-performance when exposed
affinity-only
cpuset-partition-1h7w / cpuset-partition-2h6w when supported
irq-workqueue-isolation when supported
thp-always / thp-madvise / thp-never
allocator-glibc-bounded-arenas
storage-read-ahead only for cold_model_load
vm-system-conservative only for exposed relevant sysctls
hugetlb-bounded only when workload mappings prove use
```

Success criteria:

- every applied setting has exact before/requested/effective evidence
- every setting rolls back exactly
- no dirty host state remains
- profiles are compared only under their protocol ID
- unsupported and unexercised profiles are excluded from ranking
- artifact manifests validate remotely and after download

Lack of improvement is a valid result.

Actual Phase 21B outcome:

```text
phase21b_c7g_reversible_system_validation: passed
performance_improvement_verified: false
host_restored_to_original_state: true
```
