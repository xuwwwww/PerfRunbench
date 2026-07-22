# Phase 21D0 Serving Local Preparation

Phase 21D0 prepares workload-aware `llama-server` autotuning locally. It does
not access AWS, does not access a remote host, does not modify kernel, boot,
package, network, sysctl, or machine-wide state, and does not measure a real
serving result.

## Source Record

```text
origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c
phase21c2b_commit: 1c073fac7b63ad95392f8c70976eaa6584211b43
phase21d0_initial_commit: 1c073fac7b63ad95392f8c70976eaa6584211b43
pr6_status: open_draft
aws_accessed: false
remote_accessed: false
```

Phase 21C2B completed successfully before this phase:

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
```

## Implemented

- versioned serving capability models
- exact-binary capability discovery strategy
- standard workload profiles
- serving traffic profiles
- server runtime, memory, and network profile models
- deterministic load generator evidence
- staged bounded candidate planning
- fixture serving search
- Pareto result and named recommendation classes
- serving Passport extension payload
- hard-drift reuse guard
- lifecycle preflight for occupied ports and stale server processes
- Phase 21D2 network plan-only model
- `autotuneai serve` CLI surface

## Boundary

No Phase 21D0 result is a measured Arm serving result. All fake E2E outputs are
`test_fixture` and exist only to validate protocol, schema, guard, and artifact
behavior.

The next real phase is Phase 21D1: C7g `llama-server` loopback load generation.
