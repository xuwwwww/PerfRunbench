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

The C7g remote read-only audit found official AWS ARM64 64K package metadata:

```text
package: linux-image-6.14.0-1018-aws-64k
version: 6.14.0-1018.18~24.04.1
qualification_state: metadata_qualified_not_installed
```

This does not create a 4K/64K performance comparison. Every 64K profile remains
blocked until a future phase explicitly authorizes kernel installation,
one-time boot selection, recovery validation, and protocol-matched
measurements.

## Phase 21C2A 4K-Only Isolation Boundary

Phase 21C2A is not a 4K/64K comparison. It keeps the installed AWS 4K kernel
and changes only allowlisted one-time boot parameters:

```text
G1: nohz_full=2-7 rcu_nocbs=2-7 irqaffinity=0-1
G2: G1 + isolcpus=managed_irq,2-7
G3: G1 + isolcpus=domain,managed_irq,2-7
```

The primary causal comparison is the matched six-CPU G0 baseline versus the
experimental six-CPU profile. The full eight-CPU G0 baseline is retained only
as a secondary product-capacity tradeoff. It must not be used as the denominator
for a boot-isolation speedup percentage.

Phase 21C2B remains the earliest possible phase for installing, booting, or
measuring a 64K kernel.

## Phase 21C2B0 Local Preparation

Phase 21C2B0 does not install or boot a 64K kernel. It prepares the local
guardrails needed before a future real C7g 64K session:

- exact official AWS ARM64 64K package transaction planning
- apt simulation safety checks
- generated GRUB entry parsing
- resolved default boot target checks
- temporary G0 safety-pin planning
- G4/G5/G6/G7 one-time experiment plans
- exact package cleanup planning
- fake end-to-end rollback and failure plans

Future real Phase 21C2B must use fresh same-session baselines:

| identity | role | PAGE_SIZE | CPUs | threads |
| --- | --- | --- | --- | --- |
| H1 | 4K full-capacity baseline | 4096 | `0-7` | 8 |
| H2 | 4K matched six-CPU baseline | 4096 | `2-7` | 6 |
| H3 | 64K full-capacity baseline | 65536 | `0-7` | 8 |
| H4 | 64K matched six-CPU baseline | 65536 | `2-7` | 6 |
| H5 | 64K isolated six-CPU profile | 65536 | `2-7` | 6 |

Causal comparisons are limited to H1 versus H3, H2 versus H4, and H4 versus
H5. Old Phase 21C2A results must not be reused as denominators.

G7 (`64k-domain-isolation-interaction`) is marked as advanced because the
analogous 4K domain isolation profile regressed. It needs separate real-phase
authorization and cannot be eligible for automatic recommendation by request
alone.

## Phase 21C2B Real Results

Phase 21C2B executed fresh same-session C7g measurements for all planned
identities. Old Phase 21C2A values were not reused as denominators.

| comparison | baseline | candidate | delta | classification |
| --- | --- | --- | ---: | --- |
| Page-size full-capacity | H1 4K full `17.46 tok/s` | H3 64K full `17.55 tok/s` | +0.515% | inconclusive |
| Page-size matched six-CPU | H2 4K matched `13.63 tok/s` | H4 64K matched `13.64 tok/s` | +0.073% | inconclusive |
| 64K nohz/RCU/IRQ | H4 `13.64 tok/s` | G5 H5 `13.61 tok/s` | -0.220% | inconclusive |
| 64K managed IRQ | H4 `13.64 tok/s` | G6 H5 `13.67 tok/s` | +0.220% | inconclusive |
| 64K domain interaction | H4 `13.64 tok/s` | G7 H5 `12.63 tok/s` | -7.405% | regressed |

Server and cold-load protocols remained separate from steady-state decode.
Neither protocol showed a verified improvement outside the configured 5%
decision band.
