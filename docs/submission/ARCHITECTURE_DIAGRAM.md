# PerfRunbench Architecture Diagram

This diagram is for Devpost, the README narrative, and the demo video. It
covers the default PerfRunbench product workflow.

## Mermaid Source

```mermaid
flowchart TD
    command["User command and config"]
    doctor["Doctor and workload identity"]
    adapter["Workload adapter"]
    candidates["Candidate generator"]
    slices["Bounded tuning slices"]
    guards["Metrics, progress, and resource guards"]
    stats["Statistical confirmation"]
    passport["Performance Passport"]
    cache["RecommendationBundle and cache"]
    validate["Environment validation"]
    apply["Canary and process-scoped apply"]
    workload["Full workload"]
    cleanup["Cleanup or rollback"]

    command --> doctor
    doctor --> adapter
    adapter --> candidates
    candidates --> slices
    slices --> guards
    guards --> stats
    stats --> passport
    passport --> cache
    cache --> validate
    validate --> apply
    apply --> workload
    workload --> cleanup

    adapter -. examples .-> llama["llama.cpp"]
    adapter -. examples .-> pytorch["PyTorch training"]
    adapter -. examples .-> generic["generic command"]

    doctor -. targets .-> arm["Linux Arm64"]
    doctor -. targets .-> win["Windows x86_64"]
    doctor -. targets .-> ssh["SSH remote target"]
```

## Generated Assets

- `docs/submission/assets/perfrunbench_architecture.svg`
- `docs/submission/assets/perfrunbench_architecture.png`

The generated assets use the same flow as the Mermaid source and are sized for
16:9 demo video capture and Devpost display.
