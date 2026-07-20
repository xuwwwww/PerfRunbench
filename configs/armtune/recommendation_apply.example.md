# Guarded Recommendation Apply Example

This file is a public placeholder. Replace the paths with local artifacts from
a measured run and keep real private configuration outside git.

Show a bundle:

```bash
autotuneai recommendation show \
  --bundle .autotuneai/recommendations/<bundle_id>.json
```

Validate the bundle and its source Passport digest:

```bash
autotuneai recommendation validate \
  --bundle .autotuneai/recommendations/<bundle_id>.json
```

Dry-run process-scoped apply:

```bash
autotuneai recommendation apply \
  --bundle .autotuneai/recommendations/<bundle_id>.json \
  --scope process \
  --dry-run
```

Run a guarded canary with a lease:

```bash
autotuneai recommendation apply \
  --bundle .autotuneai/recommendations/<bundle_id>.json \
  --scope process \
  --lease-minutes 30 \
  --canary-runs 1
```

Rollback the active lease:

```bash
autotuneai recommendation rollback \
  --bundle .autotuneai/recommendations/<bundle_id>.json
```
