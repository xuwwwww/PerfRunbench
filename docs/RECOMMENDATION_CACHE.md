# Recommendation Cache

The default cache root is:

```text
.autotuneai/cache/recommendations/
```

Each cache entry points at a validated RecommendationBundle and records:

- `schema_version`
- cache ID
- bundle ID, path, and SHA-256
- workload identity digest
- environment fingerprint
- objective
- selected configuration ID
- expiry and revoked state
- summary metrics

Cache operations:

```bash
autotuneai cache list
autotuneai cache show <cache-id>
autotuneai cache invalidate <cache-id>
autotuneai cache gc
```

## Match States

- `exact_match`: workload identity matches and hard/soft environment fields
  match.
- `soft_drift`: workload identity matches, hard fields match, soft fields
  drifted.
- `hard_drift`: workload identity matches, but a hard environment field
  changed or is unknown.
- `expired`: only matching entries are past their TTL.
- `revoked`: only matching entries were manually invalidated.
- `missing`: no matching workload identity exists.
- `corrupt`: an entry, bundle path, or bundle digest is invalid.
- `ambiguous`: more than one eligible matching entry exists and operator
  action is required.

New `tune` results revoke older entries for the same workload identity before
inserting the new entry.

For training workloads, the cached RecommendationBundle records both a bounded
slice launch spec for canary validation and a full-run launch spec for the
actual workload launch. Cache reuse remains tied to the original workload
identity, environment fingerprint, objective, metric protocol, and adapter
protocol digests.

