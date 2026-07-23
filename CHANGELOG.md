# Changelog

## v0.3.0

- Package workload-aware `llama-server` serving autotuning as the main v0.3
  release-candidate feature.
- Add validated remote serving recommendations for `slots4-cont-on` under the
  tested C7g server, T4g remote-client topology, Qwen2.5-7B Q4_K_M model,
  pinned binary, concurrency 4, and direct private network protocol.
- Add `autotuneai serve recommend`, `autotuneai serve explain`, and
  `autotuneai serve run --best` dry-run guard reporting.
- Add Serving Performance Passport output while preserving existing v0.2
  Performance Passport parsing as a separate passport kind.
- Add public demo docs and validated result summaries for serving autotuning.
- Preserve negative guard results for compiler/backend, reversible system,
  mmap/mlock/prefault, allocator, THP/HugeTLB, storage, 4K isolation, 64K
  kernel, and network tuning categories.

## v0.2.0

- Release the public golden-path benchmarking, tuning, RecommendationCache,
  guarded apply, and ArmTune v0.2 workflow.

