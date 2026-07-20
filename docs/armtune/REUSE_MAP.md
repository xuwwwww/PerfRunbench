# ArmTune Reuse Map

This map records the exact existing PerfRunbench symbols ArmTune plans to
reuse. If a required capability is missing, the gap is stated explicitly.

## Run ID and Run-Directory Creation

- Exact file path: `autotune/resource/run_state.py`
- Exact class/function/symbol name: `RUNS_DIR`
- Current responsibility: Defines the PerfRunbench run artifact root as
  `.autotuneai/runs`.
- How ArmTune will reuse it: Store ArmTune artifacts under
  `.autotuneai/runs/<run_id>/armtune/` rather than creating a second run store.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low; read-only reuse.
- Planned tests: `tests/test_arm_tune_cli.py` verifies ArmTune output paths are
  nested under the existing run root when a run directory is supplied.

- Exact file path: `autotune/resource/run_state.py`
- Exact class/function/symbol name: `RunManifest`
- Current responsibility: Dataclass for existing PerfRunbench run metadata.
- How ArmTune will reuse it: Phase 1 will not instantiate it directly, but later
  benchmark phases should attach ArmTune artifacts to existing run manifests.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if ArmTune keeps its own manifests under an
  `armtune/` subdirectory.
- Planned tests: Later benchmark-phase tests should assert existing manifest
  fields remain unchanged.

- Exact file path: `autotune/resource/run_state.py`
- Exact class/function/symbol name: `create_run`
- Current responsibility: Creates unique run IDs, run directories, initial
  manifests, git snapshots, and environment snapshots.
- How ArmTune will reuse it: Later benchmark phases should call `create_run`
  before creating the `armtune/` subdirectory.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if no signature changes are made.
- Planned tests: Later tests should assert ArmTune run creation preserves
  `manifest.json`, `before_status.txt`, `before_diff.patch`, and `head.txt`.

## Process Execution

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `run_with_budget`
- Current responsibility: Runs a workload with resource monitoring, executor
  selection, tuning lifecycle, and run artifact writes.
- How ArmTune will reuse it: Later local target benchmark execution can wrap
  llama.cpp commands with this function when target execution is local Linux
  Arm64.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium if ArmTune later needs target-side
  command semantics; keep any target-specific logic outside this function.
- Planned tests: Later fake workload tests should assert existing commands still
  work and ArmTune stores extra artifacts only under `armtune/`.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `launch_performance`
- Current responsibility: Runs a workload without resource sampling after
  applying cached performance tuning, then restores tuning.
- How ArmTune will reuse it: Later confirmation runs may use the same lifecycle
  pattern when minimal monitoring is selected.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium; ArmTune must not use this to avoid raw
  artifact preservation requirements.
- Planned tests: Later confirmation-run tests should assert raw stdout/stderr
  are still captured by ArmTune-specific wrappers.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_open_process`
- Current responsibility: Internal `subprocess.Popen` wrapper that starts a
  local process with process-group handling.
- How ArmTune will reuse it: Do not call this private symbol directly in Phase
  1. Reuse the design pattern only.
- Modification required: No.
- Backward-compatibility risk: High if used directly because it is private.
- Planned tests: None for direct ArmTune use; existing tests cover workload
  runner behavior.

## Process Lifecycle and Cleanup

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_tuning_lifecycle`
- Current responsibility: Applies system/GPU tuning and restores it in a
  `finally` block.
- How ArmTune will reuse it: Reuse the lifecycle pattern; direct reuse is later
  only through `run_with_budget` or `launch_performance`.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if ArmTune calls the private context
  manager directly.
- Planned tests: Later target benchmark tests should simulate failure and
  verify cleanup artifacts.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_terminate_process_tree`
- Current responsibility: Terminates local child processes and process groups,
  escalating from graceful termination to kill.
- How ArmTune will reuse it: Reuse only through existing workload runner in
  local benchmark phases; SSH worker needs its own tested remote cleanup.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if called directly due to private API.
- Planned tests: Existing workload runner tests plus later worker cleanup tests.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_terminate_systemd_scope`
- Current responsibility: Kills a systemd scope and waits for the child process.
- How ArmTune will reuse it: Reuse only through existing workload runner.
- Modification required: No.
- Backward-compatibility risk: High if called directly.
- Planned tests: Existing systemd executor tests; no direct Phase 1 ArmTune
  tests.

## Signal Handling

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_interrupt_restore_guard`
- Current responsibility: Temporarily handles SIGINT/SIGTERM, terminates the
  child process, and restores prior handlers.
- How ArmTune will reuse it: Reuse pattern and later indirect use through
  `run_with_budget`; remote worker signal handling must be implemented and
  tested separately.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if ArmTune directly depends on the private
  symbol.
- Planned tests: Later worker tests for SIGINT/SIGTERM cleanup; existing tests
  continue to cover workload runner behavior.

## CPU Affinity

- Exact file path: `autotune/resource/affinity.py`
- Exact class/function/symbol name: `apply_cpu_affinity`
- Current responsibility: Applies Linux process CPU affinity based on
  `ResourceBudget.allowed_threads`.
- How ArmTune will reuse it: Later local benchmark phases can apply affinity to
  the target-side process; Phase 1 topology inspection remains read-only.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium; ArmTune must not assume affinity proves
  Arm-specific topology usage.
- Planned tests: Later affinity-mode tests using monkeypatched psutil; Phase 1
  parser tests only inspect topology.

- Exact file path: `autotune/resource/budget.py`
- Exact class/function/symbol name: `ResourceBudget.allowed_threads`
- Current responsibility: Computes allowed thread count from reserved cores and
  CPU quota.
- How ArmTune will reuse it: Later candidate generation can respect controller
  or target thread budgets.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if treated as generic budget logic.
- Planned tests: Existing `tests/test_resource_budget.py`; later ArmTune
  candidate tests.

## Resource Sampling

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `ChildSample`
- Current responsibility: Dataclass for sampled RSS, CPU, cgroup, and per-CPU
  metrics.
- How ArmTune will reuse it: Later local benchmark phases may use existing
  resource summaries; ArmTune-specific inference metrics remain separate.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if treated as generic resource data.
- Planned tests: Existing run report and workload runner tests.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_monitor_child`
- Current responsibility: Samples local process-tree CPU/memory usage.
- How ArmTune will reuse it: Indirectly through `run_with_budget` only.
- Modification required: No.
- Backward-compatibility risk: High if called directly.
- Planned tests: Existing workload runner tests; later ArmTune local benchmark
  smoke tests.

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_summarize_timeline`
- Current responsibility: Converts `ChildSample` rows into resource summary
  metrics.
- How ArmTune will reuse it: Indirectly through `run_with_budget` only.
- Modification required: No.
- Backward-compatibility risk: High if called directly.
- Planned tests: Existing report and run analysis tests.

## Cgroup, Systemd, Docker, and Local Execution

- Exact file path: `autotune/resource/workload_runner.py`
- Exact class/function/symbol name: `_resolve_executor`
- Current responsibility: Chooses `local`, `systemd`, or `docker` executor and
  sudo mode.
- How ArmTune will reuse it: Later local-target benchmark phases may use the
  existing executor resolution; SSH target execution remains separate.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if called directly because it is private.
- Planned tests: Existing CLI and executor tests; later ArmTune local benchmark
  tests.

- Exact file path: `autotune/resource/systemd_executor.py`
- Exact class/function/symbol name: `build_systemd_run_command`
- Current responsibility: Builds a safe `systemd-run --scope` command with
  memory/CPU limits and environment variables.
- How ArmTune will reuse it: Later local Arm64 benchmark execution can use it
  through the existing workload runner.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium if ArmTune adds environment fields; avoid
  changing existing command semantics.
- Planned tests: Existing `tests/test_systemd_executor.py`; later ArmTune local
  target tests.

- Exact file path: `autotune/resource/cgroup_monitor.py`
- Exact class/function/symbol name: `read_cgroup_stats`
- Current responsibility: Reads cgroup memory and CPU counters.
- How ArmTune will reuse it: Later target-side local benchmark monitoring can
  reuse cgroup metrics where available.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if used read-only.
- Planned tests: Existing `tests/test_cgroup_monitor.py`; later ArmTune
  artifact tests.

- Exact file path: `autotune/resource/docker_executor.py`
- Exact class/function/symbol name: `build_docker_run_command`
- Current responsibility: Builds a Docker command with CPU/memory limits.
- How ArmTune will reuse it: Not used in Phase 1. Docker is not an initial
  production Arm target, but this remains reusable generic infrastructure.
- Modification required: No.
- Backward-compatibility risk: Low if left untouched.
- Planned tests: Existing `tests/test_docker_executor.py`.

- Exact file path: `autotune/resource/executor_capabilities.py`
- Exact class/function/symbol name: `collect_executor_capabilities`
- Current responsibility: Reports local/systemd/Docker capability status.
- How ArmTune will reuse it: `arm doctor` may report controller-side executor
  information as controller diagnostics, not Arm production evidence.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if labels remain controller diagnostics.
- Planned tests: `tests/test_arm_tune_cli.py` should verify controller
  diagnostics are separate from target validation.

## Paired Baseline Comparison

- Exact file path: `autotune/resource/comparison_runner.py`
- Exact class/function/symbol name: `compare_tuning`
- Current responsibility: Runs baseline/tuned pairs and records comparison
  JSON.
- How ArmTune will reuse it: Conceptual reuse only in Phase 1. Later ArmTune
  benchmark search will likely need ArmTune-specific candidate schemas while
  preserving paired baseline behavior.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium if generalized too early; avoid changing
  existing comparison JSON.
- Planned tests: Existing `tests/test_comparison_runner.py`; later ArmTune
  search tests.

- Exact file path: `autotune/resource/comparison_runner.py`
- Exact class/function/symbol name: `build_comparison_result`
- Current responsibility: Builds comparison records from two existing run IDs.
- How ArmTune will reuse it: Later passport/report phases may reuse or mirror
  delta calculation after compatibility checks.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if used read-only.
- Planned tests: Later passport comparison tests.

## Noise-Band Logic

- Exact file path: `autotune/recommendation/optimizer.py`
- Exact class/function/symbol name: `DEFAULT_NOISE_BAND_PERCENT`
- Current responsibility: Defines the default `+/-2%` noise band for existing
  recommendation decisions.
- How ArmTune will reuse it: Later ArmTune search should use the same default
  unless benchmark methodology justifies a different protocol value recorded in
  the experiment spec.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Low if imported as a constant later.
- Planned tests: Existing `tests/test_optimizer.py`; later ArmTune statistics
  tests.

- Exact file path: `autotune/recommendation/optimizer.py`
- Exact class/function/symbol name: `_decision_summary`
- Current responsibility: Produces baseline-best, within-noise,
  meaningful-speedup, or regression interpretation.
- How ArmTune will reuse it: Do not call this private symbol directly in Phase
  1. Later search may extract a small public generic helper if needed.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if ArmTune depends on a private function.
- Planned tests: Later public helper extraction tests if extraction is needed.

- Exact file path: `autotune/recommendation/optimizer.py`
- Exact class/function/symbol name: `_run_thermal_controlled_trials`
- Current responsibility: Runs candidate trials with nearby paired baseline
  controls for performance sweeps.
- How ArmTune will reuse it: Conceptual reuse only; ArmTune will need
  target-aware trial execution.
- Modification required: No for Phase 1.
- Backward-compatibility risk: High if called directly.
- Planned tests: Later ArmTune staged-search tests.

## Generic Serialization

- Exact file path: `autotune/resource/run_state.py`
- Exact class/function/symbol name: `write_json`
- Current responsibility: Writes JSON with sorted keys and UTF-8 encoding.
- How ArmTune will reuse it: Phase 1 should not use it for important ArmTune
  manifests because it is not atomic.
- Modification required: No for Phase 1; create an ArmTune-local atomic writer.
- Backward-compatibility risk: Low if left unchanged.
- Planned tests: `tests/test_arm_tune_protocol.py` should verify atomic JSON
  writes for ArmTune manifests.

- Exact file path: `autotune/utils/config.py`
- Exact class/function/symbol name: `load_config`
- Current responsibility: Loads JSON/YAML config files with a PyYAML fallback.
- How ArmTune will reuse it: Phase 1 config loading should call it, then apply
  ArmTune schema validation.
- Modification required: No.
- Backward-compatibility risk: Low if wrapped rather than modified.
- Planned tests: `tests/test_arm_tune_config.py` for JSON and minimal YAML
  target specs.

- Exact file path: none
- Exact class/function/symbol name: atomic JSON manifest writer
- Current responsibility: Does not exist in generic PerfRunbench.
- How ArmTune will reuse it: Not reusable; Phase 1 must add an ArmTune-local
  atomic writer in `autotune/arm_tune/targets/protocol.py`.
- Modification required: Add new ArmTune-local implementation.
- Backward-compatibility risk: Low because generic code is untouched.
- Planned tests: `tests/test_arm_tune_protocol.py`.

## HTML and JSON Report Helpers

- Exact file path: `autotune/report/run_report.py`
- Exact class/function/symbol name: `generate_run_report`
- Current responsibility: Generates Markdown or HTML reports for existing
  PerfRunbench run directories.
- How ArmTune will reuse it: Not used in Phase 1. Later ArmTune reports should
  reuse styling/helper patterns but keep Arm-specific sections under
  `autotune.arm_tune.report`.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium if Arm-specific sections are inserted
  into generic reports too early.
- Planned tests: Existing `tests/test_run_report.py`; later ArmTune report
  smoke tests.

- Exact file path: `autotune/report/comparison_report.py`
- Exact class/function/symbol name: `generate_comparison_report`
- Current responsibility: Generates Markdown or HTML reports for comparison
  JSON files.
- How ArmTune will reuse it: Later passport comparison reports may reuse helper
  patterns, but Phase 1 does not need report generation.
- Modification required: No for Phase 1.
- Backward-compatibility risk: Medium if generic report schemas are changed.
- Planned tests: Existing `tests/test_run_report.py`; later ArmTune comparison
  report tests.

- Exact file path: `autotune/resource/run_state.py`
- Exact class/function/symbol name: `write_json`
- Current responsibility: Generic JSON artifact helper.
- How ArmTune will reuse it: Non-critical JSON only; important ArmTune manifests
  require atomic writes.
- Modification required: No.
- Backward-compatibility risk: Low.
- Planned tests: ArmTune protocol tests for atomic writes; existing tests for
  generic JSON behavior.

## CLI Parser Insertion Point

- Exact file path: `autotune/cli.py`
- Exact class/function/symbol name: `build_parser`
- Current responsibility: Creates the top-level `autotuneai` argparse parser
  and all existing subcommands.
- How ArmTune will reuse it: Add a minimal `arm` subparser that forwards
  remaining arguments to `autotune.arm_tune.cli.main`.
- Modification required: Yes, minimal insertion only.
- Backward-compatibility risk: Medium because this is the top-level CLI; tests
  must verify existing commands still dispatch.
- Planned tests: `tests/test_arm_tune_cli.py` for `autotuneai arm doctor` and
  `autotuneai arm inspect-host`; existing `tests/test_cli.py` remains covered
  by the fast suite.

- Exact file path: `autotune/cli.py`
- Exact class/function/symbol name: `main`
- Current responsibility: Parses top-level args and dispatches to a subcommand
  handler.
- How ArmTune will reuse it: Existing dispatch remains unchanged; the `arm`
  handler is just another handler.
- Modification required: No direct signature change.
- Backward-compatibility risk: Low if `build_parser` insertion is narrow.
- Planned tests: Existing CLI tests plus new ArmTune CLI tests.
