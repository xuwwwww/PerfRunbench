from __future__ import annotations

from pathlib import Path

from autotune.golden_path.config import load_golden_path_config
from autotune.training_adapter.config import validate_training_config
from autotune.training_adapter.errors import TrainingConfigError


PYTORCH_TRAINING_TEMPLATE = """# PerfRunbench PyTorch training adapter config.
# Public placeholder values only; keep private paths, tokens, datasets, and cloud
# target configuration outside git.
schema_version: "1.0"
workload:
  adapter: pytorch-training
  cwd: .
  timeout_seconds: 120
  identity_files:
    - train.py
objective:
  metric: samples_per_second
  direction: maximize
metrics:
  source: stdout_json_prefix
  prefix: "AUTOTUNEAI_TRAIN_RESULT "
training:
  schema_version: "1.0"
  world_size: 1
  slice:
    mode: cli_argument
    flag: --max-steps
    warmup_steps: 1
    measured_steps: 8
    minimum_completed_steps: 8
    max_wall_time_seconds: 120
    omit_from_full_run: true
  progress:
    source: stdout_json_prefix
    step_prefix: "AUTOTUNEAI_TRAIN_STEP "
    result_prefix: "AUTOTUNEAI_TRAIN_RESULT "
    startup_timeout_seconds: 30
    no_progress_timeout_seconds: 30
  semantics:
    effective_batch_policy: preserve
    semantic_change_allowed: false
  capabilities:
    amp_modes:
      - disabled
    compile_modes:
      - disabled
    cpu_only: true
  search:
    max_candidates: 8
    shuffle_seed: 42
    stages:
      - input_pipeline
      - batch_config
      - cpu_runtime
  knobs:
    - name: micro_batch_size
      baseline: 4
      candidates: [4, 8]
      binding:
        type: cli_argument
        flag: --micro-batch-size
    - name: gradient_accumulation
      baseline: 2
      candidates: [2, 1]
      binding:
        type: cli_argument
        flag: --gradient-accumulation
    - name: dataloader_workers
      baseline: 0
      candidates: [0, 2]
      binding:
        type: cli_argument
        flag: --num-workers
    - name: cpu_threads
      baseline: 1
      candidates: [1, 2]
      binding:
        type: env_var
        name: OMP_NUM_THREADS
tuning:
  preset: quick
  warmup_runs: 0
  search_runs: 1
  confirmation_runs: 2
  min_valid_trials: 2
  noise_band_percent: 2.0
  shuffle_seed: 42
  max_candidates: 8
cache:
  root: .autotuneai/cache/recommendations
  retune_policy: on_hard_drift_or_missing
  canary_policy: on_new_or_drift
  ttl_days: 14
apply:
  canary_runs: 1
  lease_minutes: 30
"""


def write_init_config(output: str | Path, *, adapter: str, force: bool = False) -> Path:
    if adapter != "pytorch-training":
        raise TrainingConfigError(f"unsupported init adapter: {adapter}")
    path = Path(output)
    if path.exists() and not force:
        raise TrainingConfigError(f"refusing to overwrite existing config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PYTORCH_TRAINING_TEMPLATE, encoding="utf-8", newline="\n")
    validate_training_config(load_golden_path_config(path))
    return path
