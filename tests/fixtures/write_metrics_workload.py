from __future__ import annotations

import json
import os
from pathlib import Path


run_dir = os.environ.get("AUTOTUNEAI_RUN_DIR")
if not run_dir:
    raise RuntimeError("AUTOTUNEAI_RUN_DIR not set")

payload = {
    "duration_seconds": 0.01,
    "epoch_time_mean_seconds": 0.005,
    "step_time_mean_seconds": 0.002,
    "samples_per_second": 123.0,
    "final_accuracy": 0.95,
    "final_loss": 0.1,
    "peak_batch_payload_mb": 1.5,
}
Path(run_dir, "training_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
print("metrics written")
