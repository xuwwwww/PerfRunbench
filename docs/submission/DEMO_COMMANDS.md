# Demo Commands

These commands are for a clean local Windows PowerShell demo. They do not
depend on AWS, private artifacts, PyTorch, downloaded models, a developer
worktree, or editable installation.

Use the GitHub Release wheel downloaded from:

```text
https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1
```

## 1. Create A Temporary Demo Directory

```powershell
$DemoRoot = Join-Path $env:TEMP "perfrunbench-v031-demo"
Remove-Item -LiteralPath $DemoRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $DemoRoot | Out-Null
Set-Location $DemoRoot
```

## 2. Create A Fresh Virtual Environment

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## 3. Install The v0.3.1 Wheel

Replace `<downloads>` with the directory where the public GitHub Release wheel
was saved.

```powershell
& .\.venv\Scripts\python.exe -m pip install "<downloads>\perfrunbench-0.3.1-py3-none-any.whl"
```

## 4. Confirm CLI Availability

```powershell
$Python = Resolve-Path .\.venv\Scripts\python.exe
$Autotune = Resolve-Path .\.venv\Scripts\autotuneai.exe
& $Autotune --help
```

## 5. Create A Dependency-Free Demo Workload

```powershell
@'
from __future__ import annotations
import argparse
import os
from autotune.training_metrics import emit_result, emit_step

parser = argparse.ArgumentParser()
parser.add_argument("--micro-batch-size", type=int, default=4)
parser.add_argument("--gradient-accumulation", type=int, default=2)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=None)
args = parser.parse_args()

steps = int(args.max_steps or 3)
cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
throughput = 100.0 + args.micro_batch_size * 10.0 + args.num_workers * 4.0 + cpu_threads * 3.0
if args.gradient_accumulation == 1:
    throughput += 8.0

step_time_ms = 1000.0 / throughput
warmup_steps = 1 if args.max_steps else 0
measured_steps = max(1, steps - warmup_steps)
for step in range(1, steps + 1):
    emit_step(completed_steps=step, metrics={"step_time_ms": step_time_ms})
emit_result(
    completed_steps=measured_steps,
    metrics={
        "completed_steps": measured_steps,
        "samples_per_second": throughput,
        "step_time_ms": step_time_ms,
        "final_loss": 0.1,
        "examples_processed": measured_steps * args.micro_batch_size,
        "slice_limit_applied": args.max_steps is not None,
    },
)
'@ | Set-Content -LiteralPath train.py -Encoding UTF8
```

## 6. Initialize Public Config

```powershell
& $Autotune init --adapter pytorch-training --output autotune.yaml
```

## 7. Doctor

```powershell
& $Autotune doctor --config autotune.yaml -- $Python train.py
```

## 8. First Auto Run: Cache Miss

```powershell
& $Autotune run --auto --config autotune.yaml -- $Python train.py
```

Expected talking points:

- cache miss;
- bounded tuning;
- selected candidate or honest inconclusive result;
- no unsafe machine-wide tuning.

## 9. Second Auto Run: Cache Hit

```powershell
& $Autotune run --auto --config autotune.yaml -- $Python train.py
```

Expected talking points:

- cache lookup;
- environment validation;
- no retuning on exact match;
- canary and process-scoped reuse when eligible.

## 10. Run Best

```powershell
& $Autotune run --best --config autotune.yaml -- $Python train.py
```

Expected talking point:

- `run --best` requires an existing eligible recommendation and never retunes.

## 11. Show Cache

```powershell
& $Autotune cache list
```

## 12. Show Result Summary

```powershell
Get-ChildItem -Recurse -Filter "*.json" .autotuneai | Select-Object -First 20 FullName
```

## 13. Verify Cleanup

```powershell
Get-ChildItem -Recurse .autotuneai | Where-Object {
  $_.Name -match "lease|lock|active"
}
```

## Showing Saved Real Evidence

Use public repository files only. If you are recording from a source checkout,
show these sanitized evidence documents:

```powershell
Get-Content docs\armtune\results\PHASE_9_FIRST_REAL_ARM_TUNING.md
Get-Content docs\armtune\results\PHASE_10_GUARDED_APPLY_VALIDATION.md
Get-Content docs\results\PHASE_14B_REAL_ARM_PYTORCH_VALIDATION.md
Get-Content docs\results\PHASE_15B_T4G_REAL_PYTORCH_VALIDATION.md
Get-Content docs\results\PHASE_15C_WINDOWS_X86_PYTORCH_VALIDATION.md
```

Do not show private target configs, cloud console pages, IP addresses, PEM
paths, model paths, or hidden local artifact directories.
