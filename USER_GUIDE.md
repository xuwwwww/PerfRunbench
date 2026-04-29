# AutoTuneAI-Serve User Guide

這份文件是專案 root 的主要說明書，說明目前版本怎麼使用、最終版目標、resource guard / training wrapper 的設計方向，以及哪些 tuning 可以做、哪些不該自動做。

## 1. 專案目前定位

AutoTuneAI-Serve 目前是一個 resource-aware AI inference optimization project。

它現在已經可以：

- 在 WSL conda environment 裡執行。
- 跑 synthetic benchmark，快速驗證 CLI、tuner、scheduler。
- 跑真實 PyTorch CPU ResNet18 inference benchmark。
- 匯出 ONNX 並跑 ONNX Runtime CPU benchmark。
- 記錄 latency、throughput、memory、CPU usage。
- 用 CLI 選擇 memory budget、保留 CPU cores、CPU quota。
- 用 CPU affinity 保留部分 core 給系統或其他工作。
- 模擬 inference scheduler，包括 FCFS、static batching、dynamic batching、deadline-aware batching。

它還不是最終版。現階段最重要的是建立正確的架構、資料格式和實驗流程。

## 2. WSL conda 使用方式

進入 WSL：

```bash
wsl
```

進入專案：

```bash
cd /mnt/d/School/AutoTuneAI
```

啟用 conda environment：

```bash
source /home/louis/miniforge3/etc/profile.d/conda.sh
conda activate autotuneai
```

如果環境不存在：

```bash
/home/louis/miniforge3/bin/conda env create -f environment.yml
```

如果環境存在但套件不完整：

```bash
python -m pip install -r requirements.txt
```

驗證：

```bash
python -m unittest discover -s tests
```

## 3. Benchmark 怎麼跑

### 3.1 快速 synthetic benchmark

Synthetic benchmark 不跑真模型，用公式模擬 latency / throughput / memory，適合快速測整體流程。

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode synthetic
```

預設輸出：

```text
results/raw/resnet18_profile.json
results/raw/resnet18_profile.csv
```

### 3.2 PyTorch CPU real benchmark

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --output results/raw/resnet18_pytorch_smoke.json \
  --csv-output results/raw/resnet18_pytorch_smoke.csv
```

### 3.3 ONNX Runtime CPU real benchmark

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends onnxruntime \
  --max-configs 1 \
  --output results/raw/resnet18_onnx_smoke.json \
  --csv-output results/raw/resnet18_onnx_smoke.csv
```

第一次跑 ONNX Runtime 會產生：

```text
artifacts/onnx/resnet18.onnx
```

這些產生物已經被 `.gitignore` 排除，不會進 git。

## 4. Resource budget 怎麼用

目前 config 檔不預設資源限制。這是刻意的，因為這個專案希望能跑在不只一台電腦上，不應該把你的 RAM / CPU 設定寫死在 shared config 裡。

如果你要限制 benchmark 資源，用 CLI 指定：

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90
```

這些參數目前的意義：

- `--memory-budget-gb 22`: 希望 process peak RSS 不超過 22GB。
- `--reserve-cores 1`: 至少保留 1 個 logical CPU core，不讓 benchmark 使用。
- `--cpu-quota-percent 90`: 根據 CPU 數量限制可用 thread 數，並記錄 CPU 是否超標。

結果會包含：

```text
memory_budget_mb
effective_memory_budget_mb
reserve_cores
cpu_quota_percent
allowed_threads
cpu_affinity_applied
affinity_cores
total_memory_mb
available_memory_before_mb
available_memory_after_mb
peak_rss_mb
average_process_cpu_percent
peak_process_cpu_percent
average_system_cpu_percent
peak_system_cpu_percent
memory_budget_exceeded
cpu_quota_exceeded
```

注意：WSL 看到的 RAM 可能比 Windows 少。例如 Windows 顯示 23.7GB，但 WSL 可能只暴露 11.8GB。這時候 `effective_memory_budget_mb` 會自動用 WSL 可見 RAM 扣掉 reserved memory 後的安全值，不會盲目相信 22GB。

## 5. Auto-tuning 怎麼跑

目前 auto-tuning 還是用 synthetic profiler / synthetic cost model，目的是先驗證 search flow：

```bash
python scripts/run_autotune.py \
  --config configs/resnet18.yaml \
  --search cost_model \
  --trials 18
```

目前支援：

- exhaustive search
- random search
- cost-model-guided search
- latency / throughput / memory objective
- latency budget / memory budget filtering

下一階段會把 real benchmark sweep 的結果接到 recommender，讓它從實測資料中選出 safe best configuration。

## 5.1 Real mini sweep 和 recommender

目前已經有第一版 real mini sweep：

```bash
python scripts/run_real_sweep.py \
  --config configs/resnet18.yaml \
  --output results/raw/resnet18_real_sweep.json \
  --csv-output results/raw/resnet18_real_sweep.csv
```

預設會跑一個小 search space：

```text
backend: pytorch, onnxruntime
batch size: 1, 2, 4
threads: 1, 2, 4
precision: fp32
ONNX graph optimization: disable, all
```

如果只想跑很小的 smoke test：

```bash
python scripts/run_real_sweep.py \
  --config configs/resnet18.yaml \
  --backends pytorch \
  --batch-sizes 1 \
  --threads 1 \
  --output results/raw/test_real_sweep.json \
  --csv-output results/raw/test_real_sweep.csv
```

也可以加 resource budget：

```bash
python scripts/run_real_sweep.py \
  --config configs/resnet18.yaml \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90
```

產生 sweep 結果後，可以用 recommender 從實測 records 裡挑出 safe best configuration：

```bash
python scripts/recommend_config.py \
  --input results/raw/resnet18_real_sweep.json \
  --objective throughput \
  --latency-budget-ms 100 \
  --memory-budget-gb 22
```

支援 objective：

- `throughput`
- `latency`
- `memory`

輸出會包含：

```text
Recommended configuration
Measured performance
Reasoning
```

如果要給其他程式讀：

```bash
python scripts/recommend_config.py \
  --input results/raw/resnet18_real_sweep.json \
  --objective throughput \
  --json
```

## 6. Scheduler 怎麼跑

```bash
python scripts/run_scheduler.py \
  --workload burst \
  --scheduler deadline_aware \
  --count 100
```

目前 scheduler simulator 支援：

- `fcfs`
- `static`
- `dynamic`
- `deadline_aware`

輸出：

```text
average_latency
p95_latency
p99_latency
throughput
deadline_miss_rate
```

Dynamic batching 主要對 inference serving 有用。它不是 training memory 的直接解法。對 training RAM 爆掉，更有用的是 training wrapper、memory guard、batch-size tuning、gradient accumulation、checkpointing 等策略。

## 7. 最終版目標

最終版希望變成一個可以回答這個問題的工具：

```text
在這台機器的 RAM / CPU / latency / throughput 限制下，哪個 runtime configuration 最值得用？
```

理想使用方式：

```bash
autotuneai profile \
  --model resnet18 \
  --backends pytorch onnxruntime \
  --batch-sizes 1 2 4 8 16 \
  --threads auto \
  --memory-budget-gb 22 \
  --reserve-cores 1
```

接著：

```bash
autotuneai recommend \
  --results results/raw/resnet18_real_sweep.json \
  --objective throughput \
  --latency-budget-ms 30 \
  --memory-budget-gb 22
```

輸出概念：

```text
Recommended configuration
backend: onnxruntime
batch_size: 4
threads: 4
precision: fp32
graph_optimization: all

Measured performance
p95 latency: 24.8 ms
throughput: 162 samples/sec
peak RSS: 1.2 GB
CPU affinity: cores 0-6
safe under memory budget: yes

Reasoning
- Highest throughput among configs that stayed under effective memory budget.
- p95 latency satisfied the target.
- One logical CPU core was reserved for system responsiveness.
```

## 8. Training wrapper 的方向

你的想法很合理：這個 tool 不只可以做 inference optimization，也可以包住 training entrypoint，當成 resource monitor / limiter。

目前已經有第一版 wrapper：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90 \
  -- python train.py --config configs/train.yaml
```

目前會做：

- 建立 `.autotuneai/runs/<run_id>/`。
- 記錄 `manifest.json`。
- 記錄執行前 git 狀態到 `before_status.txt`。
- 記錄執行前 git diff 到 `before_diff.patch`。
- 記錄目前 git HEAD 到 `head.txt`。
- 記錄 Python / platform environment 到 `env.json`。
- 執行 child command。
- 監控 child process 與其子 process 的 RSS / CPU。
- 輸出 `resource_timeline.json`。
- 輸出 `resource_summary.json`。
- Ctrl-C 中斷時會 terminate child process 並把 run 標成 `interrupted`。

範例：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --sample-interval-seconds 0.5 \
  -- python train.py
```

如果想讓 wrapper 在超過 effective memory budget 時終止 child process：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --hard-kill \
  -- python train.py
```

查看歷史 runs：

```bash
python scripts/list_runs.py
```

回復某個 run 修改過的檔案：

```bash
python scripts/restore_run.py --run-id <run_id>
```

目前 wrapper 還不會自動修改 source code，所以通常會看到：

```text
Run <run_id> has no changed files to restore.
```

這是正常的。`restore_run.py` 是為下一階段 reversible source tuner 先建立的安全入口。

後續可做：

- soft budget warning。
- `systemd-run --scope -p MemoryMax=22G` hard memory limit。
- Docker `--memory` / `--cpus`。
- Linux cgroup v2。
- WSL `.wslconfig` 檢查與提醒。

短期先做 portable soft guard。hard limit 要依照環境判斷，不一定每台機器都支援。

## 9. Runtime tuning 還能動哪些東西

除了 backend / batch / threads / graph optimization，還可以做更多 runtime-level tuning。

### 9.1 不需要改 source code 的 tuning

- PyTorch:
  - `torch.set_num_threads`
  - `torch.set_num_interop_threads`
  - `torch.inference_mode`
  - `torch.compile`，視模型和平台而定
  - `channels_last` memory format，主要對 CNN 有機會有效
  - AMP / bfloat16 / fp16，視硬體支援而定

- ONNX Runtime:
  - graph optimization level
  - intra-op threads
  - inter-op threads
  - execution mode
  - memory arena settings
  - execution providers

- System/runtime:
  - CPU affinity
  - process niceness
  - environment variables，例如 `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
  - cgroup CPU / memory limits

### 9.2 需要改 source code 的 tuning

有些 training / inference tuning 需要改使用者程式碼，例如：

- training batch size
- gradient accumulation steps
- activation checkpointing
- dataloader workers
- dataloader prefetch factor
- pinned memory
- mixed precision
- model eval / no_grad / inference_mode
- input shape / sequence length
- save checkpoint frequency
- logging frequency

這些可以做，但一定要有 transaction / restore 機制。

## 10. 如果要改 source code，必須怎麼保護

這個 project 如果之後支援「自動修改 training source code」，一定要遵守這個流程：

1. 執行前建立 run id。

```text
.autotuneai/runs/2026-04-29_140000/
```

2. 記錄 git 狀態。

```bash
git status --short
git diff > .autotuneai/runs/<run_id>/before.patch
git rev-parse HEAD > .autotuneai/runs/<run_id>/head.txt
```

3. 每次修改前備份原檔。

```text
.autotuneai/runs/<run_id>/backup/path/to/file.py
```

4. 修改記錄成 manifest。

```json
{
  "run_id": "...",
  "changed_files": [
    {
      "path": "train.py",
      "backup": ".autotuneai/runs/.../backup/train.py",
      "reason": "reduce batch size from 64 to 16"
    }
  ]
}
```

5. 訓練完成後自動 restore。

6. 如果 process 被中斷，也要提供手動 restore script。

```bash
python scripts/restore_run.py --run-id 2026-04-29_140000
```

7. 如果 working tree 在執行前不是 clean，要拒絕自動修改，除非使用者明確允許。

這樣可以避免 tool 把你的 source code 改壞、訓練失敗後留下一堆髒狀態。

## 11. 中斷後如何回復的目標設計

未來應該提供：

```bash
python scripts/list_runs.py
python scripts/restore_run.py --run-id <run_id>
python scripts/show_run_diff.py --run-id <run_id>
```

如果被 Ctrl-C、OOM、kernel kill，下一次可以：

```bash
python scripts/restore_latest_run.py
```

restore 應該做：

- 停止殘留 child process。
- 還原 CPU affinity / niceness。
- 還原環境變數。
- 還原被修改的 source files。
- 輸出本次 restore 做了什麼。

如果專案是 git repo，最好也記錄：

```bash
git diff
git status
```

但 restore 不應該無腦 `git reset --hard`，因為那會刪掉使用者本來的未 commit 修改。

## 12. BIOS / UEFI tuning 的邊界

這個 tool 應該主打 runtime-level tuning，不應該自動進 BIOS / UEFI 改設定。

可以做：

- backend selection
- batch size
- precision
- graph optimization
- thread count
- CPU affinity
- process priority
- cgroup / Docker / systemd resource limits
- memory and CPU monitoring
- training source-level reversible tuning

不建議自動做：

- 改 BIOS / UEFI
- 開關 SMT / Hyper-Threading
- 改 Turbo Boost
- 改 power limit
- 改 XMP / EXPO
- 改 fan curve
- 改 memory frequency / timing

原因：

- 需要 reboot。
- 不同主機板差異很大。
- 沒有穩定跨平台 API。
- 風險高，不適合自動化。

可以做的是 `inspect-system`，列出建議檢查項，但不自動修改。

## 13. 市面上類似工具

這個 project 不是完全沒有競品。類似方向包括：

- NVIDIA Triton Model Analyzer
  - 偏 Triton Inference Server ecosystem。
  - 強在 production serving / GPU deployment。

- ONNX Runtime tuning docs/tools
  - 提供 thread、graph optimization、execution provider 等 tuning knobs。
  - 但不是完整 resource-aware recommender。

- OpenVINO benchmark_app
  - 偏 Intel OpenVINO ecosystem。

- Apache TVM / MetaSchedule
  - 更底層，做 compiler/operator schedule search。
  - 強但複雜，跟 local resource guard 不是同一層。

AutoTuneAI-Serve 的差異定位：

```text
Lightweight local resource-aware optimizer and guard for constrained AI workloads.
```

重點不是取代 Triton / TVM，而是做一個能在 laptop / WSL / small workstation 上跑、能記錄資源限制、能給出 safe recommendation 的系統。

## 14. 還能再做什麼

除了目前提到的功能，還可以擴展：

- Hardware inspector:
  - CPU model
  - logical / physical cores
  - RAM
  - WSL memory cap
  - package versions
  - ONNX Runtime providers

- Real benchmark mini sweep:
  - small search space
  - 先產生第一組真實 PyTorch vs ONNX Runtime 結果

- Recommendation engine:
  - 從 real JSON/CSV 選 safe best config
  - 輸出 reasoning summary

- Resource timeline:
  - 產生 memory / CPU usage time series
  - 幫助判斷什麼時候卡住

- Training wrapper:
  - 包住 `python train.py`
  - 監控並限制 RAM / CPU

- Reversible source tuner:
  - 自動調 batch size / dataloader workers / gradient accumulation
  - 每次修改都有 backup / manifest / restore script

- Report generator:
  - 自動產生 markdown report
  - 包含 tables、plots、recommended config

- Multi-machine comparison:
  - 每台機器跑同一份 config
  - 結果帶 hardware metadata
  - 可以比較 laptop / WSL / server / cloud VM

## 15. 下一步建議

最合理的下一步是：

1. 實作 real benchmark mini sweep。
2. 實作 `recommend_config.py`，從實測結果選 safe best configuration。
3. 接著再做 `run_with_budget.py` training wrapper。
4. 最後才做 reversible source tuner。

原因是 source tuner 風險比較高，必須先有穩定的 monitoring / manifest / restore infrastructure。
