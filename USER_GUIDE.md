# AutoTuneAI 使用說明

AutoTuneAI 目前是一個「包住訓練或 benchmark 入口」的資源監控、限制、分析與 runtime tuning 工具。它的設計目標不是取代你的訓練環境，而是讓 AutoTuneAI 跑在自己的環境中，再去啟動另一個 Python、conda、venv、shell script 或 Docker container 內的 training command。

## 1. 安裝與基本檢查

建議先在 WSL/Linux 裡建立 AutoTuneAI 自己的環境：

```bash
conda env create -f environment.yml
conda activate autotuneai
python -m pip install -e .
```

如果環境已存在：

```bash
conda activate autotuneai
python -m pip install -r requirements-core.txt
python -m pip install -e .
```

確認 CLI 可用：

```bash
autotuneai --help
python -m autotune.cli --help
```

跑完整測試：

```bash
python -m unittest discover -s tests
```

這個指令會從 `tests/` 目錄自動尋找 `test_*.py`，然後執行全部 unittest。成功時最後會看到 `OK`。

## 2. 不要用 sudo su 進 root shell 跑 AutoTuneAI

正確做法是讓 AutoTuneAI 留在使用者的 conda 環境，需要 root 權限時只透過 `--sudo` 或 `sudo -v` 讓特定 systemd/sysctl 動作用 root 執行。

```bash
sudo -v
autotuneai run --executor systemd --sudo --memory-budget-gb 22 -- python train.py
```

不要這樣做：

```bash
sudo su
conda activate autotuneai
```

原因是 root shell 通常沒有使用者的 conda 初始化，會發生 `conda: command not found` 或指到錯的 Python。AutoTuneAI 只需要 systemd/cgroup/sysctl 這些動作有權限，訓練程式本身不應該因此變成 root 身份執行。

## 3. 單一 CLI 入口

主要指令：

```bash
autotuneai inspect
autotuneai executors
autotuneai run -- <command>
autotuneai analyze --run-id <run_id>
autotuneai report --run-id <run_id>
autotuneai calibrate-memory --budget-gb -5 -3 --workload-memory-mb 2048
autotuneai tune-system
autotuneai tune-system --apply --sudo
autotuneai tune-batch --file train.yaml --key batch_size --values 64 32 16 -- python train.py
autotuneai list-runs
autotuneai restore --run-id <run_id> --sudo
```

舊 script 入口仍保留，例如：

```bash
python scripts/run_with_budget.py -- python train.py
python scripts/tune_training_config.py --file train.yaml --values 64 32 -- python train.py
```

但新使用者建議優先使用 `autotuneai`。

## 4. Executor 類型

查看本機可用能力：

```bash
autotuneai executors --probe-systemd --probe-docker --check-sudo-cache
```

目前 executor：

```text
local
  跨平台 fallback。使用 psutil 做 process-tree 監控。
  CPU affinity 在 Linux 可用；memory limit 是軟限制，搭配 --hard-kill 才會主動終止。

systemd
  Linux/WSL2 + systemd 可用。透過 transient scope + cgroup 做硬 memory/CPU limit。
  常需要 sudo，因此建議先 sudo -v，再加 --sudo。

docker
  透過 docker run --memory/--cpus 做跨平台 container 限制。
  使用者需要準備含有訓練相依套件的 Docker image。

windows_job
  目前只做 capability 規劃，尚未實作。

macos
  目前走 local 或 Docker；macOS 原生 hard memory executor 尚未實作。
```

自動選擇：

```bash
autotuneai run --executor auto --allow-sudo-auto --memory-budget-gb 22 -- python train.py
```

如果 systemd 需要 sudo 而你沒有加 `--allow-sudo-auto`，AutoTuneAI 會停止並提示你明確 opt in。

## 5. 包住另一個訓練環境

AutoTuneAI 不要求使用者把訓練相依套件裝到 AutoTuneAI 的環境。你可以從 AutoTuneAI 環境啟動另一個環境的 Python。

直接指定另一個 venv/conda 的 Python：

```bash
autotuneai run \
  --executor auto \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py --config configs/train.yaml
```

使用 `conda run`：

```bash
autotuneai run \
  --memory-budget-gb 22 \
  -- conda run -n user-train-env python train.py --config configs/train.yaml
```

在 WSL 裡用完整 conda 路徑更穩：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai autotuneai run \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  -- /home/louis/miniforge3/bin/conda run -n user-train-env python train.py
```

這就是「雙環境」模型：AutoTuneAI 的環境只負責監控、限制、記錄和調度；使用者原本的 training environment 負責真正訓練。

## 6. Docker 雙環境用法

如果你要讓別人 clone 後更容易跑，Docker 是最乾淨的方式之一。AutoTuneAI 只需要本機有 Docker CLI/daemon，訓練相依套件放在 image 裡。

```bash
autotuneai run \
  --executor docker \
  --docker-image my-training-image:latest \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- python train.py --config configs/train.yaml
```

Docker executor 會把目前 repo 掛到 container 的 `/workspace`，並在 `/workspace` 內執行命令。

重要限制：

- `python:3.12-slim` 是預設 image，只適合簡單 Python script。
- 真正訓練 PyTorch/Transformers/資料集時，請建立自己的 Docker image。
- Docker executor 目前主要負責硬限制；host process sampling 看到的是 docker client，不等於 container 內所有細節。後續可再補 `docker stats` 或 cgroup path 解析。

## 7. Memory budget

正數代表絕對 GB 上限：

```bash
autotuneai run --memory-budget-gb 22 -- python train.py
```

負數代表「距離 Linux/WSL 可見總記憶體跑滿，至少保留多少 GB」：

```bash
autotuneai run --memory-budget-gb -5 -- python train.py
```

如果 WSL 可見總記憶體是 23.7GB，`-5` 會轉成約 18.7GB 的 effective budget。實際 Windows Task Manager 看到的剩餘值可能不同，因為有 page cache、shared memory、Python allocator、WSL memory reclamation 等因素。

校準本機 memory 行為：

```bash
autotuneai calibrate-memory \
  --budget-gb -5 -3 1 \
  --workload-memory-mb 2048 \
  --duration-seconds 10 \
  --workers 4 \
  --sample-interval-seconds 0.1
```

輸出：

```text
results/reports/memory_calibration.json
```

裡面會記錄每個 budget 的：

```text
effective_budget_mb
peak_memory_mb
observed_min_available_memory_gb
reserve_error_gb
memory_budget_exceeded
run_id
```

如果你要硬限制，Linux/WSL 建議：

```bash
sudo -v
autotuneai run --executor systemd --sudo --memory-budget-gb 22 -- python train.py
```

Docker 建議：

```bash
autotuneai run --executor docker --docker-image my-training-image:latest --memory-budget-gb 22 -- python train.py
```

## 8. CPU 限制與保留核心

保留 1 個 logical core：

```bash
autotuneai run --reserve-cores 1 -- python train.py
```

限制總 CPU 約 90%：

```bash
autotuneai run --cpu-quota-percent 90 -- python train.py
```

兩者一起用時，AutoTuneAI 會取更嚴格的 allowed thread 數。8 logical cores 下：

```text
--reserve-cores 1     => allowed_threads = 7
--cpu-quota-percent 50 => allowed_threads = 4
```

local executor 在 Linux 主要靠 CPU affinity，所以 Windows Task Manager/WSL host UI 可能看到每個 vCPU 都有一些活動，不一定會呈現「某一顆 core 完全 0%」。判斷是否生效請看：

```bash
autotuneai analyze --run-id <run_id>
```

重點欄位：

```text
allowed_threads
affinity_cores
expected_max_total_cpu_percent
observed_peak_process_cpu_percent
```

如果要更硬的 CPU cap，優先用：

```bash
autotuneai run --executor systemd --sudo --cpu-quota-percent 90 -- python train.py
autotuneai run --executor docker --cpu-quota-percent 90 --docker-image my-image -- python train.py
```

## 9. 高壓測試命令

CPU 壓滿，但保留 1 core：

```bash
autotuneai run \
  --executor local \
  --reserve-cores 1 \
  --sample-interval-seconds 0.1 \
  -- python scripts/stress_workload.py --workers 8 --duration-seconds 60 --memory-mb 256
```

CPU 限制 90%：

```bash
autotuneai run \
  --executor local \
  --cpu-quota-percent 90 \
  --sample-interval-seconds 0.1 \
  -- python scripts/stress_workload.py --workers 8 --duration-seconds 60 --memory-mb 256
```

Memory soft guard 測試：

```bash
autotuneai run \
  --executor local \
  --memory-budget-gb 0.5 \
  --hard-kill \
  --sample-interval-seconds 0.05 \
  -- python scripts/stress_workload.py --workers 2 --duration-seconds 30 --memory-mb 2048
```

Systemd hard memory/CPU 測試：

```bash
sudo -v
autotuneai run \
  --executor systemd \
  --sudo \
  --memory-budget-gb 0.5 \
  --cpu-quota-percent 90 \
  --sample-interval-seconds 0.1 \
  -- python scripts/stress_workload.py --workers 8 --duration-seconds 60 --memory-mb 2048
```

## 10. Runtime system tuning

查看建議，不修改系統：

```bash
autotuneai tune-system
```

套用 Linux training-safe profile：

```bash
sudo -v
autotuneai tune-system --apply --sudo
```

目前 profile 會嘗試調整 Linux runtime sysctl，例如：

```text
vm.swappiness
kernel.numa_balancing
vm.dirty_background_ratio
vm.dirty_ratio
vm.zone_reclaim_mode
```

每次套用都會建立 run directory：

```text
.autotuneai/runs/<run_id>/
  system_tuning_plan.json
  system_tuning_before.json
  system_tuning_after.json
  system_tuning_diff.json
```

恢復：

```bash
autotuneai restore --run-id <run_id> --sudo
```

也可以把 system tuning 包進訓練 lifecycle，訓練完自動恢復：

```bash
sudo -v
autotuneai run \
  --tune-system linux-training-safe \
  --system-tuning-sudo \
  --memory-budget-gb 22 \
  -- python train.py
```

也可以改用自動模式，讓 AutoTuneAI 在 Linux/WSL 上選目前建議的 runtime system tuning profile：

```bash
sudo -v
autotuneai run \
  --auto-tune-system \
  --system-tuning-sudo \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  -- python train.py
```

只要 run 有套用 system tuning，run directory 會包含：

```text
system_tuning_before.json
system_tuning_after.json
system_tuning_diff.json
system_tuning_restore_after.json
```

目前可用 profile：

```text
linux-training-safe
  一般訓練用的保守 profile。
  包含 swappiness、dirty page ratio、zone reclaim、NUMA balancing、Transparent Huge Pages madvise。

linux-memory-conservative
  RAM 緊張時使用，會更積極保留訓練記憶體空間。
  包含 swappiness=1、vfs_cache_pressure=200、page-cluster=0、較低 dirty ratio、THP madvise。

linux-throughput
  資料讀取或 checkpoint throughput 優先時使用。
  包含較高 dirty ratio、較低 vfs_cache_pressure、THP madvise。

linux-low-latency
  想降低 flush/THP latency spike 時使用。
  包含較低 dirty ratio、較頻繁 writeback、THP never。
```

自動選擇規則：

```text
--auto-tune-system + 有 memory budget / reserve memory
  -> linux-memory-conservative

--auto-tune-system + 沒有 memory budget
  -> linux-training-safe
```

手動指定更激進的 profile：

```bash
sudo -v
autotuneai run \
  --tune-system linux-low-latency \
  --system-tuning-sudo \
  --executor systemd \
  --sudo \
  --memory-budget-gb -3 \
  -- python train.py
```

`autotuneai report --run-id <run_id>` 的 `Before / After` 區塊會集中顯示 memory start/end/min、peak memory、system tuning snapshots，以及 source/config change 數量。

如果被中斷，仍可用 run id 做 restore：

```bash
autotuneai list-runs
autotuneai restore --run-id <run_id> --sudo
```

BIOS/UEFI、SMT、Turbo Boost、fan curve、XMP/EXPO 這些不是 runtime tool 能安全跨機器修改的範圍。目前 AutoTuneAI 聚焦 runtime-level tuning，原因是可記錄、可回復、可跨機器。

## 11. Reversible source/config tuning

單次 source edit dry run：

```bash
python scripts/tune_source.py \
  --file train.py \
  --find "batch_size = 64" \
  --replace "batch_size = 32"
```

真的套用：

```bash
python scripts/tune_source.py \
  --file train.py \
  --find "batch_size = 64" \
  --replace "batch_size = 32" \
  --apply
```

套用後會在 run manifest 記錄修改前內容。恢復：

```bash
autotuneai restore --run-id <run_id>
```

包住訓練，跑完自動恢復：

```bash
python scripts/run_tuned_with_budget.py \
  --edit train.py "batch_size = 64" "batch_size = 32" \
  --memory-budget-gb 22 \
  -- python train.py
```

不要自動恢復：

```bash
python scripts/run_tuned_with_budget.py \
  --edit train.py "batch_size = 64" "batch_size = 32" \
  --keep-changes \
  -- python train.py
```

## 12. Training config tuner

AutoTuneAI 可以調任意「單一 numeric key」，不是只能調 batch size。

範例 YAML：

```yaml
batch_size: 64
num_workers: 4
gradient_accumulation_steps: 1
```

調 batch size：

```bash
autotuneai tune-batch \
  --file configs/train.yaml \
  --key batch_size \
  --values 128 64 32 16 \
  --memory-budget-gb 22 \
  --executor auto \
  -- python train.py --config configs/train.yaml
```

調 dataloader workers：

```bash
autotuneai tune-batch \
  --file configs/train.yaml \
  --key num_workers \
  --values 0 2 4 8 \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  -- python train.py --config configs/train.yaml
```

調 gradient accumulation：

```bash
autotuneai tune-batch \
  --file configs/train.yaml \
  --key gradient_accumulation_steps \
  --values 1 2 4 \
  -- python train.py --config configs/train.yaml
```

支援的 assignment 格式：

```yaml
batch_size: 64
```

```python
batch_size = 64
```

限制：

- 同一個 key 在檔案中只能出現一次，避免改錯。
- 目前只支援整數 numeric value。
- 每個 trial 都會先改 config、跑訓練、記錄 run、再恢復原檔。
- summary 預設輸出到 `results/reports/training_tuning_summary.json`。

summary 重要欄位：

```text
key
original_value
candidate_values
recommended_value
recommended_run_id
trials[].value
trials[].safe
trials[].reason
trials[].resource_summary
```

## 13. Run artifacts、分析與報告

每次 run 會建立：

```text
.autotuneai/runs/<run_id>/
  manifest.json
  env.json
  before_status.txt
  before_diff.patch
  head.txt
  resource_timeline.json
  resource_summary.json
```

列出 run：

```bash
autotuneai list-runs
```

分析 run：

```bash
autotuneai analyze --run-id <run_id>
autotuneai analyze --run-id <run_id> --json
```

產生 Markdown 報告：

```bash
autotuneai report --run-id <run_id>
```

預設輸出：

```text
.autotuneai/runs/<run_id>/report.md
```

報告會包含：

```text
Summary
Executor
CPU
Memory
Cgroup
Diagnostics
System Tuning Diff
```

## 14. 專案最終目標

最終版希望長成：

```text
使用者 clone repo
建立 AutoTuneAI 環境
autotuneai inspect / executors 看本機能力
autotuneai run 包住自己的 training command
autotuneai tune-batch / tune-system / calibrate-memory 做調教
autotuneai analyze / report 看 before-after 與效果
autotuneai restore 在中斷或失敗後恢復 runtime/source/config
```

希望能支援：

- WSL/Linux systemd hard limits
- Docker cross-platform hard limits
- Windows Job Object native executor
- macOS fallback + Docker
- training config tuning
- reversible source tuning
- runtime sysctl tuning with before/after/diff/restore
- run report 與校準報告

## 15. 市面上類似工具

相近但不完全相同的工具：

```text
systemd-run / cgroups
  能限制資源，但不懂 training workflow、config tuning、報告與恢復。

Docker
  能做 container limit，但不會自動分析 training budget，也不會幫你改 config 後恢復。

Kubernetes / Slurm / Ray
  適合 cluster scheduling，但對單機 WSL/個人工作站太重。

Weights & Biases / MLflow
  擅長 experiment tracking，不是 system/runtime resource guard。

Optuna / Ray Tune
  擅長 hyperparameter tuning，但不是專門做 root/system/cgroup/runtime guard。

PyTorch Profiler
  擅長 profiling，不負責 system tuning、cgroup hard limit 或 source restore。
```

AutoTuneAI 的定位是把「單機訓練入口包住」，同時做資源限制、runtime tuning、config tuning、分析報告與恢復。

## 16. 目前限制

- BIOS/UEFI 不做自動修改，只能提供人工建議。
- local executor 的 memory limit 是監控加 hard-kill，不是 kernel hard cap。
- systemd executor 需要 Linux/WSL2 + systemd，且可能需要 sudo。
- Docker executor 需要使用者提供合適 image。
- Windows Job Object executor 尚未實作。
- config tuner 目前只支援單一檔案、單一 numeric key。
