# AutoTuneAI-Serve User Guide

這份文件只寫「目前 repo 真的能跑的東西」和「下一步目標」。如果文件提到 YAML / JSON / Python 範例檔，檔案會放在 repo 裡，可以直接照指令跑。

## 0. 目前做到哪

目前已完成：

- Synthetic inference benchmark。
- PyTorch CPU ResNet18 benchmark。
- ONNX export + ONNX Runtime CPU benchmark。
- Resource-aware benchmark monitoring。
- System inspector，偵測 CPU/RAM/WSL/package/runtime provider/executor capabilities。
- Real mini sweep + safe config recommender。
- Training / arbitrary command resource wrapper。
- systemd/root executor preflight。
- Reversible source tuning transaction。
- Source edit + workload wrapper + auto-restore。
- Batch-size training tuner。

目前還沒完成：

- 真正 ML cost model training。
- INT8 real quantization。
- GPU backend / TensorRT / TVM。
- Plot/report generator。
- AST-level source refactor。
- BIOS / UEFI tuning。

## 1. 環境啟動

從 PowerShell 進 WSL：

```powershell
wsl
```

進入專案：

```bash
cd /mnt/d/School/AutoTuneAI
```

啟用 conda：

```bash
source /home/louis/miniforge3/etc/profile.d/conda.sh
conda activate autotuneai
```

如果是在非互動 shell、script、systemd 或 Codex 這類工具裡執行，建議不要依賴 shell 自動啟用 conda，直接使用完整路徑：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/inspect_system.py
```

如果別人的 conda 裝在不同位置，把 `/home/louis/miniforge3/bin/conda` 換成自己的 conda executable，例如 `/home/<user>/miniforge3/bin/conda`、`/home/<user>/miniconda3/bin/conda`，或 `which conda` 顯示的路徑。

不要用 `sudo su` 進 root shell 後再啟用 conda。root shell 通常不會有使用者的 conda 初始化設定，而且可能讀到舊的 conda 路徑，例如 `/home/louis/miniconda3`。需要 root 權限時，讓 AutoTuneAI 保持在使用者 conda environment 裡執行，只在 systemd/cgroup 那一層使用 `--sudo`。

如果只要使用 resource guard / system inspector / source-safe tuning，建立 core environment：

```bash
/home/louis/miniforge3/bin/conda env create -f environment.yml
```

如果環境存在但 core 套件不完整：

```bash
python -m pip install -r requirements-core.txt
```

如果你要跑 PyTorch / ONNX Runtime real benchmark，再另外安裝 benchmark 依賴：

```bash
python -m pip install -r requirements-benchmark.txt
```

也可以另建 benchmark environment：

```bash
/home/louis/miniforge3/bin/conda env create -f environment-benchmark.yml
```

驗證：

```bash
python -m unittest discover -s tests
```

預期看到：

```text
OK
```

## 2. System Inspector

在跑 benchmark 或 training tuning 前，建議先收集目前機器環境：

```bash
python scripts/inspect_system.py
```

預設會印出 JSON，並寫到：

```text
results/reports/system_info.json
```

如果只想印出，不想寫檔：

```bash
python scripts/inspect_system.py --no-write
```

如果要指定輸出位置：

```bash
python scripts/inspect_system.py --output results/reports/my_system_info.json
```

目前會偵測：

```text
system / release / machine / processor
python_version
is_wsl
cpu_count_logical
cpu_count_physical
cpu_affinity_supported
current_cpu_affinity
total_memory_mb
available_memory_mb
cgroup_memory_max_mb
systemd_run_available
systemd_state
wsl_config_visible
torch / torchvision / onnx / onnxruntime / psutil versions
torch_cuda_available
torch_num_threads
torch_num_interop_threads
onnxruntime_providers
executor_capabilities
system_tuning_recommendations
notes
```

`notes` 不是寫死的固定說明，而是根據偵測結果產生。例如：

- 如果偵測到 WSL，會列出 WSL 內 Linux 可見 RAM。
- 如果 cgroup memory limit 小於可見 RAM，會提醒 cgroup limit。
- 如果 CPU affinity 不支援，會提醒不能用 affinity 保留 core。
- 如果 `systemd-run` 不存在，會提醒 systemd hard-limit executor 不能使用。
- 如果 PyTorch 或 ONNX Runtime 沒安裝，會提醒相關 real benchmark 不能跑。
- 如果 ONNX Runtime 沒有 `CPUExecutionProvider`，會提醒 CPU backend 不可用。
- 如果 systemd 是目前機器最適合的 hard-limit executor，會提醒建議使用 systemd。
- 如果只能使用 local executor，會提醒 hard memory/CPU limit 可能不可用。
- 如果有 runtime system tuning 建議，會提醒看 `system_tuning_recommendations`。

如果只想看 resource executor 能力：

```bash
python scripts/inspect_executors.py
```

如果要實測 systemd transient scope 是否需要 sudo：

```bash
python scripts/inspect_executors.py --probe-systemd --check-sudo-cache
```

在你的 WSL 環境中，目前預期會看到類似：

```json
{
  "platform": "linux-wsl",
  "recommended_executor": "systemd",
  "executors": {
    "local": {
      "available": true,
      "hard_memory_limit": false,
      "process_tree_monitoring": true
    },
    "systemd": {
      "available": true,
      "requires_sudo": true,
      "hard_memory_limit": true,
      "hard_cpu_limit": true,
      "cgroup_monitoring": true
    },
    "docker": {
      "available": false,
      "implemented": false
    },
    "windows_job": {
      "available": false,
      "implemented": false
    }
  }
}
```

跨平台策略目前是：

```text
local
  已實作。
  Windows / macOS / Linux 都可以作為 fallback。
  可監控 process tree；hard memory / hard CPU limit 不保證。

systemd
  已實作。
  Linux / WSL2 + systemd 可用。
  支援 cgroup hard memory / CPU limit 和 scope-level cgroup monitoring。

docker
  已納入 capability detection，但 executor 尚未實作。
  未來可用 Docker container memory / CPU limit，讓 Windows/macOS/Linux 都能用同一種限制模型。

windows_job
  已納入 capability detection，但 executor 尚未實作。
  未來 Windows native 可用 Job Object 限制 process group memory / CPU。

macos
  已納入 capability detection，但 executor 尚未實作。
  macOS 原生 hard memory limit 較弱，短期會以 local monitoring 或 Docker executor 為主。
```

這個檔案很重要，因為同一組 benchmark 數字必須和硬體/環境一起看。之後 report 或跨機器比較都應該附上 `system_info.json`。

## 3. Runtime System Tuning

這一層是 runtime tuning，不是 BIOS / UEFI tuning。它只會考慮白名單內的 Linux runtime sysctl 設定，不會修改 `/etc/sysctl.conf` 這類永久設定檔。

先看推薦，不修改系統：

```bash
python scripts/tune_system.py
```

在 Linux / WSL 上，輸出會包含：

```json
{
  "profile": "linux-training-safe",
  "settings": [
    {
      "key": "vm.swappiness",
      "current": "60",
      "target": "10",
      "would_change": true
    }
  ]
}
```

套用 runtime system tuning：

```bash
sudo -v
python scripts/tune_system.py --apply --sudo
```

套用後會建立 run directory：

```text
.autotuneai/runs/<run_id>/
```

裡面會有：

```text
system_tuning_plan.json
system_tuning_before.json
system_tuning_after.json
system_tuning_diff.json
system_tuning_restore_after.json   # restore 後才會出現
```

`system_tuning_before.json` 是修改前 snapshot，`system_tuning_after.json` 是修改後 snapshot，`system_tuning_diff.json` 是差異，例如：

```json
[
  {
    "key": "vm.swappiness",
    "before": "60",
    "target": "10",
    "after": "10",
    "changed": true,
    "applied": true
  }
]
```

還原 runtime system tuning：

```bash
python scripts/restore_run.py --run-id <run_id> --sudo
```

目前內建 profile：

```text
linux-training-safe
  vm.swappiness=10
    減少 swap 壓力，避免 training 把 RAM 撐爆後系統變得非常卡才反應。

  kernel.numa_balancing=0
    如果這個 setting 存在，降低訓練期間自動 NUMA migration 造成的變動。
    在很多 WSL 環境中這個 setting 不存在，AutoTuneAI 會自動跳過。
```

非 Linux 平台：

```text
Windows / macOS
  tune_system.py 只會輸出 unsupported，不會套用 sysctl。
  後續 Windows 應該走 Windows Job Object 或 Docker executor。
```

## 4. Inference Benchmark

### 4.1 Config 檔在哪

目前 inference benchmark config 放在：

```text
configs/resnet18.yaml
configs/mobilenetv3.yaml
```

`configs/resnet18.yaml` 內容控制：

- model name
- input shape
- search space
- objective
- profiler warmup / repeat
- output path
- ONNX output directory

目前 config 不預設 resource limit。資源限制用 CLI 加，避免把你的機器設定寫死到 shared config。

### 2.2 Synthetic benchmark

Synthetic benchmark 不跑真模型，用公式模擬 latency / throughput / memory，適合快速確認流程。

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode synthetic
```

輸出：

```text
results/raw/resnet18_profile.json
results/raw/resnet18_profile.csv
```

### 2.3 PyTorch CPU real benchmark

```bash
python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --output results/raw/resnet18_pytorch_smoke.json \
  --csv-output results/raw/resnet18_pytorch_smoke.csv
```

### 2.4 ONNX Runtime CPU real benchmark

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

`artifacts/` 和 `results/` 產物已被 `.gitignore` 排除。

## 4. Resource Budget

Resource budget 用 CLI 指定：

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

參數意思：

- `--memory-budget-gb 22`: process peak RSS 希望不超過 22GB。
- `--reserve-cores 1`: 保留 1 個 logical CPU core。
- `--cpu-quota-percent 90`: 根據 CPU 數量限制可用 thread 數，並記錄 CPU 是否超標。

結果 JSON 裡會出現：

```text
peak_rss_mb
memory_budget_mb
effective_memory_budget_mb
available_memory_before_mb
available_memory_after_mb
cpu_affinity_applied
affinity_cores
average_process_cpu_percent
peak_process_cpu_percent
memory_budget_exceeded
cpu_quota_exceeded
```

注意：WSL 可見 RAM 可能比 Windows 少。如果 Windows 顯示 23.7GB，但 WSL 只暴露 11.8GB，`effective_memory_budget_mb` 會用 WSL 實際可用上限計算。

## 5. Real Mini Sweep + Recommender

### 4.1 跑 mini sweep

```bash
python scripts/run_real_sweep.py \
  --config configs/resnet18.yaml \
  --output results/raw/resnet18_real_sweep.json \
  --csv-output results/raw/resnet18_real_sweep.csv
```

預設 search space：

```text
backend: pytorch, onnxruntime
batch size: 1, 2, 4
threads: 1, 2, 4
precision: fp32
ONNX graph optimization: disable, all
```

很小的 smoke test：

```bash
python scripts/run_real_sweep.py \
  --config configs/resnet18.yaml \
  --backends pytorch \
  --batch-sizes 1 \
  --threads 1 \
  --output results/raw/test_real_sweep.json \
  --csv-output results/raw/test_real_sweep.csv
```

### 4.2 推薦 safe config

```bash
python scripts/recommend_config.py \
  --input results/raw/resnet18_real_sweep.json \
  --objective throughput \
  --latency-budget-ms 100 \
  --memory-budget-gb 22
```

支援 objective：

```text
throughput
latency
memory
```

JSON 輸出：

```bash
python scripts/recommend_config.py \
  --input results/raw/resnet18_real_sweep.json \
  --objective throughput \
  --json
```

## 6. Scheduler Simulator

```bash
python scripts/run_scheduler.py \
  --workload burst \
  --scheduler deadline_aware \
  --count 100
```

支援 scheduler：

```text
fcfs
static
dynamic
deadline_aware
```

輸出：

```text
average_latency
p95_latency
p99_latency
throughput
deadline_miss_rate
```

Dynamic batching 主要是 inference serving 的 throughput/latency trade-off，不是 training RAM 爆掉的直接解法。

## 7. Training / Command Resource Wrapper

這個 wrapper 可以包住任意命令，例如 training。AutoTuneAI 可以在自己的 environment 裡執行，training command 可以使用使用者原本的 conda / venv / Python。

推薦做法是直接指定使用者 training environment 的 Python：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  -- /path/to/user/env/bin/python train.py --config configs/train.yaml
```

如果使用 conda，也可以用 `conda run` 包住原本環境：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  -- conda run -n user-train-env python train.py --config configs/train.yaml
```

所以使用者不需要把 training dependencies 裝進 AutoTuneAI 的 environment。

local soft-guard 模式：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90 \
  -- python train.py --config configs/train.yaml
```

auto executor 模式會先跑 capability detection，再選目前機器最合適的 executor：

```bash
python scripts/run_with_budget.py \
  --executor auto \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py --config configs/train.yaml
```

行為規則：

```text
Windows / macOS / no systemd
  auto 會選 local。

Linux / WSL + systemd 可用，且不需要 sudo
  auto 會選 systemd。

Linux / WSL + systemd 可用，但 transient scope 需要 sudo
  auto 會停止並提示你加 --allow-sudo-auto 或改用 --executor systemd --sudo。
```

如果你允許 auto 模式在需要時使用 sudo：

```bash
sudo -v

python scripts/run_with_budget.py \
  --executor auto \
  --allow-sudo-auto \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py --config configs/train.yaml
```

`--allow-sudo-auto` 只代表允許 AutoTuneAI 在 auto 選到 systemd 且需要 sudo 時使用 `sudo systemd-run`。它不會把 training script 改成 root 身分執行；workload 仍會盡量以原使用者身份執行。

每次 run 的 `manifest.json` 會記錄實際選擇：

```text
requested_executor=auto
selected_executor=systemd
executor_platform=linux-wsl
systemd_requires_sudo=True
sudo_used=True
```

目前它會：

- 建立 `.autotuneai/runs/<run_id>/`
- 記錄 `manifest.json`
- 記錄 `before_status.txt`
- 記錄 `before_diff.patch`
- 記錄 `head.txt`
- 執行 child command
- 監控 child process 與子 process RSS / CPU
- 輸出 `resource_timeline.json`
- 輸出 `resource_summary.json`

如果要讓超過 memory budget 時終止 child process：

```bash
python scripts/run_with_budget.py \
  --memory-budget-gb 22 \
  --hard-kill \
  -- python train.py
```

systemd hard-limit 模式：

先做 preflight，確認目前機器是否支援 systemd scope、sudo、MemoryMax、CPUQuota：

```bash
python scripts/check_system_executor.py \
  --probe \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- python train.py
```

如果你打算用 root 權限建立 systemd scope：

```bash
python scripts/check_system_executor.py \
  --sudo \
  --check-sudo-cache \
  --probe \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py
```

`--probe` 會用 `true` 試跑一個短命 systemd scope，不會執行你的 training command。它可以提前抓到 `Interactive authentication required` 這類權限問題。

`--check-sudo-cache` 不會要求你輸入密碼，只會檢查目前 shell 是否已經通過 `sudo -v`。如果顯示 `sudo credential is not cached`，代表你需要先在終端機手動跑一次：

```bash
sudo -v
```

在 WSL 中，常見情況是非 sudo 的 `systemd-run --scope` 會失敗：

```text
Failed to start transient scope unit: Interactive authentication required.
```

這代表 systemd/polkit 不允許一般使用者建立 transient scope。這時不要改成 `sudo su`，而是先在同一個 WSL shell 執行 `sudo -v`，再讓 AutoTuneAI 用 `--sudo` 呼叫 `sudo systemd-run`。

確認後再正式包住訓練入口：

```bash
python scripts/run_with_budget.py \
  --executor systemd \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py
```

有些 Linux/WSL 環境不允許一般使用者建立 transient systemd scope，這時會看到類似：

```text
Interactive authentication required.
```

這代表要改用 `--sudo`，或由系統管理員設定 polkit/systemd 權限。

如果需要 root 權限：

```bash
python scripts/run_with_budget.py \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py
```

`--sudo` 會用 `sudo systemd-run` 建立 scope，但會嘗試讓 workload 仍以原使用者身份執行。這樣 hard limit 由 systemd/cgroup 執行，training script 本身不必變成 root。

如果 AutoTuneAI 自己在 `autotuneai` conda environment，但 training 要跑另一個 conda environment，可以這樣寫：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_with_budget.py \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /home/louis/miniforge3/bin/conda run -n user-train-env python train.py --config configs/train.yaml
```

如果 training environment 是 venv 或固定 Python 路徑，也可以直接指定：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_with_budget.py \
  --executor systemd \
  --sudo \
  --memory-budget-gb 22 \
  --cpu-quota-percent 90 \
  -- /path/to/user/env/bin/python train.py --config configs/train.yaml
```

systemd executor 會建立可辨識的 transient scope，名稱類似：

```text
autotuneai-20260501_120047.scope
```

啟動後 AutoTuneAI 會用 `systemctl show <scope> -p ControlGroup --value` 找到實際 cgroup，然後讀取：

```text
memory.current
memory.peak
cpu.stat
```

因此 systemd 模式的 `resource_timeline.json` 會盡量包含整個 training scope 的 cgroup 使用量，而不是只看 `systemd-run` wrapper process。相關欄位包括：

```text
cgroup_path
cgroup_memory_current_mb
cgroup_memory_peak_mb
cgroup_cpu_percent
cgroup_cpu_usage_usec
```

`resource_summary.json` 也會在拿得到 cgroup stats 時輸出：

```text
peak_cgroup_memory_current_mb
peak_cgroup_memory_peak_mb
average_cgroup_cpu_percent
peak_cgroup_cpu_percent
cgroup_path
```

如果目前系統不能查到 ControlGroup，AutoTuneAI 仍會回退到 process-tree monitoring。要做每個 child process 的詳細拆解，仍然優先用 local executor；要做整個 training scope 的硬限制與總量監控，優先用 systemd executor。

查看歷史 runs：

```bash
python scripts/list_runs.py
```

restore 某個 run 修改過的檔案：

```bash
python scripts/restore_run.py --run-id <run_id>
```

## 8. Reversible Source Tuning

### 7.1 單次安全 find/replace

Dry run，不會修改檔案：

```bash
python scripts/tune_source.py \
  --file train.py \
  --find "batch_size = 64" \
  --replace "batch_size = 16"
```

真的修改：

```bash
python scripts/tune_source.py \
  --file train.py \
  --find "batch_size = 64" \
  --replace "batch_size = 16" \
  --apply
```

限制：

- `find` 文字必須剛好出現一次。
- 出現 0 次或多次會拒絕修改。
- 修改前會備份。
- manifest 會記錄 changed file。
- 可以用 `restore_run.py` 還原。

### 7.2 Source edit + command + auto-restore

先建立 edits JSON。範例檔已提供：

```text
examples/source_edits.json
```

內容格式：

```json
[
  {
    "file": "examples/train_config.yaml",
    "find": "batch_size: 64",
    "replace": "batch_size: 16"
  }
]
```

執行：

```bash
python scripts/run_tuned_with_budget.py \
  --edits-file examples/source_edits.json \
  --sample-interval-seconds 0.05 \
  -- python examples/dummy_train.py
```

流程：

1. 備份 `examples/train_config.yaml`
2. 改成 `batch_size: 16`
3. 執行 `examples/dummy_train.py`
4. 監控 RAM / CPU
5. 結束後自動還原成 `batch_size: 64`

如果刻意不想還原：

```bash
--keep-changes
```

一般不建議使用 `--keep-changes`，除非你明確知道要保留修改。

不要同時對同一個檔案跑多個 tuning command。source tuner 會備份與還原檔案，但目前沒有跨 process file lock；兩個 tuner 同時改同一個檔案會造成競態。

## 9. Batch-size Training Tuner

這是目前最接近「真的幫 training 調參」的功能。

範例 training config：

```text
examples/train_config.yaml
```

內容：

```yaml
batch_size: 64
dataloader_workers: 4
```

範例 training command：

```text
examples/dummy_train.py
```

跑 batch-size tuner：

```bash
python scripts/tune_training_config.py \
  --file examples/train_config.yaml \
  --batch-size-key batch_size \
  --values 32 16 8 \
  --output results/reports/example_training_tuning_summary.json \
  --sample-interval-seconds 0.05 \
  -- python examples/dummy_train.py
```

它會：

1. 讀 `examples/train_config.yaml`
2. 找到 `batch_size: 64`
3. 依序測 `32`, `16`, `8`
4. 每次修改 config 後執行 training command
5. 每次 trial 結束後自動還原 config
6. 寫出 summary JSON
7. 推薦最大的 safe batch size

加入 memory/CPU budget：

```bash
python scripts/tune_training_config.py \
  --file examples/train_config.yaml \
  --batch-size-key batch_size \
  --values 64 32 16 8 \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90 \
  --executor auto \
  --output results/reports/example_training_tuning_summary.json \
  -- python examples/dummy_train.py
```

目前支援的 batch size assignment 格式：

```yaml
batch_size: 64
```

或：

```python
batch_size = 64
```

限制：

- `batch_size` 必須只出現一次。
- 目前只調單一 numeric key。
- 推薦邏輯目前是 largest safe batch size。
- 目前不會自動調 gradient accumulation / dataloader workers。

## 10. 常見輸出位置

```text
results/raw/       benchmark JSON/CSV
results/reports/   recommender/training tuner summary
artifacts/onnx/    exported ONNX model
.autotuneai/runs/  wrapper/tuning run manifests and backups
```

這些產物大多被 `.gitignore` 排除。

## 11. 如果被中斷怎麼辦

列出 runs：

```bash
python scripts/list_runs.py
```

找出 run id 後 restore：

```bash
python scripts/restore_run.py --run-id <run_id>
```

這會根據 `.autotuneai/runs/<run_id>/manifest.json` 裡的 backup 還原被改過的檔案。

不會做：

```bash
git reset --hard
```

原因是不能刪掉使用者原本未 commit 的改動。

## 12. BIOS / UEFI tuning 邊界

這個 tool 主要做 runtime-level tuning。

可以做：

- backend selection
- batch size
- precision
- graph optimization
- thread count
- CPU affinity
- process priority
- memory / CPU monitoring
- reversible source/config tuning
- cgroup / Docker / systemd resource limit，未來擴充

不建議自動做：

- BIOS / UEFI 修改
- SMT / Hyper-Threading 開關
- Turbo Boost 設定
- power limit
- XMP / EXPO
- fan curve
- memory frequency / timing

這些需要重開機，硬體差異大，風險高，不適合做成 portable tool 的自動功能。

## 13. 下一步建議

下一步最值得做：

1. 把 batch-size tuner 的 summary 變成 markdown report。
2. 加 `dataloader_workers` tuner。
3. 加 `gradient_accumulation_steps` tuner。
4. 加 hardware inspector，輸出 CPU/RAM/WSL/package metadata。
5. 把 real benchmark sweep 結果畫成圖。
