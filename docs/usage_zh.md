# AutoTuneAI-Serve 使用說明

這份文件說明目前版本怎麼在 WSL conda 裡使用。

## 1. 進入 WSL

你在 PowerShell 看到的 `conda env list` 是 Windows 的 conda，不是 WSL 裡的 conda。

目前專案環境建立在 WSL 的 miniforge：

```bash
/home/louis/miniforge3/envs/autotuneai
```

從 PowerShell 進入 WSL：

```powershell
wsl
```

進入專案：

```bash
cd /mnt/d/School/AutoTuneAI
```

查看 WSL 裡的 conda env：

```bash
/home/louis/miniforge3/bin/conda info --envs
```

啟用環境：

```bash
source /home/louis/miniforge3/etc/profile.d/conda.sh
conda activate autotuneai
```

如果 conda 已經初始化到 shell，也可以直接：

```bash
conda activate autotuneai
```

## 2. 如果環境不存在，重新建立

```bash
cd /mnt/d/School/AutoTuneAI
/home/louis/miniforge3/bin/conda env create -f environment.yml
```

如果環境已存在但套件不完整：

```bash
cd /mnt/d/School/AutoTuneAI
/home/louis/miniforge3/bin/conda run -n autotuneai python -m pip install -r requirements.txt
```

## 3. 快速驗證

```bash
cd /mnt/d/School/AutoTuneAI
/home/louis/miniforge3/bin/conda run -n autotuneai python -m unittest discover -s tests
```

應該看到類似：

```text
Ran 6 tests
OK
```

## 4. 跑 synthetic benchmark

synthetic benchmark 不跑真模型，用公式模擬 latency / throughput / memory，適合快速測 CLI、tuner、資料格式。

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode synthetic
```

輸出：

```text
results/raw/resnet18_profile.json
results/raw/resnet18_profile.csv
```

## 5. 跑 PyTorch CPU 真實 benchmark

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --output results/raw/resnet18_pytorch_smoke.json \
  --csv-output results/raw/resnet18_pytorch_smoke.csv
```

目前 real mode 先支援 `fp32`，`int8` 之後會做。

## 6. 跑 ONNX Runtime CPU 真實 benchmark

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends onnxruntime \
  --max-configs 1 \
  --output results/raw/resnet18_onnx_smoke.json \
  --csv-output results/raw/resnet18_onnx_smoke.csv
```

第一次跑 ONNX Runtime 時，會自動 export：

```text
artifacts/onnx/resnet18.onnx
```

這個檔案不會進 git。

WSL 裡 ONNX Runtime 可能印 GPU discovery warning，但目前指定的是 CPU provider，所以不影響 CPU benchmark。

## 7. Resource-aware benchmark

目前 config 裡有：

```yaml
resource_budget:
  memory_budget_gb: 22
  reserve_memory_gb: 1.7
  reserve_cores: 1
  cpu_quota_percent: 90
  enforce: true
```

意思是：

- 理想上最多使用 22GB memory
- 至少保留 1.7GB 給系統
- 至少保留 1 個 CPU core
- 只允許使用約 90% CPU capacity
- benchmark 時會用 CPU affinity 限制 process 使用的 cores

你也可以用 CLI 覆蓋：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_benchmark.py \
  --config configs/resnet18.yaml \
  --mode real \
  --backends pytorch \
  --max-configs 1 \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90
```

結果會包含：

```text
peak_rss_mb
memory_budget_mb
effective_memory_budget_mb
available_memory_before_mb
available_memory_after_mb
average_process_cpu_percent
peak_process_cpu_percent
average_system_cpu_percent
peak_system_cpu_percent
memory_budget_exceeded
cpu_quota_exceeded
affinity_cores
```

注意：如果 WSL 只分配到 12GB RAM，即使你設定 `memory_budget_gb: 22`，真正安全上限也會變成 WSL 可見 RAM 扣掉 reserved memory。這就是 `effective_memory_budget_mb` 的用途。

## 8. Auto-tuning demo

目前 tuner 先使用 synthetic profiler 和 synthetic cost model：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_autotune.py \
  --config configs/resnet18.yaml \
  --search cost_model \
  --trials 18
```

它會比較 search space，輸出推薦 configuration。

## 9. Scheduler simulator

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_scheduler.py \
  --workload burst \
  --scheduler deadline_aware \
  --count 100
```

目前支援：

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

## 10. 常用指令

查看 git 狀態：

```bash
git status
```

查看最近 commit：

```bash
git log --oneline -5
```

跑全部目前 demo：

```bash
/home/louis/miniforge3/bin/conda run -n autotuneai python scripts/run_all_experiments.py
```

