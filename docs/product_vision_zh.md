# AutoTuneAI-Serve 最終目標與定位

## 1. 最終版想解決什麼問題

AutoTuneAI-Serve 的目標不是單純跑 benchmark，而是做成一個 resource-aware AI inference optimizer。

它要回答這個問題：

> 在這台機器的 RAM、CPU、latency、throughput 限制下，哪個 model runtime configuration 最值得用？

使用者輸入：

- model
- backend candidates
- batch size range
- thread count range
- precision candidates
- latency budget
- memory budget
- reserve CPU cores
- CPU quota
- workload pattern

系統輸出：

- recommended backend
- recommended batch size
- recommended precision
- recommended thread count
- recommended graph optimization level
- predicted latency / measured latency
- throughput
- peak memory
- CPU usage
- resource safety status
- reasoning summary

## 2. 最終版希望怎麼被使用

### 2.1 Local laptop / WSL optimization

使用者只有有限 RAM 和 CPU，希望不要把整台機器卡死。

範例：

```bash
python scripts/recommend_config.py \
  --model resnet18 \
  --objective throughput \
  --memory-budget-gb 22 \
  --reserve-cores 1 \
  --cpu-quota-percent 90
```

輸出概念：

```text
Best safe configuration
Backend: onnxruntime
Batch size: 4
Threads: 4
Precision: fp32
Graph optimization: all

Reasoning:
- Highest throughput among safe configurations.
- Peak RSS stayed below effective memory budget.
- p95 latency stayed within target.
- CPU affinity reserved one logical core.
```

### 2.2 Server inference deployment

使用者要部署 inference service，想知道 batch size / backend / threads 怎麼設。

目標：

- maximize throughput
- keep p95 latency below target
- avoid memory pressure
- avoid using all CPU cores

### 2.3 Experiment/report generation

一條指令產生 technical report 所需資料：

```bash
python scripts/run_real_sweep.py --config configs/resnet18.yaml
python scripts/recommend_config.py --input results/raw/resnet18_real_sweep.json
python scripts/generate_report.py --input results/raw/resnet18_real_sweep.json
```

輸出：

- tables
- JSON
- CSV
- figures
- markdown report

## 3. 跟市面上工具的關係

這個 project 不是要取代大型 production framework，而是做一個小而清楚、可研究、可展示、可在 local machine 跑的 optimizer。

類似工具包括：

- NVIDIA Triton Inference Server
  - 有 dynamic batching、concurrent model execution、model configuration optimization。
  - 更偏 production inference serving。
  - 官方文件：https://docs.nvidia.com/deeplearning/triton-inference-server/

- NVIDIA Triton Model Analyzer
  - 會尋找 batch size、instance count 等 deployment configuration。
  - 更偏 GPU / Triton deployment。
  - 官方介紹：https://developer.nvidia.com/blog/fast-and-scalable-ai-model-deployment-with-nvidia-triton-inference-server/

- ONNX Runtime performance tuning
  - 提供 graph optimization、threading、execution provider 等 performance knobs。
  - 官方文件：https://onnxruntime.ai/docs/performance/tune-performance/

- Apache TVM / MetaSchedule
  - 搜尋低階 compiler schedule，做更深層的 operator/compiler auto-tuning。
  - 官方文件：https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html

- OpenVINO
  - Intel 生態下的 model optimization / inference runtime。
  - 官方網站：https://www.openvino.ai/

AutoTuneAI-Serve 的差異定位：

- 更適合做 academic/project demonstration。
- 重點是清楚呈現 cost model、configuration search、resource budget、scheduler simulation。
- 可以跑在 local WSL / CPU-only machine。
- 不綁定特定 production server。
- 可以把使用者自己的 RAM/CPU 限制作為 first-class constraint。

## 4. Dynamic batching 對 RAM 問題的真實作用

Dynamic batching 主要解決的是 inference serving throughput，不是 training memory。

有效情境：

- 多個 inference request 同時進來。
- 單次 request 太小，CPU/GPU utilization 不高。
- 可以接受一點 batching wait time。

可能沒效或有害的情境：

- 單一 request workload。
- deadline 很緊。
- batch 變大導致 memory pressure。
- training workload。

所以最終版應該做的是 memory-aware dynamic batching：

- 根據目前 available memory 決定 batch 是否還能變大。
- 如果接近 memory budget，就提前 dispatch 或縮小 batch。
- 同時觀察 deadline miss rate。

## 5. Training memory guard 能做到什麼

這個 project 目前主軸是 inference，但 resource guard 的概念也可以套到 training script wrapper。

可以做到：

- benchmark 前檢查 available RAM。
- 記錄 training/inference process peak RSS。
- 用 CPU affinity 保留 core。
- 用 thread count 控制 PyTorch / ONNX Runtime CPU usage。
- 用 cgroup 或 systemd 做 hard memory limit。

短期先做：

- soft guard
- measurement
- recommendation

後續可做：

```bash
python scripts/run_with_budget.py \
  --memory-max-gb 22 \
  --reserve-cores 1 \
  -- python train.py --config ...
```

## 6. BIOS / UEFI tuning 的邊界

這個 tool 主要做 runtime-level tuning，不會直接修改 BIOS/UEFI。

可以做的：

- backend selection
- batch size
- precision
- thread count
- graph optimization
- CPU affinity
- process-level memory monitoring
- scheduler policy
- cgroup / Docker / systemd resource limits

通常不應該由這個 tool 自動做的：

- 修改 BIOS/UEFI 設定
- 改 CPU turbo boost
- 改 SMT / hyper-threading
- 改 memory profile / XMP / EXPO
- 改 power limits
- 改 fan curve

原因：

- BIOS/UEFI 設定跨主機差異很大。
- 需要重開機。
- 有穩定性風險。
- 不適合做成一般使用者自動化工具。

但 final report 可以把 BIOS/UEFI 當作 limitation 或 controlled variable：

```text
All experiments are runtime-level optimizations. BIOS/UEFI settings, power limits, and memory frequency are treated as fixed hardware conditions.
```

## 7. 如何跑在不只一台機器上

要支援不同機器，最終版需要：

- hardware metadata collection
  - CPU model
  - logical / physical cores
  - RAM
  - OS
  - WSL or native Linux
  - Python / package versions

- portable configs
  - 不寫死本機路徑
  - 使用 relative paths
  - 支援 CPU-only fallback

- environment files
  - `environment.yml`
  - `requirements.txt`
  - later: Dockerfile

- benchmark result schema
  - 每筆結果都帶 hardware metadata
  - 不同機器結果可以比較，但不混為同一個 model

最終可以有：

```bash
python scripts/collect_hardware.py
python scripts/run_real_sweep.py --config configs/resnet18.yaml
python scripts/recommend_config.py --input results/raw/resnet18_real_sweep.json
```

這樣同一份 repo 可以在 laptop、WSL、lab server、cloud VM 上跑。

