# Dependency-Free Training Fixture

This fixture is a deterministic public workload for the `pytorch-training`
adapter tests and demos. It does not import PyTorch, download data, require a
GPU, or access the network.

Example:

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai run --auto --config autotune.yaml -- python examples/training_fixture/train_fixture.py
```
