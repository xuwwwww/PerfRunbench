from __future__ import annotations

import argparse
from pathlib import Path

from training_workload import (
    load_flat_config,
    load_iris_records,
    print_metrics_summary,
    split_train_test,
    train_softmax_classifier,
    write_training_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a higher-pressure Iris-based training workload.")
    parser.add_argument("--config", default="examples/stress_train_config.yaml")
    parser.add_argument("--data", default="examples/iris.data")
    args = parser.parse_args()

    config = load_flat_config(args.config)
    records = load_iris_records(Path(args.data))
    train_records, test_records = split_train_test(records)
    metrics = train_softmax_classifier(train_records, test_records, config)
    metrics["dataset"] = "iris-stress"
    metrics["config_path"] = args.config
    write_training_metrics(metrics)
    print_metrics_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
