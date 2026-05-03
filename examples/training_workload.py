from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable


def load_flat_config(path: str | Path) -> dict[str, object]:
    config: dict[str, object] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [item.strip() for item in line.split(":", 1)]
        config[key] = _parse_scalar(value)
    return config


def load_iris_records(path: str | Path) -> list[tuple[list[float], int]]:
    labels = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    records: list[tuple[list[float], int]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        features = [float(item) for item in parts[:4]]
        records.append((features, labels[parts[4]]))
    if not records:
        raise RuntimeError(f"no iris records found in {path}")
    return _normalize_records(records)


def expand_features(features: list[float], multiplier: int) -> list[float]:
    if multiplier <= 1:
        return list(features)
    return [value for value in features for _ in range(multiplier)]


def split_train_test(records: list[tuple[list[float], int]], seed: int = 13) -> tuple[list, list]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8))
    return shuffled[:split], shuffled[split:]


def train_softmax_classifier(
    train_records: list[tuple[list[float], int]],
    test_records: list[tuple[list[float], int]],
    config: dict[str, object],
) -> dict[str, object]:
    batch_size = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 8))
    learning_rate = float(config.get("learning_rate", 0.05))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    feature_multiplier = int(config.get("feature_multiplier", 1))
    cpu_burn_per_batch = int(config.get("cpu_burn_per_batch", 0))
    memory_padding_mb = int(config.get("memory_padding_mb", 0))
    memory_target_mb = int(config.get("memory_target_mb", 0))
    preload_copies = int(config.get("preload_copies", 1))
    min_duration_seconds = float(config.get("min_duration_seconds", 0.0))
    max_duration_seconds = float(config.get("max_duration_seconds", 0.0))

    expanded_train = [
        (expand_features(features, feature_multiplier), label)
        for features, label in train_records
    ]
    expanded_test = [(expand_features(features, feature_multiplier), label) for features, label in test_records]
    cached_train_views = [list(expanded_train) for _ in range(max(1, preload_copies))]

    feature_count = len(expanded_train[0][0])
    classes = 3
    weights = [[0.0 for _ in range(feature_count)] for _ in range(classes)]
    bias = [0.0 for _ in range(classes)]
    padding = [0] * max(0, memory_padding_mb * 256 * 1024)
    memory_target = bytearray(max(0, memory_target_mb * 1024 * 1024))
    _ = cached_train_views[0][0]
    batch_payload_values: list[float] = []
    epoch_times: list[float] = []
    step_times: list[float] = []
    loss_history: list[float] = []
    optimizer_steps = 0
    total_samples = 0
    completed_epochs = 0

    overall_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - overall_start
        if completed_epochs >= epochs and elapsed >= min_duration_seconds:
            break
        if max_duration_seconds > 0 and elapsed >= max_duration_seconds:
            break
        epoch_start = time.perf_counter()
        grad_w = [[0.0 for _ in range(feature_count)] for _ in range(classes)]
        grad_b = [0.0 for _ in range(classes)]
        accumulation = 0
        random.shuffle(expanded_train)
        for batch_index in range(0, len(expanded_train), batch_size):
            batch_start = time.perf_counter()
            batch = expanded_train[batch_index : batch_index + batch_size]
            total_samples += len(batch)
            batch_payload_values.append((len(batch) * feature_count * 8) / (1024 * 1024))
            batch_loss = 0.0
            for features, label in batch:
                logits = [dot(weights[class_index], features) + bias[class_index] for class_index in range(classes)]
                probabilities = softmax(logits)
                batch_loss -= math.log(max(probabilities[label], 1e-9))
                for class_index in range(classes):
                    error = probabilities[class_index] - (1.0 if class_index == label else 0.0)
                    grad_b[class_index] += error
                    for feature_index, value in enumerate(features):
                        grad_w[class_index][feature_index] += error * value
            loss_history.append(batch_loss / max(1, len(batch)))
            accumulation += 1
            if accumulation >= gradient_accumulation_steps:
                scale = learning_rate / max(1, len(batch) * accumulation)
                for class_index in range(classes):
                    bias[class_index] -= scale * grad_b[class_index]
                    for feature_index in range(feature_count):
                        weights[class_index][feature_index] -= scale * grad_w[class_index][feature_index]
                        grad_w[class_index][feature_index] = 0.0
                    grad_b[class_index] = 0.0
                accumulation = 0
                optimizer_steps += 1
            if cpu_burn_per_batch > 0:
                burn_cpu(cpu_burn_per_batch, feature_count)
            _ = padding[:1]
            _ = memory_target[:1]
            step_times.append(time.perf_counter() - batch_start)
        if accumulation:
            scale = learning_rate / max(1, batch_size * accumulation)
            for class_index in range(classes):
                bias[class_index] -= scale * grad_b[class_index]
                for feature_index in range(feature_count):
                    weights[class_index][feature_index] -= scale * grad_w[class_index][feature_index]
            optimizer_steps += 1
        epoch_times.append(time.perf_counter() - epoch_start)
        completed_epochs += 1

    duration_seconds = time.perf_counter() - overall_start
    accuracy = evaluate_accuracy(expanded_test, weights, bias)
    return {
        "duration_seconds": round(duration_seconds, 6),
        "epoch_time_mean_seconds": round(sum(epoch_times) / max(1, len(epoch_times)), 6),
        "epoch_time_max_seconds": round(max(epoch_times), 6),
        "step_time_mean_seconds": round(sum(step_times) / max(1, len(step_times)), 6),
        "samples_per_second": round(total_samples / max(duration_seconds, 1e-9), 3),
        "final_accuracy": round(accuracy, 4),
        "final_loss": round(loss_history[-1] if loss_history else 0.0, 6),
        "optimizer_steps": optimizer_steps,
        "completed_epochs": completed_epochs,
        "feature_count": feature_count,
        "train_samples": len(expanded_train),
        "test_samples": len(expanded_test),
        "peak_batch_payload_mb": round(max(batch_payload_values) if batch_payload_values else 0.0, 6),
        "cache_copies": preload_copies,
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "feature_multiplier": feature_multiplier,
            "cpu_burn_per_batch": cpu_burn_per_batch,
            "memory_padding_mb": memory_padding_mb,
            "memory_target_mb": memory_target_mb,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "preload_copies": preload_copies,
        },
    }


def write_training_metrics(metrics: dict[str, object], *, filename: str = "training_metrics.json") -> Path | None:
    run_dir = os.environ.get("AUTOTUNEAI_RUN_DIR")
    if not run_dir:
        return None
    path = Path(run_dir) / filename
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_metrics_summary(metrics: dict[str, object]) -> None:
    print(
        "training completed "
        f"accuracy={metrics.get('final_accuracy')} "
        f"loss={metrics.get('final_loss')} "
        f"samples_per_second={metrics.get('samples_per_second')}"
    )


def dot(weights: Iterable[float], features: Iterable[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exps = [math.exp(item - maximum) for item in logits]
    total = sum(exps)
    return [item / total for item in exps]


def evaluate_accuracy(records: list[tuple[list[float], int]], weights: list[list[float]], bias: list[float]) -> float:
    correct = 0
    for features, label in records:
        logits = [dot(weights[class_index], features) + bias[class_index] for class_index in range(len(weights))]
        prediction = max(range(len(weights)), key=lambda idx: logits[idx])
        if prediction == label:
            correct += 1
    return correct / max(1, len(records))


def burn_cpu(iterations: int, feature_count: int) -> float:
    value = 0.0
    limit = max(1, iterations)
    for index in range(limit):
        angle = ((index % max(1, feature_count)) + 1) / max(1, feature_count)
        value += math.sin(angle) * math.cos(angle * 2.0)
    return value


def _parse_scalar(value: str) -> object:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _normalize_records(records: list[tuple[list[float], int]]) -> list[tuple[list[float], int]]:
    feature_count = len(records[0][0])
    mins = [min(features[index] for features, _label in records) for index in range(feature_count)]
    maxs = [max(features[index] for features, _label in records) for index in range(feature_count)]
    normalized: list[tuple[list[float], int]] = []
    for features, label in records:
        scaled = []
        for index, value in enumerate(features):
            low = mins[index]
            high = maxs[index]
            denom = high - low
            scaled.append(0.0 if denom == 0 else (value - low) / denom)
        normalized.append((scaled, label))
    return normalized
