from __future__ import annotations


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def mean_absolute_percentage_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    return 100.0 * sum(abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0) / len(actual)

