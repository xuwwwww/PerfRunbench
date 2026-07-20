from __future__ import annotations


class TrainingAdapterError(RuntimeError):
    """Base error for the PyTorch training adapter."""


class TrainingConfigError(TrainingAdapterError, ValueError):
    """Raised when a training adapter configuration is invalid."""


class TrainingBindingError(TrainingAdapterError):
    """Raised when an allowlisted knob binding cannot be applied safely."""


class TrainingMetricError(TrainingAdapterError):
    """Raised when training metrics are missing or invalid."""
