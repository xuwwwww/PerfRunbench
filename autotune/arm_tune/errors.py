from __future__ import annotations


class ArmTuneError(Exception):
    """Base class for ArmTune errors."""


class ArmTuneConfigError(ArmTuneError):
    """Raised when ArmTune configuration is invalid."""


class ArmTuneProtocolError(ArmTuneError):
    """Raised when a versioned protocol payload is invalid."""


class TargetValidationError(ArmTuneError):
    """Raised when a target cannot satisfy the requested operation."""


class TargetExecutionError(ArmTuneError):
    """Raised when a target command fails."""
