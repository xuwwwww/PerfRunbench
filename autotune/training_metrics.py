from __future__ import annotations

import json
import sys
from typing import Any, TextIO


STEP_PREFIX = "AUTOTUNEAI_TRAIN_STEP "
RESULT_PREFIX = "AUTOTUNEAI_TRAIN_RESULT "


def emit_step(
    *,
    completed_steps: int,
    metrics: dict[str, Any] | None = None,
    stream: TextIO | None = None,
    prefix: str = STEP_PREFIX,
    **fields: Any,
) -> None:
    """Emit a deterministic training progress record to stdout."""

    payload = {"schema_version": "1.0", "completed_steps": int(completed_steps)}
    if metrics is not None:
        payload["metrics"] = metrics
    payload.update(fields)
    _emit(prefix, payload, stream=stream)


def emit_result(
    *,
    completed_steps: int,
    metrics: dict[str, Any],
    stream: TextIO | None = None,
    prefix: str = RESULT_PREFIX,
    **fields: Any,
) -> None:
    """Emit one final deterministic training result record to stdout."""

    payload = {"schema_version": "1.0", "completed_steps": int(completed_steps), "metrics": metrics}
    payload.update(fields)
    _emit(prefix, payload, stream=stream)


def _emit(prefix: str, payload: dict[str, Any], *, stream: TextIO | None) -> None:
    target = stream or sys.stdout
    target.write(prefix + json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    target.flush()
