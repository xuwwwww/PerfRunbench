from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.targets.protocol import atomic_write_json


def write_text_artifact(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", errors="replace", newline="\n")
    return target


def artifact_record(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"label": label, "path": str(target), "exists": False}
    return {
        "label": label,
        "path": str(target),
        "exists": True,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def write_artifact_manifest(path: str | Path, artifacts: dict[str, dict[str, Any]]) -> Path:
    return atomic_write_json(path, {"schema_version": "1.0", "artifacts": artifacts})
