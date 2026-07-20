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


def artifact_record(path: str | Path, *, label: str, root: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    record: dict[str, Any] = {"label": label, "path": str(target)}
    if root is not None:
        try:
            record["relative_path"] = target.resolve(strict=False).relative_to(Path(root).resolve(strict=False)).as_posix()
        except ValueError:
            record["relative_path"] = ""
    if not target.exists():
        record["exists"] = False
        return record
    record.update(
        {
            "exists": True,
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    )
    return record


def write_artifact_manifest(path: str | Path, artifacts: dict[str, dict[str, Any]]) -> Path:
    return atomic_write_json(path, {"schema_version": "1.0", "artifacts": artifacts})
