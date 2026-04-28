from __future__ import annotations

from pathlib import Path

from autotune.utils.config import ensure_parent


def write_summary(title: str, metrics: dict, output: str) -> Path:
    output_path = ensure_parent(output)
    lines = [f"# {title}", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path

