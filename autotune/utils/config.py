from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on minimal systems
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            if yaml is not None:
                return yaml.safe_load(handle) or {}
            return _load_simple_yaml(handle.read())
        if config_path.suffix == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _load_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1]
            current_section = {}
            result[key] = current_section
            continue
        if line.startswith("  ") and current_section is not None:
            key, value = line.strip().split(":", 1)
            current_section[key] = _parse_scalar(value.strip())
            continue
        raise ValueError(f"Unsupported YAML line: {raw_line}")
    return result


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
