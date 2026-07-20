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
    with config_path.open("r", encoding="utf-8-sig") as handle:
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
    entries: list[tuple[int, str, str]] = []
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        entries.append((indent, line.strip(), raw_line))

    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    for index, (indent, line, raw_line) in enumerate(entries):
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else result
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            item_text = line[2:].strip()
            if not item_text:
                item: dict[str, Any] = {}
                parent.append(item)
                stack.append((indent, item))
                continue
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                item = {}
                parent.append(item)
                value = value.strip()
                if value:
                    item[key.strip()] = _parse_scalar(value)
                else:
                    container = _new_child_container(entries, index, indent)
                    item[key.strip()] = container
                    stack.append((indent + 1, container))
                stack.append((indent, item))
                continue
            parent.append(_parse_scalar(item_text))
            continue

        if not isinstance(parent, dict) or ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue
        container = _new_child_container(entries, index, indent)
        parent[key] = container
        stack.append((indent, container))
    return result


def _parse_scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
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


def _new_child_container(entries: list[tuple[int, str, str]], index: int, indent: int) -> dict[str, Any] | list[Any]:
    for next_indent, next_line, _raw_line in entries[index + 1 :]:
        if next_indent <= indent:
            break
        return [] if next_line.startswith("- ") else {}
    return {}


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line
