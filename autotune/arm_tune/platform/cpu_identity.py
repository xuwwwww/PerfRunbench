from __future__ import annotations

import json
from typing import Any

from autotune.arm_tune.schemas import ArmCpuIdentity, UNKNOWN


def parse_cpuinfo_identity(text: str) -> ArmCpuIdentity:
    records = _parse_cpuinfo_records(text)
    first = records[0] if records else {}
    return ArmCpuIdentity(
        vendor=_first_known(first, "vendor_id", "CPU implementer"),
        model_name=_first_known(first, "model name", "Processor"),
        architecture=_first_known(first, "CPU architecture"),
        implementer=_first_known(first, "CPU implementer"),
        variant=_first_known(first, "CPU variant"),
        part=_first_known(first, "CPU part"),
        revision=_first_known(first, "CPU revision"),
        source="/proc/cpuinfo" if records else UNKNOWN,
    )


def parse_lscpu_identity(text: str) -> ArmCpuIdentity:
    values = _lscpu_values(text)
    if not values:
        return ArmCpuIdentity(source=UNKNOWN)
    return ArmCpuIdentity(
        vendor=values.get("Vendor ID", UNKNOWN),
        model_name=values.get("Model name", UNKNOWN),
        architecture=values.get("Architecture", UNKNOWN),
        implementer=values.get("CPU implementer", UNKNOWN),
        variant=values.get("CPU variant", UNKNOWN),
        part=values.get("CPU part", UNKNOWN),
        revision=values.get("CPU revision", UNKNOWN),
        source="lscpu --json",
    )


def merge_cpu_identity(cpuinfo: ArmCpuIdentity, lscpu: ArmCpuIdentity) -> ArmCpuIdentity:
    return ArmCpuIdentity(
        vendor=_prefer(cpuinfo.vendor, lscpu.vendor),
        model_name=_prefer(lscpu.model_name, cpuinfo.model_name),
        architecture=_prefer(cpuinfo.architecture, lscpu.architecture),
        implementer=_prefer(cpuinfo.implementer, lscpu.implementer),
        variant=_prefer(cpuinfo.variant, lscpu.variant),
        part=_prefer(cpuinfo.part, lscpu.part),
        revision=_prefer(cpuinfo.revision, lscpu.revision),
        source="cpuinfo+lscpu" if cpuinfo.source != UNKNOWN or lscpu.source != UNKNOWN else UNKNOWN,
    )


def _parse_cpuinfo_records(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        records.append(current)
    return records


def _lscpu_values(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    rows = payload.get("lscpu") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    values: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field", "")).rstrip(":")
        data: Any = row.get("data")
        if field:
            values[field] = str(data) if data is not None else UNKNOWN
    return values


def _first_known(record: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return UNKNOWN


def _prefer(primary: str, fallback: str) -> str:
    return primary if primary != UNKNOWN else fallback
