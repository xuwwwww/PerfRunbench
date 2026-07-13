from __future__ import annotations

import json

from autotune.arm_tune.schemas import CpuTopology, UNKNOWN


def parse_lscpu_topology(text: str) -> CpuTopology:
    values = _lscpu_values(text)
    if not values:
        return CpuTopology(source=UNKNOWN)
    caches = {
        key: value
        for key, value in values.items()
        if key.lower().startswith("l1") or key.lower().startswith("l2") or key.lower().startswith("l3")
    }
    return CpuTopology(
        logical_cpus=_optional_int(values.get("CPU(s)")),
        cores_per_socket=_optional_int(values.get("Core(s) per socket")),
        sockets=_optional_int(values.get("Socket(s)")),
        threads_per_core=_optional_int(values.get("Thread(s) per core")),
        numa_nodes=_optional_int(values.get("NUMA node(s)")),
        online_cpus=values.get("On-line CPU(s) list", UNKNOWN),
        caches=caches,
        source="lscpu --json",
    )


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
        data = row.get("data")
        if field:
            values[field] = str(data) if data is not None else UNKNOWN
    return values


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).split()[0])
    except (ValueError, IndexError):
        return None
