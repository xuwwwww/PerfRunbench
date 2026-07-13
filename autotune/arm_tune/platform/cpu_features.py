from __future__ import annotations

import json

from autotune.arm_tune.schemas import ArmHostCapabilities, UNKNOWN


def parse_cpuinfo_features(text: str) -> set[str]:
    features: set[str] = set()
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key.strip().lower() in {"features", "flags"}:
            features.update(item.strip().lower() for item in value.split() if item.strip())
    return features


def parse_lscpu_features(text: str) -> set[str]:
    if not text.strip():
        return set()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return set()
    rows = payload.get("lscpu") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field", "")).rstrip(":").lower()
        if field in {"flags", "features"}:
            return {item.strip().lower() for item in str(row.get("data", "")).split() if item.strip()}
    return set()


def capabilities_from_features(features: set[str], *, source: str) -> ArmHostCapabilities:
    clean = sorted(features)
    return ArmHostCapabilities(
        features=clean,
        neon=_has_any(features, "asimd", "neon"),
        dotprod=_has_any(features, "asimddp", "dotprod"),
        i8mm=_has_any(features, "i8mm", "svei8mm"),
        bf16=_has_any(features, "bf16", "svebf16"),
        sve=_has_any(features, "sve"),
        sve2=_has_any(features, "sve2"),
        sme=_has_any(features, "sme"),
        sme2=_has_any(features, "sme2"),
        evidence_source=source if clean else UNKNOWN,
    )


def merge_features(cpuinfo_text: str, lscpu_text: str) -> ArmHostCapabilities:
    cpuinfo = parse_cpuinfo_features(cpuinfo_text)
    lscpu = parse_lscpu_features(lscpu_text)
    features = cpuinfo | lscpu
    source = "cpuinfo+lscpu" if cpuinfo and lscpu else "/proc/cpuinfo" if cpuinfo else "lscpu --json" if lscpu else UNKNOWN
    return capabilities_from_features(features, source=source)


def _has_any(features: set[str], *names: str) -> bool | None:
    if not features:
        return None
    return any(name in features for name in names)
