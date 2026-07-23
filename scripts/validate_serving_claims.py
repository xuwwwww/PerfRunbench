from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_SNIPPETS = [
    "17.032",
    "31.355",
    "+84.092%",
    "3055.682",
    "271.088",
    "91.128%",
    "17.046",
    "31.790",
    "+86.490%",
    "10126.795",
    "202.424",
    "98.001%",
    "16.563",
    "28.054",
    "+69.375%",
    "3967.424",
    "341.693",
    "91.388%",
    "15.334",
    "28.295",
    "+84.529%",
    "608",
    "network_tuning_not_executed_evidence_absent",
]

FILES_WITH_FULL_MATRIX = [
    ROOT / "docs" / "SERVING_AUTOTUNING.md",
    ROOT / "docs" / "results" / "VALIDATED_PERFORMANCE_SUMMARY.md",
    ROOT / "docs" / "results" / "PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md",
    ROOT / "docs" / "releases" / "v0.3.0.md",
]

FILES_WITH_J1_MATRIX = [
    ROOT / "README.md",
    ROOT / "docs" / "demo" / "SERVING_AUTOTUNING_DEMO.md",
]

PUBLIC_FILES = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "SERVING_AUTOTUNING.md",
    ROOT / "docs" / "results" / "VALIDATED_PERFORMANCE_SUMMARY.md",
    ROOT / "docs" / "results" / "PHASE_21D3_SERVING_AUTOTUNING_PRODUCTIZATION.md",
    ROOT / "docs" / "results" / "PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md",
    ROOT / "docs" / "demo" / "SERVING_AUTOTUNING_DEMO.md",
    ROOT / "docs" / "demo" / "SERVING_AUTOTUNING_COMMANDS.md",
    ROOT / "docs" / "demo" / "SERVING_AUTOTUNING_TALK_TRACK.md",
    ROOT / "docs" / "releases" / "v0.3.0.md",
]

J1_SNIPPETS = [snippet for snippet in CANONICAL_SNIPPETS if snippet not in {"15.334", "28.295", "+84.529%"}]

PRIVATE_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(r"\.pem\b", re.IGNORECASE),
    re.compile(r"PRIVATE KEY", re.IGNORECASE),
    re.compile(r"\b(public_ipv4|private_ipv4|HostKey|ssh_config)\b", re.IGNORECASE),
    re.compile(r"/home/ubuntu/models", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\(?:Users|School|CodexHome|Downloads)\\", re.IGNORECASE),
]

FORBIDDEN_CLAIMS = [
    "network tuning improved",
    "universal best llama-server",
    "is a global optimum",
    "as a global optimum",
    "64k performance improvement verified: true",
    "compiler improvement verified: true",
]


def main() -> int:
    failures: list[str] = []
    for path in FILES_WITH_FULL_MATRIX:
        failures.extend(_require(path, CANONICAL_SNIPPETS))
    for path in FILES_WITH_J1_MATRIX:
        failures.extend(_require(path, J1_SNIPPETS))
    for path in PUBLIC_FILES:
        text = _read(path)
        failures.extend(_private_scan(path, text))
        lower = text.lower()
        for claim in FORBIDDEN_CLAIMS:
            if claim in lower:
                failures.append(f"{path}: forbidden claim {claim!r}")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("serving_claim_consistency: passed")
    return 0


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _require(path: Path, snippets: list[str]) -> list[str]:
    text = _read(path)
    return [f"{path}: missing canonical snippet {snippet!r}" for snippet in snippets if snippet not in text]


def _private_scan(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(text):
            failures.append(f"{path}: private-looking pattern {pattern.pattern!r}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
