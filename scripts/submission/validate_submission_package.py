"""Validate the public competition submission package."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "docs" / "submission"
DECK = SUBMISSION / "arm-ai-challenge-deck"
DEVPOST_ASSETS = SUBMISSION / "assets" / "devpost"

EXPECTED_FILES = (
    ROOT / "LICENSE",
    SUBMISSION / "DEVPOST_FINAL.md",
    SUBMISSION / "PROJECT_FACTS.md",
    SUBMISSION / "SUBMISSION_CHECKLIST.md",
    SUBMISSION / "evidence" / "public_fixture_run_auto.json",
    SUBMISSION / "evidence" / "public_fixture_run_auto_summary.json",
    DECK / "slides.md",
    DECK / "PerfRunbench-Arm-AI-Challenge.pdf",
)

EXPECTED_SCREENSHOTS = (
    "01-perfrunbench-overview.png",
    "02-real-arm-serving-throughput.png",
    "03-real-arm-serving-latency.png",
    "04-public-fixture-run-auto.png",
    "05-recommendation-guards.png",
    "06-arm-and-platform-validation.png",
    "07-t4g-training-tuning-evidence.png",
    "08-c7g4xlarge-workload-aware-evidence.png",
    "09-c7g-size-thread-scaling.png",
    "10-c7g-size-before-after.png",
)

TEXT_SUFFIXES = {".md", ".json", ".py", ".css", ".mjs", ".svg"}
SCAN_PATTERNS = {
    "windows_absolute_path": re.compile(r"\b[A-Za-z]:\\"),
    "unix_private_home": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "ipv4_address": re.compile(
        r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
    ),
    "pem_path": re.compile(r"(?i)\.pem(?:\b|\\|/)"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "pending_placeholder": re.compile(r"\b(?:TBD|DEMO QR PENDING)\b"),
    "unsafe_host_key_check": re.compile(r"StrictHostKeyChecking\s*=\s*no"),
}


def main() -> int:
    failures: list[dict[str, str]] = []
    for path in EXPECTED_FILES:
        if not path.is_file():
            failures.append({"kind": "missing_file", "path": _relative(path)})

    scanned_files = 0
    for path in _text_files():
        scanned_files += 1
        text = path.read_text(encoding="utf-8")
        for name, pattern in SCAN_PATTERNS.items():
            if match := pattern.search(text):
                failures.append(
                    {
                        "kind": name,
                        "path": _relative(path),
                        "match": match.group(0),
                    }
                )

    screenshot_records = []
    for filename in EXPECTED_SCREENSHOTS:
        path = DEVPOST_ASSETS / filename
        if not path.is_file():
            failures.append({"kind": "missing_screenshot", "path": _relative(path)})
            continue
        width, height = _png_dimensions(path)
        if width * 9 != height * 16:
            failures.append(
                {
                    "kind": "screenshot_not_16_9",
                    "path": _relative(path),
                    "match": f"{width}x{height}",
                }
            )
        screenshot_records.append(
            {
                "path": _relative(path),
                "dimensions": f"{width}x{height}",
                "sha256": _sha256(path),
            }
        )

    raw = _load_json(SUBMISSION / "evidence" / "public_fixture_run_auto.json", failures)
    summary = _load_json(
        SUBMISSION / "evidence" / "public_fixture_run_auto_summary.json", failures
    )
    if raw and raw.get("auto_status") != "no_eligible_candidate":
        failures.append({"kind": "fixture_status_changed", "path": "public_fixture_run_auto.json"})
    if summary and summary.get("performance_evidence") is not False:
        failures.append({"kind": "fixture_promoted_to_performance", "path": "public_fixture_run_auto_summary.json"})

    pdf_path = DECK / "PerfRunbench-Arm-AI-Challenge.pdf"
    pdf_record = None
    if pdf_path.is_file():
        if not pdf_path.read_bytes().startswith(b"%PDF-"):
            failures.append({"kind": "invalid_pdf_header", "path": _relative(pdf_path)})
        pdf_record = {
            "path": _relative(pdf_path),
            "bytes": pdf_path.stat().st_size,
            "sha256": _sha256(pdf_path),
        }

    result = {
        "schema_version": "1.0",
        "status": "passed" if not failures else "failed",
        "scanned_text_files": scanned_files,
        "screenshots": screenshot_records,
        "pdf": pdf_record,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _text_files() -> list[Path]:
    paths = [ROOT / "README.md", ROOT / "LICENSE", ROOT / "docs" / "releases" / "v0.3.1.md"]
    paths.extend(
        path
        for path in SUBMISSION.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "node_modules" not in path.parts
        and "dist" not in path.parts
        and ".qa" not in path.parts
        and path.name != "package-lock.json"
    )
    return sorted(set(paths))


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"{_relative(path)} is not a PNG")
    return struct.unpack(">II", data[16:24])


def _load_json(path: Path, failures: list[dict[str, str]]) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append({"kind": "invalid_json", "path": _relative(path), "match": str(exc)})
        return None
    if not isinstance(payload, dict):
        failures.append({"kind": "json_not_object", "path": _relative(path)})
        return None
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
