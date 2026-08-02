from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.targets.protocol import atomic_write_json


STARTER_SCHEMA_VERSION = "1.0"
STARTER_TARGET_TYPES = ("ssh", "local")
STARTER_WORKLOADS = ("llama-bench", "llama-server")
STARTER_PRESETS = ("quick", "standard", "deep")


@dataclass(frozen=True)
class SearchPreset:
    name: str
    purpose: str
    warmup_runs: int
    search_runs: int
    confirmation_runs: int
    cooldown_seconds: float
    max_candidates: int
    noise_band_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STARTER_SCHEMA_VERSION,
            "name": self.name,
            "purpose": self.purpose,
            "warmup_runs": self.warmup_runs,
            "search_runs": self.search_runs,
            "confirmation_runs": self.confirmation_runs,
            "cooldown_seconds": self.cooldown_seconds,
            "max_candidates": self.max_candidates,
            "noise_band_percent": self.noise_band_percent,
        }


SEARCH_PRESETS = {
    "quick": SearchPreset(
        name="quick",
        purpose="Validate the end-to-end workflow with the smallest useful search.",
        warmup_runs=1,
        search_runs=1,
        confirmation_runs=3,
        cooldown_seconds=1.0,
        max_candidates=4,
        noise_band_percent=5.0,
    ),
    "standard": SearchPreset(
        name="standard",
        purpose="Run a bounded comparison with repeated search and confirmation measurements.",
        warmup_runs=1,
        search_runs=3,
        confirmation_runs=5,
        cooldown_seconds=2.0,
        max_candidates=8,
        noise_band_percent=5.0,
    ),
    "deep": SearchPreset(
        name="deep",
        purpose="Use broader bounded coverage for a stable, already-qualified target.",
        warmup_runs=2,
        search_runs=5,
        confirmation_runs=10,
        cooldown_seconds=2.0,
        max_candidates=12,
        noise_band_percent=2.0,
    ),
}


CONCEPTS: dict[str, dict[str, str]] = {
    "prefill": {
        "plain_name": "Prompt processing",
        "meaning": "The model reads and encodes the input tokens before it generates the first output token.",
        "why_it_matters": "Long prompts and retrieved context spend more time here.",
        "primary_metric": "prefill tokens per second and time to first token",
    },
    "decode": {
        "plain_name": "Token generation",
        "meaning": "The model generates output tokens one at a time after prompt processing finishes.",
        "why_it_matters": "Long answers are usually limited by decode throughput.",
        "primary_metric": "decode or output tokens per second",
    },
    "interactive-short": {
        "plain_name": "Short interactive request",
        "meaning": "A short prompt with a short-to-medium response, similar to an interactive assistant turn.",
        "why_it_matters": "Users notice first-token and end-to-end latency immediately.",
        "primary_metric": "p95 time to first token, end-to-end latency, and output throughput",
    },
    "decode-heavy": {
        "plain_name": "Long-generation request",
        "meaning": "A short prompt that asks for a much longer response.",
        "why_it_matters": "It isolates sustained output-token generation and concurrent decode scheduling.",
        "primary_metric": "output tokens per second, with latency regression guards",
    },
    "prefill-heavy": {
        "plain_name": "Long-context request",
        "meaning": "A long prompt followed by a relatively short response.",
        "why_it_matters": "It stresses prompt ingestion, memory traffic, queueing, and time to first token.",
        "primary_metric": "prefill throughput and p95 time to first token",
    },
    "continuous-batching": {
        "plain_name": "Continuous request batching",
        "meaning": "The server schedules active requests together at token boundaries instead of waiting for a fixed batch to finish.",
        "why_it_matters": "It can improve multi-user throughput but may hurt long-prompt latency, so ArmTune validates workloads separately.",
        "primary_metric": "throughput plus workload-specific latency guards",
    },
    "ttft": {
        "plain_name": "Time to first token",
        "meaning": "Elapsed time from sending a request until the first generated token arrives.",
        "why_it_matters": "It is the most visible responsiveness metric for streaming inference.",
        "primary_metric": "milliseconds; lower is better",
    },
    "p95": {
        "plain_name": "95th percentile",
        "meaning": "Ninety-five percent of measured samples are at or below this value.",
        "why_it_matters": "It exposes slow-tail behavior that a median can hide.",
        "primary_metric": "metric-dependent; lower is better for latency",
    },
    "noise-band": {
        "plain_name": "Minimum meaningful change",
        "meaning": "A configured zone around the baseline where differences are treated as measurement noise.",
        "why_it_matters": "ArmTune does not recommend a candidate merely because its average is slightly higher.",
        "primary_metric": "classification: improved, regressed, or within noise",
    },
    "performance-passport": {
        "plain_name": "Auditable result record",
        "meaning": "A versioned artifact that binds a result to the target, model, runtime, protocol, evidence, and guards.",
        "why_it_matters": "It prevents a recommendation from being reused as if it applied to a different environment.",
        "primary_metric": "verification and eligibility state",
    },
}


def create_starter_bundle(
    output_dir: str | Path,
    *,
    target_type: str,
    workload: str,
    preset_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if target_type not in STARTER_TARGET_TYPES:
        raise ArmTuneConfigError(f"unsupported starter target type: {target_type}")
    if workload not in STARTER_WORKLOADS:
        raise ArmTuneConfigError(f"unsupported starter workload: {workload}")
    if preset_name not in SEARCH_PRESETS:
        raise ArmTuneConfigError(f"unsupported starter preset: {preset_name}")

    root = Path(output_dir).expanduser()
    paths = {
        "manifest": root / "armtune-project.json",
        "target": root / "target.json",
        "benchmark": root / "benchmark.json",
        "optimize": root / "optimize.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ArmTuneConfigError(
            "starter bundle already contains managed files; use --overwrite to replace them: "
            + ", ".join(existing)
        )

    root.mkdir(parents=True, exist_ok=True)
    target = _target_payload(target_type)
    benchmark = _benchmark_payload(target, workload)
    optimize = _optimize_payload(benchmark, SEARCH_PRESETS[preset_name])
    manifest = {
        "schema_version": STARTER_SCHEMA_VERSION,
        "kind": "armtune_starter_project",
        "target_type": target_type,
        "workload": workload,
        "preset": SEARCH_PRESETS[preset_name].to_dict(),
        "files": {key: path.name for key, path in paths.items() if key != "manifest"},
        "result_origin": "unverified",
        "notes": [
            "Replace every angle-bracket placeholder before target access.",
            "Keep real SSH identity, host, binary, and model paths outside public repositories.",
            "A generated starter does not constitute measured Arm performance evidence.",
        ],
    }
    atomic_write_json(paths["target"], target)
    atomic_write_json(paths["benchmark"], benchmark)
    atomic_write_json(paths["optimize"], optimize)
    atomic_write_json(paths["manifest"], manifest)
    return {
        "schema_version": STARTER_SCHEMA_VERSION,
        "kind": "armtune_starter_bundle_result",
        "output_directory": str(root.resolve()),
        "files": {key: str(path.resolve()) for key, path in paths.items()},
        "preset": SEARCH_PRESETS[preset_name].to_dict(),
    }


def guided_project_status(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).expanduser()
    expected = {
        "manifest": root / "armtune-project.json",
        "target": root / "target.json",
        "benchmark": root / "benchmark.json",
        "optimize": root / "optimize.json",
    }
    missing = [key for key, path in expected.items() if not path.is_file()]
    placeholders: list[str] = []
    invalid_json: list[str] = []
    for key, path in expected.items():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json.append(key)
            continue
        placeholders.extend(f"{key}:{item}" for item in _placeholder_paths(payload))
    if missing:
        readiness = "not_initialized"
    elif invalid_json:
        readiness = "invalid_configuration"
    elif placeholders:
        readiness = "needs_configuration"
    else:
        readiness = "ready_for_doctor"
    return {
        "schema_version": STARTER_SCHEMA_VERSION,
        "kind": "armtune_guided_project_status",
        "project_directory": str(root.resolve()),
        "readiness": readiness,
        "missing_files": missing,
        "invalid_json_files": invalid_json,
        "placeholder_fields": sorted(placeholders),
        "workflow": _guided_workflow(root),
        "safety": [
            "doctor and guide do not run a benchmark",
            "production performance requires a real Linux Arm64 target",
            "recommendations remain bound to target, model, runtime, and protocol evidence",
        ],
    }


def concept_payload(term: str | None = None) -> dict[str, Any]:
    if term is not None and term not in CONCEPTS:
        raise ArmTuneConfigError(f"unknown ArmTune concept: {term}")
    concepts = {term: CONCEPTS[term]} if term else CONCEPTS
    return {
        "schema_version": STARTER_SCHEMA_VERSION,
        "kind": "armtune_concepts",
        "concepts": concepts,
    }


def format_starter_result(result: dict[str, Any]) -> str:
    files = result["files"]
    return "\n".join(
        [
            "ArmTune starter project created",
            f"project_directory: {result['output_directory']}",
            f"preset: {result['preset']['name']} ({result['preset']['max_candidates']} candidates maximum)",
            f"target_config: {files['target']}",
            f"benchmark_config: {files['benchmark']}",
            f"optimize_config: {files['optimize']}",
            "",
            "Next: replace angle-bracket placeholders, then run:",
            f"  autotuneai arm guide --project \"{result['output_directory']}\"",
        ]
    )


def format_guided_status(status: dict[str, Any]) -> str:
    lines = [
        "ArmTune guided workflow",
        f"project_directory: {status['project_directory']}",
        f"readiness: {status['readiness']}",
    ]
    if status["missing_files"]:
        lines.append("missing_files: " + ", ".join(status["missing_files"]))
        lines.append("initialize: autotuneai arm init")
    if status["invalid_json_files"]:
        lines.append("invalid_json_files: " + ", ".join(status["invalid_json_files"]))
    if status["placeholder_fields"]:
        lines.append(f"placeholder_count: {len(status['placeholder_fields'])}")
        for item in status["placeholder_fields"][:8]:
            lines.append(f"  - {item}")
        if len(status["placeholder_fields"]) > 8:
            lines.append(f"  - ... and {len(status['placeholder_fields']) - 8} more")
    lines.extend(["", "Workflow:"])
    for index, step in enumerate(status["workflow"], start=1):
        lines.append(f"  {index}. {step['label']}")
        lines.append(f"     {step['command']}")
    lines.extend(["", "Safety:"])
    lines.extend(f"  - {item}" for item in status["safety"])
    return "\n".join(lines)


def format_concepts(payload: dict[str, Any]) -> str:
    lines = ["ArmTune concepts"]
    for term, details in payload["concepts"].items():
        lines.extend(
            [
                "",
                f"{term} - {details['plain_name']}",
                f"  Meaning: {details['meaning']}",
                f"  Why it matters: {details['why_it_matters']}",
                f"  Read: {details['primary_metric']}",
            ]
        )
    return "\n".join(lines)


def _target_payload(target_type: str) -> dict[str, Any]:
    if target_type == "local":
        target: dict[str, Any] = {"type": "local"}
    else:
        target = {
            "type": "ssh",
            "host": "<arm-linux-hostname>",
            "user": "<ssh-user>",
            "port": 22,
            "identity_file": "<private-identity-file-path>",
            "remote_python": "python3",
            "remote_module": "autotune.cli",
            "remote_workdir": "<remote-perfrunbench-workdir>",
            "connect_timeout_seconds": 30,
            "command_timeout_seconds": 300,
            "keep_remote_artifacts": False,
            "strict_host_key_checking": True,
        }
    return {"schema_version": SCHEMA_VERSION, "target": target}


def _benchmark_payload(target_payload: dict[str, Any], workload: str) -> dict[str, Any]:
    target = target_payload["target"]
    common: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_origin": "unverified",
        "target": target,
        "executable": {"path": "<absolute-llama-runtime-binary>", "expected_sha256": None},
        "model": {"path": "<absolute-gguf-model-file>", "expected_sha256": None},
    }
    if workload == "llama-bench":
        common.update(
            {
                "benchmark": {
                    "threads": 1,
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                    "repetitions": 3,
                    "timeout_seconds": 300,
                    "additional_supported_options": {},
                },
                "artifacts": {"preserve_raw_stdout": True, "preserve_raw_stderr": True},
            }
        )
    else:
        common.update(
            {
                "runtime": {"adapter": "llama_server"},
                "server": {
                    "host": "127.0.0.1",
                    "port": "auto",
                    "startup_timeout_seconds": 30,
                    "request_timeout_seconds": 60,
                    "shutdown_timeout_seconds": 5,
                    "threads": 1,
                    "context_size": 2048,
                    "seed": 42,
                    "temperature": 0.0,
                    "affinity_mode": "scheduler-default",
                },
                "workload": {
                    "mode": "server",
                    "prompt_suite_path": "<absolute-public-prompt-suite-json>",
                    "predict_tokens": 96,
                    "repetitions": 3,
                    "warmup_requests": 1,
                    "streaming": True,
                    "cooldown_seconds": 1.0,
                },
                "artifacts": {
                    "preserve_server_stdout": True,
                    "preserve_server_stderr": True,
                    "preserve_request_stream": True,
                    "preserve_generated_output": True,
                },
            }
        )
    return common


def _optimize_payload(benchmark: dict[str, Any], preset: SearchPreset) -> dict[str, Any]:
    payload = json.loads(json.dumps(benchmark))
    payload["search"] = {
        "warmup_runs": preset.warmup_runs,
        "search_runs": preset.search_runs,
        "confirmation_runs": preset.confirmation_runs,
        "cooldown_seconds": preset.cooldown_seconds,
        "random_seed": 42,
        "insert_paired_baseline_every": 2,
        "noise_band_percent": preset.noise_band_percent,
        "max_candidates": preset.max_candidates,
        "affinity_modes": ["scheduler-default", "compact", "spread", "reserve-one-core"],
    }
    return payload


def _placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_placeholder_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and "<" in value and ">" in value:
        found.append(prefix)
    return found


def _guided_workflow(root: Path) -> list[dict[str, str]]:
    target = root / "target.json"
    benchmark = root / "benchmark.json"
    optimize = root / "optimize.json"
    return [
        {"label": "Validate the controller and Arm target", "command": f'autotuneai arm doctor --target-config "{target}" --require-production'},
        {"label": "Inspect CPU identity, topology, and environment", "command": f'autotuneai arm inspect-host --target-config "{target}"'},
        {"label": "Establish a fixed baseline", "command": f'autotuneai arm benchmark --config "{benchmark}"'},
        {"label": "Run a bounded runtime search and confirmation", "command": f'autotuneai arm optimize --config "{optimize}"'},
        {"label": "Review artifacts before generating or applying a recommendation", "command": "autotuneai arm explain performance-passport"},
    ]
