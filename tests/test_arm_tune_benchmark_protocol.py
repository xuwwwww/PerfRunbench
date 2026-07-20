from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.config import parse_benchmark_config
from autotune.arm_tune.errors import ArmTuneConfigError, ArmTuneProtocolError
from autotune.arm_tune.targets.protocol import atomic_write_json, validate_worker_job


class ArmTuneBenchmarkProtocolTest(unittest.TestCase):
    def test_parse_benchmark_config_rejects_relative_paths(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_benchmark_config(_config(executable_path="relative/fake", model_path="/models/fake.gguf"))
        with self.assertRaises(ArmTuneConfigError):
            parse_benchmark_config(_config(executable_path="/bin/llama-bench", model_path="models/fake.gguf"))

    def test_parse_benchmark_config_rejects_unknown_command_field(self) -> None:
        payload = _config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf")
        payload["benchmark"]["extra_command"] = ["rm", "-rf", "/"]

        with self.assertRaises(ArmTuneConfigError):
            parse_benchmark_config(payload)

    def test_parse_benchmark_config_rejects_invalid_numeric_ranges(self) -> None:
        payload = _config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf")
        payload["benchmark"]["threads"] = 0

        with self.assertRaises(ArmTuneConfigError):
            parse_benchmark_config(payload)

    def test_benchmark_worker_job_validation_rejects_protocol_mismatch(self) -> None:
        _, job = parse_benchmark_config(_config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf"))
        payload = job.to_dict()
        payload["protocol_version"] = "2.0"

        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job(payload)

    def test_benchmark_worker_job_validation_rejects_unsupported_operation(self) -> None:
        _, job = parse_benchmark_config(_config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf"))
        payload = job.to_dict()
        payload["operation"] = "benchmark.arbitrary"

        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job(payload)

    def test_benchmark_worker_job_validation_rejects_unknown_nested_fields(self) -> None:
        _, job = parse_benchmark_config(_config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf"))
        payload = job.to_dict()
        payload["executable"]["argv"] = ["not", "allowed"]

        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job(payload)

    def test_atomic_benchmark_job_write_is_versioned(self) -> None:
        _, job = parse_benchmark_config(_config(executable_path="/bin/llama-bench", model_path="/models/fake.gguf"))
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmark_job.json"
            atomic_write_json(path, job.to_dict())

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["operation"], "benchmark.llama_bench")


def _config(*, executable_path: str, model_path: str) -> dict:
    return {
        "schema_version": "1.0",
        "result_origin": "test_fixture",
        "target": {"type": "local"},
        "executable": {"path": executable_path},
        "model": {"path": model_path},
        "benchmark": {
            "threads": 1,
            "prompt_tokens": 16,
            "generation_tokens": 8,
            "repetitions": 1,
            "timeout_seconds": 5,
        },
        "artifacts": {"preserve_raw_stdout": True, "preserve_raw_stderr": True},
    }


if __name__ == "__main__":
    unittest.main()
