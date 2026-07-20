from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.config import load_target_spec, parse_optimize_config, parse_target_spec
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ArmTuneConfigTest(unittest.TestCase):
    def test_parse_local_target_defaults(self) -> None:
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "local"}})

        self.assertEqual(spec.type, "local")
        self.assertTrue(spec.strict_host_key_checking)
        self.assertEqual(spec.schema_version, "1.0")

    def test_parse_ssh_target_defaults_to_strict_host_key_checking(self) -> None:
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh", "host": "arm-host"}})

        self.assertEqual(spec.type, "ssh")
        self.assertEqual(spec.host, "arm-host")
        self.assertEqual(spec.port, 22)
        self.assertEqual(spec.remote_python, "python3")
        self.assertEqual(spec.remote_module, "autotune.cli")
        self.assertTrue(spec.strict_host_key_checking)

    def test_parse_ssh_target_accepts_structured_remote_worker_fields(self) -> None:
        spec = parse_target_spec(
            {
                "schema_version": "1.0",
                "target": {
                    "type": "ssh",
                    "host": "arm-host",
                    "remote_python": "/usr/bin/python3",
                    "remote_module": "autotune.cli",
                },
            }
        )

        self.assertEqual(spec.remote_python, "/usr/bin/python3")
        self.assertEqual(spec.remote_module, "autotune.cli")

    def test_rejects_unknown_target_fields(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_target_spec({"schema_version": "1.0", "target": {"type": "local", "secret": "nope"}})

    def test_rejects_arbitrary_remote_command_field(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_target_spec(
                {
                    "schema_version": "1.0",
                    "target": {
                        "type": "ssh",
                        "host": "arm-host",
                        "remote_worker_command": "python3 -m autotune.cli arm worker",
                    },
                }
            )

    def test_rejects_invalid_remote_module(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_target_spec(
                {
                    "schema_version": "1.0",
                    "target": {"type": "ssh", "host": "arm-host", "remote_module": "autotune.cli;echo nope"},
                }
            )

    def test_rejects_ssh_without_host(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh"}})

    def test_load_json_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.json"
            path.write_text('{"schema_version": "1.0", "target": {"type": "local"}}', encoding="utf-8")

            spec = load_target_spec(path)

        self.assertEqual(spec.type, "local")

    def test_load_json_config_accepts_utf8_bom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.json"
            path.write_text('{"schema_version": "1.0", "target": {"type": "local"}}', encoding="utf-8-sig")

            payload = load_config(path)

        self.assertEqual(payload["schema_version"], "1.0")

    def test_sanitized_dict_redacts_identity_file_path(self) -> None:
        spec = parse_target_spec(
            {
                "schema_version": "1.0",
                "target": {"type": "ssh", "host": "arm-host", "identity_file": "~/.ssh/id_ed25519"},
            }
        )

        self.assertEqual(spec.sanitized_dict()["identity_file"], "<configured>")

    def test_parse_optimize_config_accepts_candidate_controls(self) -> None:
        _, _, search = parse_optimize_config(
            _optimize_config(
                search={
                    "max_candidates": 3,
                    "affinity_modes": ["scheduler-default", "compact", "compact"],
                }
            )
        )

        self.assertEqual(search.max_candidates, 3)
        self.assertEqual(search.affinity_modes, ["scheduler-default", "compact"])

    def test_parse_optimize_config_rejects_invalid_candidate_controls(self) -> None:
        with self.assertRaises(ArmTuneConfigError):
            parse_optimize_config(_optimize_config(search={"max_candidates": 0}))

        with self.assertRaises(ArmTuneConfigError):
            parse_optimize_config(_optimize_config(search={"affinity_modes": ["compact", "shell"]}))

    def test_public_phase9_optimize_examples_parse(self) -> None:
        _, bench_job, bench_search = parse_optimize_config(
            load_config(ROOT / "configs" / "armtune" / "llama_bench_optimize.example.json")
        )
        _, server_job, server_search = parse_optimize_config(
            load_config(ROOT / "configs" / "armtune" / "llama_server_optimize.example.json")
        )

        self.assertEqual(bench_job.executable.path, "/<remote-llama-bench-binary>")
        self.assertEqual(bench_search.max_candidates, 12)
        self.assertEqual(bench_search.affinity_modes, ["scheduler-default", "compact", "spread", "reserve-one-core"])
        self.assertEqual(server_job.workload.prompt_suite_path, "/<remote-prompt-suite-json>")
        self.assertFalse(server_search.quality_gate_enabled)
        self.assertIsNone(server_search.quality_suite_path)
        self.assertIsNone(server_search.quality_normalization_policy)


def _optimize_config(*, search: dict | None = None) -> dict:
    payload = {
        "schema_version": "1.0",
        "result_origin": "test_fixture",
        "target": {"type": "local"},
        "executable": {"path": "/bin/echo"},
        "model": {"path": "/tmp/model.gguf"},
        "benchmark": {
            "threads": 1,
            "prompt_tokens": 16,
            "generation_tokens": 8,
            "repetitions": 1,
            "timeout_seconds": 5,
        },
    }
    if search is not None:
        payload["search"] = search
    return payload


if __name__ == "__main__":
    unittest.main()
