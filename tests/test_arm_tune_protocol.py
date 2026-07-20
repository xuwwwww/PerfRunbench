from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.errors import ArmTuneProtocolError
from autotune.arm_tune.schemas import WorkerJobSpec
from autotune.arm_tune.targets.protocol import atomic_write_json, read_json, safe_remote_path, validate_worker_job
from autotune.arm_tune.targets.worker import run_worker


class ArmTuneProtocolTest(unittest.TestCase):
    def test_atomic_write_json_writes_schema_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            atomic_write_json(path, {"schema_version": "1.0", "kind": "test"})

            payload = read_json(path)

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["kind"], "test")

    def test_validate_worker_job_rejects_unknown_operation(self) -> None:
        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job({"schema_version": "1.0", "operation": "benchmark"})

    def test_validate_worker_job_rejects_protocol_version_mismatch(self) -> None:
        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job({"schema_version": "2.0", "operation": "inspect_host", "target_type": "ssh"})

    def test_validate_worker_job_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ArmTuneProtocolError):
            validate_worker_job(
                {
                    "schema_version": "1.0",
                    "operation": "inspect_host",
                    "target_type": "ssh",
                    "remote_worker_command": "python3 -m autotune.cli arm worker",
                }
            )

    def test_safe_remote_path_rejects_shell_metacharacters(self) -> None:
        with self.assertRaises(ArmTuneProtocolError):
            safe_remote_path("~/runs/job.json;rm -rf /")
        with self.assertRaises(ArmTuneProtocolError):
            safe_remote_path("~/runs with spaces/job.json")

    def test_worker_inspect_host_writes_versioned_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = root / "job.json"
            run_dir = root / "run"
            atomic_write_json(job, WorkerJobSpec(operation="inspect_host", target_type="local").to_dict())

            result = run_worker(job, run_dir)

            output = json.loads((run_dir / "inspect_host_result.json").read_text(encoding="utf-8"))
        self.assertEqual(result.status, "ok")
        self.assertEqual(output["schema_version"], "1.0")
        self.assertIn("host", output)
        self.assertIn("workload_execution_evidence", output)
        self.assertEqual(output["workload_execution_evidence"]["status"], "not_collected")

    def test_worker_check_prerequisites_writes_versioned_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = root / "job.json"
            run_dir = root / "run"
            atomic_write_json(job, WorkerJobSpec(operation="check_prerequisites", target_type="local").to_dict())

            result = run_worker(job, run_dir)

            output = json.loads((run_dir / "prerequisite_result.json").read_text(encoding="utf-8"))
        self.assertEqual(result.status, "ok")
        self.assertEqual(output["schema_version"], "1.0")
        self.assertTrue(output["python_available"])
        self.assertTrue(output["module_importable"])
        self.assertTrue(output["worker_protocol_compatible"])
        self.assertTrue(output["remote_workdir_writable"])

    def test_worker_rejects_non_writable_run_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = root / "job.json"
            run_dir = root / "not-a-directory"
            run_dir.write_text("block directory creation", encoding="utf-8")
            atomic_write_json(job, WorkerJobSpec(operation="check_prerequisites", target_type="local").to_dict())

            with self.assertRaises(ArmTuneProtocolError):
                run_worker(job, run_dir)


if __name__ == "__main__":
    unittest.main()
