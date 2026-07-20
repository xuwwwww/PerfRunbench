from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotune.arm_tune.benchmark.artifact_validation import apply_artifact_validation, validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest, write_text_artifact
from autotune.arm_tune.schemas import ArchitectureInfo, ArmBenchmarkResult, ArmCpuIdentity, ArmHostCapabilities, CpuTopology, HostInfo
from autotune.arm_tune.targets.protocol import atomic_write_json


class ArmTuneArtifactValidationTest(unittest.TestCase):
    def test_correct_digest_zero_byte_crlf_and_lf_files_pass(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            lf = write_text_artifact(raw / "lf.txt", "one\n")
            crlf = raw / "crlf.txt"
            crlf.write_bytes(b"one\r\n")
            empty = raw / "empty.txt"
            empty.write_bytes(b"")
            artifacts = {
                "lf": artifact_record(lf, label="lf", root=root),
                "crlf": artifact_record(crlf, label="crlf", root=root),
                "empty": artifact_record(empty, label="empty", root=root),
            }
            write_artifact_manifest(root / "artifact_manifest.json", artifacts)

            result = validate_artifact_manifest(root)

        self.assertEqual(result.validation_status, "passed")
        self.assertEqual(result.checked_artifacts, 3)

    def test_corrupted_artifact_fails_without_deleting_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_text_artifact(root / "raw" / "stdout.txt", "before\n")
            record = artifact_record(path, label="stdout", root=root)
            write_artifact_manifest(root / "artifact_manifest.json", {"stdout": record})
            path.write_text("after\n", encoding="utf-8")

            result = validate_artifact_manifest(root)

            self.assertTrue(path.exists())
        self.assertEqual(result.validation_status, "failed")
        self.assertIn("sha256 mismatch", result.issues[0].issue)

    def test_failed_artifact_validation_prevents_successful_benchmark_status(self) -> None:
        host = HostInfo(
            architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", is_linux=True, is_arm64=True),
            cpu_identity=ArmCpuIdentity(),
            host_capabilities=ArmHostCapabilities(),
            topology=CpuTopology(),
        )
        result = ArmBenchmarkResult(
            run_id="run",
            operation="benchmark.llama_bench",
            target_type="local",
            result_origin="measured",
            validation_status="passed",
            execution_status="completed",
            parse_status="parsed",
            benchmark_status="passed",
            host=host,
            message="benchmark completed",
        )
        validation = validate_artifact_manifest(Path("does-not-exist"))

        updated = apply_artifact_validation(result, validation, "does-not-exist")

        self.assertEqual(updated.artifact_validation.validation_status, "failed")
        self.assertEqual(updated.benchmark_status, "failed")

    def test_missing_artifact_and_duplicate_manifest_entry_fail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_text_artifact(root / "raw" / "stdout.txt", "data\n")
            digest = hashlib.sha256(b"data\n").hexdigest()
            write_artifact_manifest(
                root / "artifact_manifest.json",
                {
                    "stdout": {"label": "stdout", "relative_path": "raw/stdout.txt", "sha256": digest},
                    "duplicate": {"label": "duplicate", "relative_path": "raw/stdout.txt", "sha256": digest},
                    "missing": {"label": "missing", "relative_path": "raw/missing.txt", "sha256": "0" * 64},
                },
            )

            result = validate_artifact_manifest(root)

        self.assertEqual(result.validation_status, "failed")
        self.assertTrue(any("duplicate" in issue.issue for issue in result.issues))
        self.assertTrue(any("missing" in issue.issue for issue in result.issues))

    def test_path_traversal_and_manifest_outside_run_directory_fail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            root.mkdir()
            outside = Path(temp_dir) / "outside.json"
            atomic_write_json(outside, {"schema_version": "1.0", "artifacts": {}})
            write_artifact_manifest(
                root / "artifact_manifest.json",
                {"escape": {"label": "escape", "relative_path": "../outside.txt", "sha256": "0" * 64}},
            )

            traversal = validate_artifact_manifest(root)
            outside_manifest = validate_artifact_manifest(root, outside)

        self.assertEqual(traversal.validation_status, "failed")
        self.assertEqual(outside_manifest.validation_status, "failed")

    def test_symlink_escape_fails_where_symlinks_are_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            root.mkdir()
            outside = Path(temp_dir) / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")
            link = root / "raw-link.txt"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not available")
            write_artifact_manifest(
                root / "artifact_manifest.json",
                {"link": {"label": "link", "relative_path": "raw-link.txt", "sha256": hashlib.sha256(b"outside\n").hexdigest()}},
            )

            result = validate_artifact_manifest(root)

        self.assertEqual(result.validation_status, "failed")
        self.assertTrue(any("outside run directory" in issue.issue or "escapes" in issue.issue for issue in result.issues))

    def test_invalid_utf8_output_is_preserved_with_replacement(self) -> None:
        with TemporaryDirectory() as temp_dir:
            text = b"bad:\xff\n".decode("utf-8", errors="replace")
            path = write_text_artifact(Path(temp_dir) / "raw" / "stderr.txt", text)

            payload = path.read_text(encoding="utf-8")

        self.assertIn("\ufffd", payload)

    def test_interrupted_atomic_write_cleans_temporary_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "result.json"
            with patch("autotune.arm_tune.targets.protocol.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    atomic_write_json(target, {"schema_version": "1.0"})

            leftovers = list(root.glob(".result.json.*.tmp"))

        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
