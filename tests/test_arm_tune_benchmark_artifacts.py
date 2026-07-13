from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest, write_text_artifact
from autotune.arm_tune.benchmark.hashes import sha256_file


class ArmTuneBenchmarkArtifactsTest(unittest.TestCase):
    def test_streaming_hash_matches_hashlib(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "large.bin"
            payload = (b"0123456789abcdef" * 8192) + b"tail"
            path.write_bytes(payload)

            digest = sha256_file(path, chunk_size=17)

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_text_artifact_record_includes_digest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = write_text_artifact(Path(temp_dir) / "raw" / "stdout.txt", "fixture output\n")

            record = artifact_record(path, label="stdout")

        self.assertTrue(record["exists"])
        self.assertEqual(record["label"], "stdout")
        self.assertEqual(record["sha256"], hashlib.sha256(b"fixture output\n").hexdigest())

    def test_artifact_manifest_is_versioned_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "artifact_manifest.json"
            write_artifact_manifest(manifest, {"stdout": {"path": "stdout.txt"}})

            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertIn("stdout", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
