from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class SubmissionPackageTest(unittest.TestCase):
    def test_public_submission_package_passes_safety_validation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "submission"
            / "validate_submission_package.py"
        )
        spec = importlib.util.spec_from_file_location(
            "validate_submission_package",
            script,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        output = io.StringIO()
        with redirect_stdout(output):
            code = module.main()

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["failures"], [])
        self.assertEqual(len(payload["screenshots"]), 6)
        self.assertTrue(payload["pdf"]["path"].endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()
