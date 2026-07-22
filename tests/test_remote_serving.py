from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_21D2B_DOC = ROOT / "docs" / "results" / "PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md"


class RemoteServingValidationDocsTest(unittest.TestCase):
    def test_phase21d2b_records_required_completion_flags_and_protocol_boundary(self) -> None:
        text = PHASE_21D2B_DOC.read_text(encoding="utf-8")

        self.assertIn("phase21d2b_remote_serving_validation: passed", text)
        self.assertIn("remote_slots4_throughput_improvement_verified: true", text)
        self.assertIn("remote_slots4_latency_improvement_verified: true", text)
        self.assertIn("full_capacity_remote_recommendation_created: true", text)
        self.assertIn("network_tuning_skipped_due_to_no_bottleneck: true", text)
        self.assertIn("dual_host_restored: true", text)
        self.assertIn("ready_to_stop_c7g_t4g: true", text)
        self.assertIn("measured_requests_total: 608", text)
        self.assertIn("paired_block_pairs: 6", text)
        self.assertIn("J2 seven CPU", text)
        self.assertIn("J1 full eight CPU", text)
        self.assertIn("does not reuse the Phase 21D1", text)
        self.assertIn("must not be combined with loopback", text)

    def test_phase21d2b_public_doc_excludes_private_connection_details(self) -> None:
        text = PHASE_21D2B_DOC.read_text(encoding="utf-8")

        forbidden = [
            r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
            r"sg-[0-9a-f]+",
            r"\.pem",
            r"phase21d2b-evidence",
            r"/home/ubuntu/models",
            r"ssh_config",
            r"HostKey",
            r"public_ipv4",
            r"private_ipv4",
        ]
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, text, flags=re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
