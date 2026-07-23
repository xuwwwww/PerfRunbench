from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_21D2B_DOC = ROOT / "docs" / "results" / "PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md"
VALIDATED_SUMMARY_DOC = ROOT / "docs" / "results" / "VALIDATED_PERFORMANCE_SUMMARY.md"
DEMO_DOC = ROOT / "docs" / "demo" / "SERVING_AUTOTUNING_DEMO.md"
RELEASE_DOC = ROOT / "docs" / "releases" / "v0.3.0.md"
PHASE_21D3_DOC = ROOT / "docs" / "results" / "PHASE_21D3_SERVING_AUTOTUNING_PRODUCTIZATION.md"


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

    def test_validated_summary_packages_remote_serving_without_overclaiming(self) -> None:
        text = VALIDATED_SUMMARY_DOC.read_text(encoding="utf-8")

        lowered = text.lower()
        self.assertIn("did not verify a compiler flag", lowered)
        self.assertIn("backend flag", lowered)
        self.assertIn("kernel profile", lowered)
        self.assertIn("17.032", text)
        self.assertIn("31.355", text)
        self.assertIn("+84.092%", text)
        self.assertIn("17.046", text)
        self.assertIn("31.790", text)
        self.assertIn("+86.490%", text)
        self.assertIn("16.563", text)
        self.assertIn("28.054", text)
        self.assertIn("+69.375%", text)
        self.assertIn("network_tuning_not_executed_evidence_absent", text)
        self.assertIn("do not claim universal", lowered)

    def test_demo_and_release_docs_include_sanitized_commands_and_exact_claims(self) -> None:
        for path in [DEMO_DOC, RELEASE_DOC]:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("autotuneai serve", text)
                self.assertIn("slots4-cont-on", text)
                self.assertIn("network_tuning_not_executed_evidence_absent", text)
                self.assertIsNone(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text))
                self.assertNotIn(".pem", text)

    def test_phase21d3_records_productization_source_ledger(self) -> None:
        text = PHASE_21D3_DOC.read_text(encoding="utf-8")

        self.assertIn("origin_main_commit: 8e53cb9b597dd2fdca4bfdc42af30c5d40b8566c", text)
        self.assertIn("phase21d2b_commit: d89bada8088cdc933d53081e4e730d00325d16b0", text)
        self.assertIn("phase21d3_initial_commit: d89bada8088cdc933d53081e4e730d00325d16b0", text)
        self.assertIn("pr6_status: open draft", text)
        self.assertIn("pr6_head: 09e5809d05dda933c7c27b88b1222e067674090e", text)
        self.assertIn("network_tuning_not_executed_evidence_absent", text)
        self.assertIsNone(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text))


if __name__ == "__main__":
    unittest.main()
