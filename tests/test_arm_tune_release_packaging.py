from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArmTuneReleasePackagingTest(unittest.TestCase):
    def test_release_docs_present_required_public_claims(self) -> None:
        readme = _read("README.md")
        devpost = _read("docs/armtune/DEVPOST_DRAFT.md")
        demo = _read("docs/armtune/DEMO_SCRIPT.md")
        limitations = _read("docs/armtune/LIMITATIONS.md")

        for content in [readme, devpost, demo]:
            self.assertIn("PerfRunbench", content)
            self.assertIn("ArmTune", content)
            self.assertIn("10.825 tok/s", content)
            self.assertIn("17.530 tok/s", content)
            self.assertIn("17.380 tok/s", content)
            self.assertIn("13.630 tok/s", content)
            self.assertIn("17.480 tok/s", content)
            self.assertIn("28.246515%", content)
            self.assertIn("97.260 ms", content)
            self.assertIn("62.826 ms", content)
            self.assertIn("eligible_preliminary", content)
            self.assertIn("guarded apply validation", content)

        self.assertIn("must not be combined", readme)
        self.assertIn("These results must not be combined", devpost)
        self.assertIn("split into the standalone BootTraceBench project", limitations)

    def test_architecture_diagram_and_default_flow_are_documented(self) -> None:
        architecture = _read("docs/armtune/ARCHITECTURE.md")

        for term in [
            "Controller",
            "Target inspection",
            "Workload adapter",
            "Build profiles",
            "Search engine",
            "Statistical guard",
            "Performance Passport",
            "Recommendation safety/apply",
        ]:
            self.assertIn(term, architecture)

        self.assertIn("flowchart LR", architecture)
        self.assertIn("BootTraceBench is an external historical/experimental project", architecture)

    def test_phase10_sanitized_evidence_contains_apply_validation(self) -> None:
        evidence = _read("docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md")

        for term in [
            "92b0eb69b189a06baae62f15",
            "environment_validation: pass",
            "dry_run_status: dry_run_passed",
            "canary_status: passed",
            "lease_state_after_apply: active",
            "rollback_state: rolled_back",
            "remote_manifest_revalidation_checked: 26",
            "downloaded_manifest_revalidation_checked: 26",
            "cleanup_status: passed",
        ]:
            self.assertIn(term, evidence)

        _assert_no_endpoint_or_private_path(evidence)

    def test_public_placeholder_examples_are_parseable_and_placeholder_only(self) -> None:
        json_examples = [
            Path("configs/armtune/target_ssh.example.json"),
            Path("configs/armtune/llama_bench.example.json"),
            Path("configs/armtune/llama_server.example.json"),
            Path("configs/armtune/build_profiles.example.json"),
            Path("configs/armtune/llama_bench_optimize.example.json"),
            Path("configs/armtune/llama_server_optimize.example.json"),
        ]
        for relative in json_examples:
            content = _read(relative)
            payload = json.loads(content)
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertIn("<", content)
            self.assertIn(">", content)
            _assert_no_endpoint_or_private_path(content)
            self.assertNotIn("~/.ssh", content)
            self.assertNotIn("/opt/llama.cpp", content)

        recommendation = _read("configs/armtune/recommendation_apply.example.md")
        self.assertIn("autotuneai recommendation show", recommendation)
        self.assertIn("autotuneai recommendation validate", recommendation)
        self.assertIn("--dry-run", recommendation)
        self.assertIn("autotuneai recommendation rollback", recommendation)
        _assert_no_endpoint_or_private_path(recommendation)

    def test_boottrace_specific_workload_is_not_in_default_code_or_configs(self) -> None:
        forbidden = re.compile(
            r"boottrace|quality-calibrate|quality_calibrate|prompt_contract",
            re.IGNORECASE,
        )

        matches: list[str] = []
        for root in [ROOT / "autotune", ROOT / "configs"]:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".py", ".json", ".md", ".yaml", ".yml"}:
                    continue
                content = path.read_text(encoding="utf-8")
                if forbidden.search(content):
                    matches.append(str(path.relative_to(ROOT)))

        self.assertEqual(matches, [])


def _read(relative: str | Path) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _assert_no_endpoint_or_private_path(content: str) -> None:
    self_hosted_ipv4 = re.compile(r"\b(?!(?:127|0)\.)\d{1,3}(?:\.\d{1,3}){3}\b")
    windows_absolute_path = re.compile(r"\b[A-Za-z]:\\")
    private_linux_home = re.compile(r"/home/(?:ubuntu|ec2-user|admin|user)\b")
    pem_path = re.compile(r"\.pem(?:\b|\.)", re.IGNORECASE)

    assert not self_hosted_ipv4.search(content)
    assert not windows_absolute_path.search(content)
    assert not private_linux_home.search(content)
    assert not pem_path.search(content)


if __name__ == "__main__":
    unittest.main()
