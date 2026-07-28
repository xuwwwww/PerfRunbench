from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from autotune.cli import main
from autotune.golden_path.config import load_golden_path_config
from autotune.report.training_tuning_report import generate_training_tuning_report
from autotune.training_adapter.candidates import generate_training_candidates
from autotune.training_adapter.config import validate_training_config


class TrainingTuningReportTest(unittest.TestCase):
    def test_public_t4g_visualization_config_is_bounded_and_semantics_preserving(self) -> None:
        config = load_golden_path_config("configs/experiments/t4g_training_visualization.yaml")
        validate_training_config(config)
        candidates, _statuses = generate_training_candidates(config)
        self.assertEqual(len(candidates), 5)
        self.assertEqual({item["effective_batch_size"] for item in candidates}, {32})
        self.assertTrue(all(item["recommendation_eligible"] for item in candidates))

    def test_generates_reusable_visual_artifacts_from_measured_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source = root / "tuning_result.json"
            source.write_text(json.dumps(_result()), encoding="utf-8")
            cloudwatch = root / "cloudwatch.json"
            cloudwatch.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "metrics": {
                            "CPUCreditBalance": [
                                {"timestamp": "2026-01-01T00:00:00Z", "value": 4.0},
                                {"timestamp": "2026-01-01T00:05:00Z", "value": 0.0},
                            ],
                            "CPUSurplusCreditBalance": [
                                {"timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
                                {"timestamp": "2026-01-01T00:05:00Z", "value": 2.0},
                            ],
                            "CPUSurplusCreditsCharged": [
                                {"timestamp": "2026-01-01T00:05:00Z", "value": 0.0},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            outputs = generate_training_tuning_report(source, root / "visuals", cloudwatch_path=cloudwatch)

            for path in outputs.values():
                self.assertTrue(Path(path).exists())
            summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["result_origin"], "measured")
            self.assertEqual(summary["paired_confirmation_count"], 3)
            self.assertAlmostEqual(summary["baseline_samples_per_second"]["median"], 100.0)
            self.assertAlmostEqual(summary["selected_samples_per_second"]["median"], 130.0)
            html = Path(outputs["report"]).read_text(encoding="utf-8")
            self.assertIn("Training tuning before and after", html)
            self.assertIn("tested training workload", html)
            self.assertIn("<code>tuning_result.json</code>", html)
            self.assertNotIn(str(root), html)
            charts = Path(outputs["charts"])
            self.assertTrue((charts / "paired_before_after.svg").exists())
            self.assertTrue((charts / "throughput_distribution.svg").exists())
            paired_chart = (charts / "paired_before_after.svg").read_text(encoding="utf-8")
            distribution_chart = (charts / "throughput_distribution.svg").read_text(encoding="utf-8")
            self.assertIn("pairs | median delta", paired_chart)
            self.assertIn("noise band 5.0% | all valid confirmation samples shown", distribution_chart)
            credit_chart = (charts / "cpu_credit_timeline.svg").read_text(encoding="utf-8")
            self.assertIn("CPUSurplusCreditBalance", credit_chart)
            self.assertIn("charged samples total 0.000", credit_chart)
            ranking_chart = (charts / "candidate_ranking.svg").read_text(encoding="utf-8")
            self.assertIn("baseline: batch", ranking_chart)

    def test_rejects_unrelated_result_kind(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            source = Path(temp_dir) / "result.json"
            source.write_text(json.dumps({"schema_version": "1.0", "kind": "other"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_training_tuning_report(source)

    def test_cli_exports_visual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source = root / "tuning_result.json"
            source.write_text(json.dumps(_result()), encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(["report-training", "--input", str(source), "--output-dir", str(root / "visuals")]),
                    0,
                )
            payload = json.loads(stdout.getvalue())
            self.assertTrue(Path(payload["report"]).exists())


def _result() -> dict:
    pairs = [(95.0, 125.0), (100.0, 130.0), (105.0, 135.0)]
    trials = [
        {
            "schema_version": "1.0",
            "candidate_id": "baseline",
            "role": "baseline",
            "phase": "search",
            "run_index": 0,
            "pair_id": None,
            "run_dir": "runs/search-baseline",
            "return_code": 0,
            "metrics": {"samples_per_second": 98.0, "step_time_ms": 10.2, "final_loss": 1.0},
            "objective_metric_value": 98.0,
            "trial_status": "valid",
            "failed": False,
        },
        {
            "schema_version": "1.0",
            "candidate_id": "selected",
            "role": "selected",
            "phase": "search",
            "run_index": 0,
            "pair_id": None,
            "run_dir": "runs/search-selected",
            "return_code": 0,
            "metrics": {"samples_per_second": 128.0, "step_time_ms": 7.8, "final_loss": 1.0},
            "objective_metric_value": 128.0,
            "trial_status": "valid",
            "failed": False,
        },
    ]
    for index, (baseline, selected) in enumerate(pairs):
        pair_id = f"confirm_{index:03d}"
        trials.extend(
            [
                {
                    "schema_version": "1.0",
                    "candidate_id": "baseline",
                    "role": "baseline",
                    "phase": "confirmation",
                    "run_index": index,
                    "pair_id": pair_id,
                    "run_dir": f"runs/{pair_id}-baseline",
                    "return_code": 0,
                    "metrics": {"samples_per_second": baseline, "step_time_ms": 1000.0 / baseline, "final_loss": 1.0},
                    "objective_metric_value": baseline,
                    "trial_status": "valid",
                    "failed": False,
                },
                {
                    "schema_version": "1.0",
                    "candidate_id": "selected",
                    "role": "selected",
                    "phase": "confirmation",
                    "run_index": index,
                    "pair_id": pair_id,
                    "run_dir": f"runs/{pair_id}-selected",
                    "return_code": 0,
                    "metrics": {"samples_per_second": selected, "step_time_ms": 1000.0 / selected, "final_loss": 1.0},
                    "objective_metric_value": selected,
                    "trial_status": "valid",
                    "failed": False,
                },
            ]
        )
    return {
        "schema_version": "1.0",
        "kind": "pytorch_training_tuning_result",
        "run_id": "test_run",
        "result_origin": "measured",
        "status": "recommended",
        "selected_candidate_id": "selected",
        "candidates": [
            {
                "candidate_id": "baseline",
                "stage": "baseline",
                "knob_values": {
                    "micro_batch_size": 8,
                    "gradient_accumulation": 4,
                    "dataloader_workers": 0,
                    "cpu_threads": 1,
                },
            },
            {
                "candidate_id": "selected",
                "stage": "batch_config",
                "knob_values": {
                    "micro_batch_size": 32,
                    "gradient_accumulation": 1,
                    "dataloader_workers": 0,
                    "cpu_threads": 1,
                },
            },
        ],
        "trials": trials,
        "statistical_evidence": {
            "schema_version": "1.0",
            "classification": "improved",
            "noise_band_percent": 5.0,
            "selected_variability_percent": 3.0,
            "paired_confirmation_count": 3,
            "failed_trials": [],
            "guard_results": [
                {
                    "schema_version": "1.0",
                    "stage": "recommendation",
                    "code": "min_valid_trials",
                    "status": "pass",
                    "message": "Enough samples.",
                },
                {
                    "schema_version": "1.0",
                    "stage": "recommendation",
                    "code": "improvement_outside_noise_band",
                    "status": "pass",
                    "message": "Outside noise.",
                },
                {
                    "schema_version": "1.0",
                    "stage": "recommendation",
                    "code": "consistent_confirmation_direction",
                    "status": "pass",
                    "message": "Direction consistent.",
                },
                {
                    "schema_version": "1.0",
                    "stage": "recommendation",
                    "code": "variability_threshold",
                    "status": "pass",
                    "message": "Variability bounded.",
                },
            ],
        },
        "search_budget": {"schema_version": "1.0", "status": "completed"},
    }


if __name__ == "__main__":
    unittest.main()
