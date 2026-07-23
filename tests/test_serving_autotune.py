from __future__ import annotations

import io
import json
import socket
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.cli import main
from autotune.serving.capabilities import discover_server_capabilities
from autotune.serving.lifecycle import check_server_lifecycle_preflight
from autotune.serving.network import create_phase21d2_network_plan
from autotune.serving.recommendations import (
    CANONICAL_PROFILE_ID,
    canonical_remote_recommendations,
    get_remote_recommendation,
    recommendation_bundle,
    serving_performance_passport,
)
from autotune.serving.safety import validate_serving_reuse
from autotune.serving.search import create_serving_plan, run_fixture_serving_search, serving_passport_extension
from autotune.serving.schemas import ServerRuntimeProfile, ServingExperimentResult, ServingMetricSummary, ServingTrafficProfile


class ServingAutotuneTest(unittest.TestCase):
    def test_capability_discovery_marks_supported_and_unsupported_options(self) -> None:
        report = discover_server_capabilities(
            help_text="server build: b9999 --parallel --threads --ctx-size --batch-size /health /metrics stream seed max_tokens",
            endpoint_probe={"slots": False},
            source_commit="b9999",
            architecture="aarch64",
            backend="cpu",
            kernel_identity="6.17.0-1019-aws",
        )

        self.assertEqual(report.runtime_identity.source_commit, "b9999")
        self.assertEqual(report.options["parallel_slots"].availability, "available")
        self.assertEqual(report.options["continuous_batching"].availability, "not_available")
        self.assertEqual(report.options["kv_cache_configuration"].availability, "not_available")
        self.assertEqual(report.endpoints["metrics"].availability, "available")
        self.assertIn("stream", report.supported_request_fields)
        self.assertTrue(report.capability_digest)

    def test_candidate_generation_respects_capabilities_context_memory_and_cap(self) -> None:
        report = discover_server_capabilities(
            help_text="--parallel --threads --ctx-size --batch-size --ubatch-size --mmap --mlock",
            source_commit="b9999",
        )
        plan = create_serving_plan(
            workload_id="prefill-heavy",
            traffic_mode="closed-loop",
            concurrency=4,
            candidate_cap=5,
            capability_report=report,
            logical_cpus=8,
            available_memory_bytes=8 * 1024 * 1024 * 1024,
        )

        self.assertEqual(plan.candidates[0].profile_id, "baseline")
        self.assertLessEqual(len(plan.candidates), 5)
        self.assertTrue(all(candidate.effective_context_per_slot is None or candidate.effective_context_per_slot >= plan.workload.max_context_tokens_required for candidate in plan.candidates))
        self.assertFalse(any(candidate.profile_id.startswith("kv-cache") for candidate in plan.candidates))

    def test_fixture_search_produces_protocol_specific_pareto_and_guards(self) -> None:
        plan = create_serving_plan(workload_id="interactive-short", traffic_mode="closed-loop", concurrency=4, candidate_cap=8)
        results, pareto = run_fixture_serving_search(plan)

        self.assertEqual(pareto.result_origin, "test_fixture")
        self.assertEqual(pareto.baseline_candidate_id, "baseline")
        self.assertGreaterEqual(len(pareto.frontier_points), 1)
        self.assertIn("interactive_latency", pareto.recommendations)
        self.assertIn("maximum_throughput", pareto.recommendations)
        self.assertTrue(all(point.quality_guard_status in {"passed", "failed"} for point in pareto.frontier_points))
        self.assertGreaterEqual(len(results), 1)

    def test_shared_prefix_recommendation_is_workload_scoped(self) -> None:
        shared = create_serving_plan(workload_id="shared-prefix", traffic_mode="shared-prefix", concurrency=2, candidate_cap=20)
        _, shared_pareto = run_fixture_serving_search(shared)
        normal = create_serving_plan(workload_id="interactive-short", traffic_mode="closed-loop", concurrency=2, candidate_cap=12)
        _, normal_pareto = run_fixture_serving_search(normal)

        self.assertEqual(shared_pareto.recommendations["shared_prefix"].status, "recommended")
        self.assertEqual(normal_pareto.recommendations["shared_prefix"].status, "no_eligible_recommendation")

    def test_quality_failure_is_excluded_from_recommendation(self) -> None:
        plan = create_serving_plan(workload_id="decode-heavy", candidate_cap=20)
        results, pareto = run_fixture_serving_search(plan)
        failing_ids = [result.candidate_id for result in results if result.quality_guard_status == "failed"]

        self.assertIn("kv-cache-q8", failing_ids)
        self.assertNotEqual(pareto.recommendations["maximum_throughput"].candidate_id, "kv-cache-q8")

    def test_serving_passport_extension_contains_drift_keys_and_serving_fields(self) -> None:
        plan = create_serving_plan(workload_id="interactive-short", candidate_cap=4)
        results, pareto = run_fixture_serving_search(plan)
        extension = serving_passport_extension(
            plan=plan,
            pareto=pareto,
            result=results[0],
            server_binary_digest="binary",
            model_digest="model",
        )

        self.assertEqual(extension["schema_version"], "1.0")
        self.assertEqual(extension["serving_workload_id"], "interactive-short")
        self.assertIn("server_capability_digest", extension)
        self.assertIn("workload_shape", extension["hard_drift_keys"])
        self.assertIn("concurrency_class", extension["hard_drift_keys"])

    def test_network_plan_is_plan_only_and_separates_loopback_from_remote(self) -> None:
        plan = create_phase21d2_network_plan()

        self.assertTrue(plan.loopback_results_excluded_from_remote_claims)
        self.assertEqual(plan.topology, "c7g_server_t4g_remote_load_generator_planned")
        self.assertEqual(plan.orchestration.phase, "21D2A")
        self.assertFalse(plan.orchestration.aws_accessed)
        self.assertFalse(plan.orchestration.remote_accessed)
        self.assertEqual(plan.orchestration.required_profile, "slots4-cont-on")
        self.assertEqual(plan.orchestration.server_role.expected_cpu_class, "neoverse_v1")
        self.assertEqual(plan.orchestration.server_role.expected_logical_cpus, 8)
        self.assertEqual(plan.orchestration.client_role.expected_cpu_class, "neoverse_n1")
        self.assertEqual(plan.orchestration.client_role.expected_logical_cpus, 2)
        self.assertTrue(all(profile.reversible for profile in plan.irq_profiles))
        self.assertTrue(any("listen_overflow_or_backlog_saturation" in profile.evidence_required for profile in plan.queue_profiles))
        self.assertEqual(
            plan.network_bottleneck_gate.default_decision,
            "skip_irq_rps_xps_until_network_bottleneck_is_observed",
        )
        self.assertIn("blind_rps_or_xps_change", plan.network_bottleneck_gate.forbidden_without_evidence)
        self.assertEqual(
            plan.recommendation_reuse_policy["phase21d1_loopback_recommendations"],
            "hard_drift_for_remote_network_topology",
        )

    def test_remote_serving_recommendations_are_evidence_scoped(self) -> None:
        recommendations = canonical_remote_recommendations()
        classes = {item["recommendation_class"] for item in recommendations}
        workloads = {item["serving_workload_id"] for item in recommendations}

        self.assertEqual(classes, {"interactive_latency_remote", "maximum_throughput_remote"})
        self.assertEqual(workloads, {"interactive-short", "decode-heavy", "prefill-heavy"})
        self.assertEqual(len(recommendations), 6)
        self.assertFalse(any(item["recommendation_class"] == "balanced_multi_user" for item in recommendations))
        self.assertFalse(any(item["recommendation_class"] == "shared_prefix" for item in recommendations))
        self.assertFalse(any(item["recommendation_class"] == "remote_network" for item in recommendations))

        decode = get_remote_recommendation(
            workload_id="decode-heavy",
            concurrency=4,
            recommendation_class="maximum_throughput_remote",
        )
        self.assertEqual(decode["profile_id"], CANONICAL_PROFILE_ID)
        self.assertEqual(decode["metrics"]["throughput_delta_percent"], 86.490)
        self.assertEqual(decode["metrics"]["ttft_p95_reduction_percent"], 98.001)
        self.assertEqual(decode["result_origin"], "measured")

    def test_remote_serving_bundle_and_passport_are_versioned_and_sanitized(self) -> None:
        recommendation = get_remote_recommendation(workload_id="interactive-short", concurrency=4)
        bundle = recommendation_bundle(recommendation)
        passport = serving_performance_passport(recommendation)
        serialized = json.dumps({"bundle": bundle, "passport": passport})

        self.assertEqual(bundle["schema_version"], "1.0")
        self.assertEqual(passport["schema_version"], "1.0")
        self.assertEqual(passport["passport_kind"], "serving_performance_passport")
        self.assertEqual(passport["verification_status"], "passed")
        self.assertIn("network_tuning_not_executed_evidence_absent", serialized)
        self.assertNotRegex(serialized, r"\b\d{1,3}(\.\d{1,3}){3}\b")
        self.assertNotIn(".pem", serialized)
        self.assertNotIn("PRIVATE KEY", serialized)

    def test_cli_recommend_explain_and_run_best_dry_run_do_not_search(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_path = root / "bundle.json"
            passport_path = root / "passport.json"
            run_path = root / "run_best.json"

            recommend_output = io.StringIO()
            with redirect_stdout(recommend_output):
                self.assertEqual(
                    main(
                        [
                            "serve",
                            "recommend",
                            "--workload",
                            "interactive-short",
                            "--concurrency",
                            "4",
                            "--recommendation-class",
                            "maximum_throughput_remote",
                            "--output",
                            str(bundle_path),
                            "--passport-output",
                            str(passport_path),
                        ]
                    ),
                    0,
                )
            self.assertTrue(bundle_path.exists())
            self.assertTrue(passport_path.exists())
            self.assertIn("result_origin: measured", recommend_output.getvalue())

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            explain_output = io.StringIO()
            with redirect_stdout(explain_output):
                self.assertEqual(main(["serve", "explain", "--recommendation", bundle["bundle_id"]]), 0)
            self.assertIn("network tuning was not executed".lower(), explain_output.getvalue().lower())

            run_output = io.StringIO()
            with redirect_stdout(run_output):
                self.assertEqual(
                    main(
                        [
                            "serve",
                            "run",
                            "--best",
                            "--workload",
                            "interactive-short",
                            "--concurrency",
                            "4",
                            "--dry-run",
                            "--output",
                            str(run_path),
                        ]
                    ),
                    0,
                )
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["run_best_performs_search"])
            self.assertFalse(payload["process_started"])
            self.assertEqual(payload["validation"]["state"], "recommendation_applicable")
            self.assertIn("--parallel", payload["expected_command"])
            self.assertIn("--cont-batching", payload["expected_command"])

    def test_cli_run_best_reports_hard_drift_without_retuning(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "serve",
                    "run",
                    "--best",
                    "--workload",
                    "interactive-short",
                    "--concurrency",
                    "4",
                    "--current-network-topology",
                    "loopback",
                    "--json",
                ]
            )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 2)
        self.assertFalse(payload["run_best_performs_search"])
        self.assertFalse(payload["process_started"])
        self.assertEqual(payload["validation"]["state"], "topology_mismatch")
        failed = [item for item in payload["validation"]["guard_results"] if item["status"] == "failed"]
        self.assertTrue(any(item["field"] == "network_topology" for item in failed))

    def test_lifecycle_preflight_refuses_occupied_port_and_stale_server(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = int(listener.getsockname()[1])
            occupied = check_server_lifecycle_preflight(port=port)

        stale = check_server_lifecycle_preflight(port=port, process_listing="123 llama-server --model fixture")

        self.assertFalse(occupied.can_start)
        self.assertIn("unknown_listener_occupies_port", occupied.issues)
        self.assertFalse(stale.can_start)
        self.assertIn("stale_llama_server_process_detected", stale.issues)

    def test_serving_reuse_guard_rejects_workload_and_capability_hard_drift(self) -> None:
        bundle = {
            "schema_version": "1.0",
            "server_binary_digest": "binary",
            "model_digest": "model",
            "serving_workload_id": "interactive-short",
            "traffic_mode": "closed-loop",
            "concurrency": 4,
            "network_profile": {"schema_version": "1.0", "topology": "loopback"},
            "server_capability_digest": "capability-a",
            "runtime_identity": {"schema_version": "1.0", "architecture": "aarch64", "kernel_identity": "4k"},
        }
        current = dict(bundle)
        current["serving_workload_id"] = "decode-heavy"
        current["server_capability_digest"] = "capability-b"

        result = validate_serving_reuse(bundle, current)

        self.assertEqual(result["status"], "failed")
        failed = {guard["field"] for guard in result["guard_results"] if guard["status"] == "failed"}
        self.assertIn("serving_workload_id", failed)
        self.assertIn("server_capability_digest", failed)

    def test_serving_reuse_guard_rejects_loopback_bundle_on_remote_network(self) -> None:
        bundle = {
            "schema_version": "1.0",
            "server_binary_digest": "binary",
            "model_digest": "model",
            "serving_workload_id": "interactive-short",
            "traffic_mode": "closed-loop",
            "concurrency": 4,
            "network_profile": {"schema_version": "1.0", "topology": "loopback"},
            "server_capability_digest": "capability",
            "runtime_identity": {"schema_version": "1.0", "architecture": "aarch64", "kernel_identity": "4k"},
        }
        current = dict(bundle)
        current["network_profile"] = {"schema_version": "1.0", "topology": "c7g_server_t4g_remote_client"}

        result = validate_serving_reuse(bundle, current)

        self.assertEqual(result["status"], "failed")
        network_guard = next(guard for guard in result["guard_results"] if guard["field"] == "network_topology")
        self.assertEqual(network_guard["reason"], "hard_drift")

    def test_cli_doctor_plan_search_pareto_and_network_fake_e2e(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "serving_plan.json"
            search_dir = root / "search"

            doctor_output = io.StringIO()
            with redirect_stdout(doctor_output):
                self.assertEqual(main(["serve", "doctor", "--json"]), 0)
            self.assertIn('"aws_accessed": false', doctor_output.getvalue())

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "serve",
                            "plan",
                            "--workload",
                            "interactive-short",
                            "--traffic",
                            "closed-loop",
                            "--concurrency",
                            "4",
                            "--candidate-cap",
                            "6",
                            "--output",
                            str(plan_path),
                        ]
                    ),
                    0,
                )
            self.assertTrue(plan_path.exists())
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan_payload["schema_version"], "1.0")
            self.assertLessEqual(len(plan_payload["candidates"]), 6)

            run_dir = root / "run"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["serve", "run", "--plan", str(plan_path), "--fixture", "--output-dir", str(run_dir)]), 0)
            self.assertTrue((run_dir / "serving_experiment_result.json").exists())

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "serve",
                            "search",
                            "--workload",
                            "interactive-short",
                            "--fixture",
                            "--candidate-cap",
                            "6",
                            "--output-dir",
                            str(search_dir),
                        ]
                    ),
                    0,
                )
            self.assertTrue((search_dir / "serving_pareto_result.json").exists())
            pareto_output = io.StringIO()
            with redirect_stdout(pareto_output):
                self.assertEqual(main(["serve", "pareto", "--experiment", str(search_dir)]), 0)
            self.assertIn('"result_origin": "test_fixture"', pareto_output.getvalue())

            baseline_path = search_dir / "candidates" / "baseline" / "serving_experiment_result.json"
            candidate_path = next(path for path in (search_dir / "candidates").glob("*/serving_experiment_result.json") if "baseline" not in path.parts)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "serve",
                            "compare",
                            "--baseline",
                            str(baseline_path),
                            "--candidate",
                            str(candidate_path),
                            "--output",
                            str(root / "compare.json"),
                        ]
                    ),
                    0,
                )

            network_path = root / "network_plan.json"
            network_output = io.StringIO()
            with redirect_stdout(network_output):
                self.assertEqual(main(["serve", "network-plan", "--output", str(network_path)]), 0)
            self.assertTrue(network_path.exists())
            network_payload = json.loads(network_path.read_text(encoding="utf-8"))
            self.assertEqual(network_payload["orchestration"]["phase"], "21D2A")
            self.assertFalse(network_payload["orchestration"]["aws_accessed"])
            self.assertFalse(network_payload["orchestration"]["remote_accessed"])
            self.assertIn("required_profile: slots4-cont-on", network_output.getvalue())
            serialized_network = json.dumps(network_payload)
            self.assertNotRegex(serialized_network, r"\b\d{1,3}(\.\d{1,3}){3}\b")
            self.assertNotIn(".pem", serialized_network)
            self.assertNotIn("PRIVATE KEY", serialized_network)

            network_json_output = io.StringIO()
            with redirect_stdout(network_json_output):
                self.assertEqual(main(["serve", "network-plan", "--output", str(root / "network_plan_json.json"), "--json"]), 0)
            self.assertIn('"topology_id": "c7g_server_t4g_remote_client"', network_json_output.getvalue())

    def test_compare_refuses_mixed_workload_protocols(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = _result(root, "baseline", "interactive-short", "closed-loop")
            candidate = _result(root, "candidate", "decode-heavy", "closed-loop")
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate.to_dict()), encoding="utf-8")

            with self.assertRaises(SystemExit):
                main(["serve", "compare", "--baseline", str(baseline_path), "--candidate", str(candidate_path)])


def _result(root: Path, candidate_id: str, workload_id: str, traffic: str) -> ServingExperimentResult:
    del root
    profile = ServerRuntimeProfile(profile_id=candidate_id)
    return ServingExperimentResult(
        experiment_id="experiment",
        candidate_id=candidate_id,
        workload_id=workload_id,
        traffic_profile=ServingTrafficProfile(traffic_mode=traffic),
        runtime_profile=profile,
        result_origin="test_fixture",
        request_count=5,
        failed_request_count=0,
        metrics={
            "aggregate_output_tokens_per_second": ServingMetricSummary("aggregate_output_tokens_per_second", valid_count=1, failed_count=0, p50=100.0),
            "ttft_ms": ServingMetricSummary("ttft_ms", valid_count=5, failed_count=0, p95=100.0),
        },
        quality_guard_status="passed",
        cleanup_status="passed",
    )


if __name__ == "__main__":
    unittest.main()
