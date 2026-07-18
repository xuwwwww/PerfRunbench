from __future__ import annotations

import io
import json
import os
import random
import stat
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotune.arm_tune.cli import main as arm_main
from autotune.arm_tune.config import parse_optimize_config
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmCpuIdentity,
    ArmHostCapabilities,
    CpuTopology,
    HostInfo,
    HostInspectionResult,
    ProductionValidation,
)
from autotune.arm_tune.search.candidates import (
    SearchCandidate,
    generate_affinity_candidates,
    generate_batch_candidates,
    generate_thread_candidates,
    generate_thread_counts,
)
from autotune.arm_tune.search.runner import SearchConfig, build_stage_schedule, run_search
from autotune.arm_tune.search.statistics import classify_noise_band, summarize_metric
from autotune.arm_tune.targets.local import LocalArmTarget


FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "arm_tune" / "fake_llama_bench.py"


class ArmTuneSearchTest(unittest.TestCase):
    def test_thread_candidate_generation_deduplicates_visible_topology(self) -> None:
        topology = CpuTopology(logical_cpus=8, cores_per_socket=4, sockets=1, threads_per_core=1, online_cpus="0-7")
        counts = generate_thread_counts(topology, baseline_threads=4)

        self.assertEqual(counts, [1, 2, 4, 7, 8])

    def test_thread_candidates_include_baseline_and_valid_counts(self) -> None:
        _, job, _ = parse_optimize_config(_config("/bin/echo", "/tmp/model.gguf"))
        candidates = generate_thread_candidates(job, CpuTopology(logical_cpus=4, online_cpus="0-3"))

        self.assertEqual(candidates[0].candidate_id, "baseline")
        self.assertTrue(candidates[0].is_baseline)
        self.assertEqual(sorted(candidate.threads for candidate in candidates), [1, 2, 3, 4])

    def test_unsupported_optional_flags_do_not_generate_batch_candidates(self) -> None:
        _, job, _ = parse_optimize_config(_config("/bin/echo", "/tmp/model.gguf"))
        base = SearchCandidate(candidate_id="threads-2", stage="thread_count", threads=2)

        candidates = generate_batch_candidates(base, job, supported_flags=["--model", "--threads"])

        self.assertEqual(candidates, [])

    def test_supported_optional_flags_generate_threads_batch_and_batch_candidate(self) -> None:
        _, job, _ = parse_optimize_config(_config("/bin/echo", "/tmp/model.gguf"))
        base = SearchCandidate(candidate_id="threads-2", stage="thread_count", threads=2)

        candidates = generate_batch_candidates(
            base,
            job,
            supported_flags=["--threads-batch", "--batch-size", "--ubatch-size"],
        )

        self.assertEqual(len(candidates), 1)
        options = candidates[0].additional_supported_options
        self.assertEqual(options["threads_batch"], 2)
        self.assertIn("batch_size", options)
        self.assertIn("ubatch_size", options)

    def test_randomized_order_is_reproducible_and_paired_baseline_is_inserted(self) -> None:
        baseline = SearchCandidate(candidate_id="baseline", stage="baseline", threads=1, is_baseline=True)
        candidates = [
            baseline,
            SearchCandidate(candidate_id="threads-2", stage="thread_count", threads=2),
            SearchCandidate(candidate_id="threads-4", stage="thread_count", threads=4),
        ]

        first = build_stage_schedule(
            candidates=candidates,
            baseline=baseline,
            repetitions_per_candidate=2,
            insert_paired_baseline_every=2,
            rng=random.Random(7),
        )
        second = build_stage_schedule(
            candidates=candidates,
            baseline=baseline,
            repetitions_per_candidate=2,
            insert_paired_baseline_every=2,
            rng=random.Random(7),
        )

        self.assertEqual([item.candidate_id for item in first], [item.candidate_id for item in second])
        self.assertGreater([item.candidate_id for item in first].count("baseline"), 2)

    def test_metric_distribution_preserves_failures_and_warns_on_small_p95(self) -> None:
        distribution = summarize_metric("decode_tokens_per_second", [10.0, 12.0, 14.0], failed_count=2)

        self.assertEqual(distribution.valid_count, 3)
        self.assertEqual(distribution.failed_count, 2)
        self.assertEqual(distribution.median, 12.0)
        self.assertIsNone(distribution.p95)
        self.assertTrue(distribution.warnings)

    def test_metric_distribution_reports_p95_when_sample_count_supports_it(self) -> None:
        distribution = summarize_metric("decode_tokens_per_second", [10, 11, 12, 13, 14], failed_count=0)

        self.assertEqual(distribution.valid_count, 5)
        self.assertIsNotNone(distribution.p95)
        self.assertFalse(distribution.warnings)

    def test_noise_band_classification_marks_inconclusive(self) -> None:
        baseline = summarize_metric("decode_tokens_per_second", [100, 101, 99, 100, 100], failed_count=0)
        candidate = summarize_metric("decode_tokens_per_second", [101, 100, 101, 100, 101], failed_count=0)

        decision = classify_noise_band(baseline=baseline, candidate=candidate, noise_band_percent=2.0)

        self.assertEqual(decision.classification, "inconclusive")

    def test_affinity_candidates_are_validated_and_numa_local_requires_numa(self) -> None:
        base = SearchCandidate(candidate_id="threads-2", stage="thread_count", threads=2)

        single_numa = generate_affinity_candidates(base, CpuTopology(logical_cpus=4, online_cpus="0-3", numa_nodes=1))
        multi_numa = generate_affinity_candidates(base, CpuTopology(logical_cpus=4, online_cpus="0-3", numa_nodes=2))

        self.assertNotIn("numa-local", {candidate.affinity_mode for candidate in single_numa})
        self.assertIn("numa-local", {candidate.affinity_mode for candidate in multi_numa})
        for candidate in multi_numa:
            self.assertTrue(candidate.cpu_affinity)
            self.assertEqual(len(candidate.cpu_affinity), len(set(candidate.cpu_affinity)))

    def test_affinity_candidates_honor_allowed_modes(self) -> None:
        base = SearchCandidate(candidate_id="threads-2", stage="thread_count", threads=2)
        topology = CpuTopology(logical_cpus=4, online_cpus="0-3", numa_nodes=1)

        compact_only = generate_affinity_candidates(base, topology, allowed_modes=["scheduler-default", "compact"])
        disabled = generate_affinity_candidates(base, topology, allowed_modes=[])

        self.assertEqual([candidate.affinity_mode for candidate in compact_only], ["compact"])
        self.assertEqual(disabled, [])

    def test_search_config_caps_candidates_and_filters_affinity_modes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job, search_config = parse_optimize_config(_config(executable, model))
            target = _SyntheticLocalTarget(logical_cpus=2)
            optimize_dir = root / "run" / "armtune" / "optimize"
            search_config = replace(
                search_config,
                warmup_runs=0,
                search_runs=1,
                confirmation_runs=1,
                cooldown_seconds=0,
                random_seed=3,
                max_candidates=3,
                affinity_modes=["compact"],
            )

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_SCALE_BY_THREADS": "1"}):
                result = run_search(
                    target=target,
                    base_job=replace(job, run_id="capped-search", target_run_directory=str(optimize_dir)),
                    config=search_config,
                    run_id="capped-search",
                    optimize_dir=optimize_dir,
                )

            candidate_ids = [candidate.candidate_id for candidate in result.candidates]
            affinity_stage = next(stage for stage in result.stages if stage.stage == "affinity")

            self.assertEqual(candidate_ids, ["baseline", "threads-2", "affinity-compact-0_1"])
            self.assertLessEqual(len(candidate_ids), 3)
            self.assertEqual(affinity_stage.candidate_ids, ["threads-2", "affinity-compact-0_1"])

    def test_fake_full_search_e2e_records_quality_gate_and_warmup_exclusion(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job, search_config = parse_optimize_config(_config(executable, model))
            target = _SyntheticLocalTarget(logical_cpus=2)
            optimize_dir = root / "run" / "armtune" / "optimize"
            search_config = replace(
                search_config,
                warmup_runs=1,
                search_runs=1,
                confirmation_runs=1,
                cooldown_seconds=0,
                random_seed=11,
                insert_paired_baseline_every=2,
            )

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_HELP_MODE": "full", "FAKE_LLAMA_BENCH_SCALE_BY_THREADS": "1"}):
                result = run_search(
                    target=target,
                    base_job=replace(job, run_id="search-run", target_run_directory=str(optimize_dir)),
                    config=search_config,
                    run_id="search-run",
                    optimize_dir=optimize_dir,
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.quality_gate_status, "not_evaluated")
            self.assertTrue((optimize_dir / "search_result.json").exists())
            self.assertTrue((optimize_dir / "search_plan.json").exists())
            self.assertTrue((optimize_dir / "repetitions" / "0001" / "raw").exists())
            self.assertLess(
                result.distributions["baseline"]["decode_tokens_per_second"].valid_count,
                sum(1 for record in result.repetitions if record.candidate_id == "baseline"),
            )
            self.assertEqual([stage.stage for stage in result.stages], ["thread_count", "batch_parameters", "affinity"])

    def test_failed_search_trials_are_preserved(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job, search_config = parse_optimize_config(_config(executable, model))
            target = _SyntheticLocalTarget(logical_cpus=1)
            search_config = replace(search_config, warmup_runs=0, search_runs=1, confirmation_runs=1, cooldown_seconds=0)

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_MODE": "nonzero"}):
                result = run_search(
                    target=target,
                    base_job=replace(job, run_id="failed-search"),
                    config=search_config,
                    run_id="failed-search",
                    optimize_dir=root / "run" / "armtune" / "optimize",
                )

            self.assertTrue(any(record.benchmark_status == "failed" for record in result.repetitions))
            self.assertTrue(any(record.failure_reason for record in result.repetitions))
            self.assertGreater(result.distributions["baseline"]["decode_tokens_per_second"].failed_count, 0)

    def test_arm_optimize_cli_help_and_exit_code(self) -> None:
        help_output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(help_output):
            arm_main(["optimize", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--warmup-runs", help_output.getvalue())
        self.assertIn("--max-candidates", help_output.getvalue())
        self.assertIn("--affinity-modes", help_output.getvalue())

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            config_path = _write_config(
                root,
                _config(
                    executable,
                    model,
                    search={
                        "warmup_runs": 0,
                        "search_runs": 1,
                        "confirmation_runs": 1,
                        "cooldown_seconds": 0,
                        "random_seed": 5,
                    },
                ),
            )
            with _cwd(root), patch("autotune.arm_tune.cli._target_from_spec", return_value=_SyntheticLocalTarget(logical_cpus=2)):
                with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_HELP_MODE": "full"}), redirect_stdout(io.StringIO()):
                    exit_code = arm_main(
                        [
                            "optimize",
                            "--config",
                            str(config_path),
                            "--max-candidates",
                            "2",
                            "--affinity-modes",
                            "compact",
                        ]
                    )

        self.assertEqual(exit_code, 0)


class _SyntheticLocalTarget(LocalArmTarget):
    def __init__(self, *, logical_cpus: int) -> None:
        super().__init__()
        self.logical_cpus = logical_cpus

    def inspect(self) -> HostInspectionResult:
        return HostInspectionResult(
            target_type="local",
            host=_host(self.logical_cpus),
            production_validation=ProductionValidation(
                valid=True,
                reason="synthetic test target",
                observed_system="Linux",
                observed_machine="aarch64",
            ),
        )


def _config(executable_path: str | Path, model_path: str | Path, *, search: dict | None = None) -> dict:
    payload = {
        "schema_version": "1.0",
        "result_origin": "test_fixture",
        "target": {"type": "local"},
        "executable": {"path": str(executable_path)},
        "model": {"path": str(model_path)},
        "benchmark": {
            "threads": 1,
            "prompt_tokens": 16,
            "generation_tokens": 8,
            "repetitions": 1,
            "timeout_seconds": 5,
        },
        "artifacts": {"preserve_raw_stdout": True, "preserve_raw_stderr": True},
    }
    if search is not None:
        payload["search"] = search
    return payload


def _host(logical_cpus: int) -> HostInfo:
    return HostInfo(
        architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", is_linux=True, is_arm64=True),
        cpu_identity=ArmCpuIdentity(model_name="Neoverse-N1", source="synthetic-test"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="synthetic-test"),
        topology=CpuTopology(logical_cpus=logical_cpus, cores_per_socket=logical_cpus, sockets=1, threads_per_core=1, numa_nodes=1, online_cpus=f"0-{logical_cpus - 1}"),
        os_release="Synthetic Linux",
    )


def _fake_executable(root: Path) -> Path:
    if os.name == "nt":
        wrapper = root / "fake_llama_bench.cmd"
        wrapper.write_text(f"@echo off\r\n\"{sys.executable}\" \"{FIXTURE_SCRIPT}\" %*\r\n", encoding="utf-8")
    else:
        wrapper = root / "fake_llama_bench"
        wrapper.write_text(f"#!/usr/bin/env sh\nexec \"{sys.executable}\" \"{FIXTURE_SCRIPT}\" \"$@\"\n", encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def _model_file(root: Path) -> Path:
    model = root / "model.gguf"
    model.write_bytes(b"synthetic model placeholder")
    return model


def _write_config(root: Path, payload: dict) -> Path:
    path = root / "optimize.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


if __name__ == "__main__":
    unittest.main()
