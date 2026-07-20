from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PerfRunbench from an installed wheel in a fresh environment.")
    parser.add_argument("--wheel", help="Wheel file to install. Defaults to the newest dist/*.whl.")
    parser.add_argument("--summary", help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    wheel = Path(args.wheel).resolve() if args.wheel else _latest_wheel(ROOT / "dist").resolve()
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "perfrunbench_release_smoke",
        "wheel": str(wheel),
        "checks": [],
    }
    with tempfile.TemporaryDirectory(prefix="perfrunbench_release_smoke_") as temp_dir:
        temp = Path(temp_dir)
        venv = temp / "fresh-venv"
        workspace = temp / "fresh user workspace"
        workspace.mkdir()
        _create_venv(venv)
        python = _venv_python(venv)
        autotuneai = _venv_script(venv, "autotuneai")
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=temp, summary=summary, name="pip_upgrade")
        _run([str(python), "-m", "pip", "install", str(wheel)], cwd=temp, summary=summary, name="wheel_install")
        source_check = _run(
            [
                str(python),
                "-c",
                "import json, pathlib, autotune; print(json.dumps({'autotune_file': pathlib.Path(autotune.__file__).as_posix()}))",
            ],
            cwd=workspace,
            summary=summary,
            name="import_location",
        )
        installed_path = json.loads(source_check.stdout.strip())["autotune_file"]
        if Path(installed_path).resolve().is_relative_to(ROOT.resolve()):
            raise SmokeFailure(f"installed import resolved to source checkout: {installed_path}")

        _write_training_fixture(workspace / "train.py")
        _run([str(autotuneai), "--help"], cwd=workspace, summary=summary, name="console_help")
        _run([str(autotuneai), "init", "--adapter", "pytorch-training", "--output", "autotune_tune.yaml"], cwd=workspace, summary=summary, name="init_tune_config")
        _run(
            [
                str(autotuneai),
                "doctor",
                "--json",
                "--config",
                "autotune_tune.yaml",
                "--",
                str(python),
                "train.py",
                "--label",
                "value with spaces",
                "--unicode",
                "\u6e2c\u8a66",
            ],
            cwd=workspace,
            summary=summary,
            name="doctor_json",
        )
        tune = _run(
            [str(autotuneai), "tune", "--config", "autotune_tune.yaml", "--", str(python), "train.py"],
            cwd=workspace,
            summary=summary,
            name="tune_fixture",
        )
        tune_payload = _json_output(tune.stdout, "tune_fixture")
        _require(tune_payload.get("status") == "recommended", "tune_fixture did not produce a recommendation")
        passport_path = _workspace_path(workspace, str(tune_payload.get("performance_passport_path")))
        _require(passport_path.exists(), "tune_fixture passport path is missing")

        _run([str(autotuneai), "init", "--adapter", "pytorch-training", "--output", "autotune_auto.yaml"], cwd=workspace, summary=summary, name="init_auto_config")
        missing = _run(
            [str(autotuneai), "run", "--best", "--config", "autotune_auto.yaml", "--", str(python), "train.py"],
            cwd=workspace,
            summary=summary,
            name="missing_cache_error",
            check=False,
        )
        _require(missing.returncode != 0 and "no usable eligible cached recommendation" in missing.stdout, "missing-cache error was not actionable")
        first_auto = _run(
            [str(autotuneai), "run", "--auto", "--config", "autotune_auto.yaml", "--", str(python), "train.py"],
            cwd=workspace,
            summary=summary,
            name="auto_cache_miss",
        )
        first_auto_payload = _json_output(first_auto.stdout, "auto_cache_miss")
        _require(first_auto_payload.get("newly_tuned") is True, "auto cache miss did not tune")
        second_auto = _run(
            [str(autotuneai), "run", "--auto", "--config", "autotune_auto.yaml", "--", str(python), "train.py"],
            cwd=workspace,
            summary=summary,
            name="auto_cache_hit",
        )
        second_auto_payload = _json_output(second_auto.stdout, "auto_cache_hit")
        _require(second_auto_payload.get("newly_tuned") is False, "auto cache hit retuned unexpectedly")
        _require(
            _is_reused_cache_state(second_auto_payload.get("cache_state")),
            f"auto cache hit did not reuse a compatible cache entry: {second_auto_payload.get('cache_state')}",
        )
        best = _run(
            [str(autotuneai), "run", "--best", "--config", "autotune_auto.yaml", "--", str(python), "train.py"],
            cwd=workspace,
            summary=summary,
            name="run_best",
        )
        best_payload = _json_output(best.stdout, "run_best")
        _require(best_payload.get("newly_tuned") is False, "run --best retuned unexpectedly")

        cache_root = workspace / ".autotuneai" / "cache" / "recommendations"
        listed = _run([str(autotuneai), "cache", "--root", str(cache_root), "list"], cwd=workspace, summary=summary, name="cache_list")
        entries = _json_output(listed.stdout, "cache_list").get("entries", [])
        _require(isinstance(entries, list) and entries, "cache list did not return entries")
        cache_id = str(entries[0]["cache_id"])
        _run([str(autotuneai), "cache", "--root", str(cache_root), "show", cache_id], cwd=workspace, summary=summary, name="cache_show")
        cache_copy = workspace / "cache copy"
        cache_copy.mkdir()
        shutil.copy2(cache_root / f"{cache_id}.json", cache_copy / f"{cache_id}.json")
        _run([str(autotuneai), "cache", "--root", str(cache_copy), "invalidate", cache_id], cwd=workspace, summary=summary, name="cache_invalidate_copy")
        _run([str(autotuneai), "cache", "--root", str(cache_copy), "gc"], cwd=workspace, summary=summary, name="cache_gc_copy")

        arm_result = _run_arm_fixture(autotuneai, python, workspace, summary)
        _require(arm_result.get("result_origin") == "test_fixture", "ArmTune fixture result_origin was not test_fixture")
        _require(_no_active_leases(cache_root), "active recommendation lease remains after release smoke")
        _require(_no_child_processes(), "release smoke left residual child processes")
        summary.update(
            {
                "status": "passed",
                "workspace": str(workspace),
                "cache_id": cache_id,
                "auto_cache_miss_newly_tuned": first_auto_payload.get("newly_tuned"),
                "auto_cache_hit_state": second_auto_payload.get("cache_state"),
                "run_best_newly_tuned": best_payload.get("newly_tuned"),
                "arm_fixture_result_origin": arm_result.get("result_origin"),
            }
        )

    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


class SmokeFailure(RuntimeError):
    pass


def _latest_wheel(dist: Path) -> Path:
    wheels = sorted(dist.glob("*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not wheels:
        raise SmokeFailure(f"no wheel found under {dist}")
    return wheels[0]


def _create_venv(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(venv: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _run(
    command: list[str],
    *,
    cwd: Path,
    summary: dict[str, Any],
    name: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        check=False,
    )
    summary["checks"].append(
        {
            "name": name,
            "return_code": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    )
    if check and completed.returncode != 0:
        raise SmokeFailure(f"{name} failed with return code {completed.returncode}: {completed.stderr or completed.stdout}")
    return completed


def _json_output(text: str, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{name} did not emit a JSON object") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"{name} JSON output was not an object")
    return payload


def _write_training_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse, os, time",
                "from autotune.training_metrics import emit_result, emit_step",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--micro-batch-size', type=int, default=4)",
                "parser.add_argument('--gradient-accumulation', type=int, default=2)",
                "parser.add_argument('--num-workers', type=int, default=0)",
                "parser.add_argument('--max-steps', type=int, default=None)",
                "parser.add_argument('--label', default='')",
                "parser.add_argument('--unicode', default='')",
                "args = parser.parse_args()",
                "steps = int(args.max_steps or 3)",
                "cpu_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))",
                "throughput = 100.0 + args.micro_batch_size * 10.0 + args.num_workers * 4.0 + cpu_threads * 3.0",
                "if args.gradient_accumulation == 1:",
                "    throughput += 8.0",
                "step_time_ms = 1000.0 / throughput",
                "warmup_steps = 1 if args.max_steps else 0",
                "measured_steps = max(1, steps - warmup_steps)",
                "for step in range(1, steps + 1):",
                "    emit_step(completed_steps=step, metrics={'step_time_ms': step_time_ms})",
                "emit_result(completed_steps=measured_steps, metrics={'completed_steps': measured_steps, 'samples_per_second': throughput, 'step_time_ms': step_time_ms, 'final_loss': 0.1, 'examples_processed': measured_steps * args.micro_batch_size, 'slice_limit_applied': args.max_steps is not None})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_arm_fixture(autotuneai: Path, python: Path, workspace: Path, summary: dict[str, Any]) -> dict[str, Any]:
    fake_py = workspace / "fake_llama_bench.py"
    fake_py.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import sys",
                "if '--version' in sys.argv:",
                "    print('fake llama-bench fixture version 1.0')",
                "    raise SystemExit(0)",
                "if '--help' in sys.argv:",
                "    print('fake llama-bench fixture')",
                "    print('  -m, --model FILE')",
                "    print('  -t, --threads N')",
                "    print('  -p, --n-prompt N')",
                "    print('  -n, --n-gen N')",
                "    print('  -r, --repetitions N')",
                "    raise SystemExit(0)",
                "print('warning: fixture output, not a measured Arm run')",
                "print('| model | threads | test | t/s |')",
                "print('| --- | --- | --- | --- |')",
                "print('| fake-q4 | 1 | pp16 | 12.50 +/- 0.1 |')",
                "print('| fake-q4 | 1 | tg8 | 3.25 +/- 0.1 |')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        wrapper = workspace / "fake_llama_bench.cmd"
        wrapper.write_text(f'@echo off\r\n"{python}" "{fake_py}" %*\r\n', encoding="utf-8")
    else:
        wrapper = workspace / "fake_llama_bench"
        wrapper.write_text(f'#!/bin/sh\nexec "{python}" "{fake_py}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    model = workspace / "fixture-model.gguf"
    model.write_bytes(b"fixture model, not real weights\n")
    config = workspace / "arm_fixture.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result_origin": "test_fixture",
                "target": {"type": "local"},
                "executable": {"path": str(wrapper.resolve())},
                "model": {"path": str(model.resolve())},
                "benchmark": {
                    "threads": 1,
                    "prompt_tokens": 16,
                    "generation_tokens": 8,
                    "repetitions": 1,
                    "timeout_seconds": 20,
                },
                "artifacts": {"preserve_raw_stdout": True, "preserve_raw_stderr": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = _run([str(autotuneai), "arm", "benchmark", "--config", str(config)], cwd=workspace, summary=summary, name="arm_fixture_benchmark")
    lines = [line.strip() for line in completed.stdout.splitlines()]
    run_dir_line = next((line for line in lines if line.startswith("artifact_directory:")), "")
    artifact_dir = _workspace_path(workspace, run_dir_line.split(":", 1)[1].strip()) if run_dir_line else None
    if not artifact_dir or not artifact_dir.exists():
        raise SmokeFailure("ArmTune fixture artifact_dir was not reported")
    result_path = artifact_dir / "benchmark_result.json"
    if not result_path.exists():
        raise SmokeFailure("ArmTune fixture benchmark_result.json is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise SmokeFailure("ArmTune fixture benchmark_result.json is not an object")
    return result


def _no_active_leases(cache_root: Path) -> bool:
    leases = list((cache_root / "apply_state" / "leases").glob("*.json"))
    for path in leases:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if payload.get("state") == "active":
            return False
    return True


def _workspace_path(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _no_child_processes() -> bool:
    try:
        import psutil
    except ModuleNotFoundError:
        return True
    current = psutil.Process()
    return not current.children(recursive=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _is_reused_cache_state(value: Any) -> bool:
    return value in {"exact_match", "soft_drift"}


def _tail(text: str, limit: int = 1200) -> str:
    return text[-limit:]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(json.dumps({"schema_version": "1.0", "status": "failed", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc
