from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from autotune.resource.budget import ResourceBudget


RUNS_DIR = Path(".autotuneai") / "runs"


@dataclass
class RunManifest:
    run_id: str
    command: list[str]
    budget: dict[str, Any]
    started_at: str
    status: str = "running"
    finished_at: str | None = None
    return_code: int | None = None
    changed_files: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def create_run(command: list[str], budget: ResourceBudget, runs_dir: Path = RUNS_DIR) -> tuple[Path, RunManifest]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=run_dir.name,
        command=command,
        budget=budget.to_record(),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    write_json(run_dir / "manifest.json", asdict(manifest))
    write_git_snapshot(run_dir)
    write_json(run_dir / "env.json", collect_environment())
    return run_dir, manifest


def finish_run(run_dir: Path, manifest: RunManifest, status: str, return_code: int | None) -> None:
    manifest.status = status
    manifest.return_code = return_code
    manifest.finished_at = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", asdict(manifest))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def write_git_snapshot(run_dir: Path) -> None:
    status = _git_output(["git", "status", "--short"])
    diff = _git_output(["git", "diff"])
    head = _git_output(["git", "rev-parse", "HEAD"])
    (run_dir / "before_status.txt").write_text(status, encoding="utf-8")
    (run_dir / "before_diff.patch").write_text(diff, encoding="utf-8")
    (run_dir / "head.txt").write_text(head, encoding="utf-8")


def collect_environment() -> dict[str, Any]:
    return {
        "python": _command_output(["python", "--version"]),
        "platform": _command_output(["python", "-c", "import platform; print(platform.platform())"]),
        "cwd": str(Path.cwd()),
    }


def list_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    runs = []
    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = load_manifest(run_dir)
            manifest["path"] = str(run_dir)
            runs.append(manifest)
    return runs


def _git_output(command: list[str]) -> str:
    return _command_output(command, check=False)


def _command_output(command: list[str], check: bool = False) -> str:
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    output = result.stdout
    if result.stderr:
        output += result.stderr
    return output

