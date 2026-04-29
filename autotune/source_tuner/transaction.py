from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import RUNS_DIR, RunManifest, create_run, load_manifest, write_json


class SourceTuningError(RuntimeError):
    pass


def apply_find_replace(
    target_file: str | Path,
    find_text: str,
    replace_text: str,
    *,
    run_id: str | None = None,
    apply: bool = False,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    path = Path(target_file)
    if not path.exists():
        raise SourceTuningError(f"target file does not exist: {path}")
    original = path.read_text(encoding="utf-8")
    count = original.count(find_text)
    if count == 0:
        raise SourceTuningError(f"find text was not found in {path}")
    if count > 1:
        raise SourceTuningError(f"find text appears {count} times in {path}; refusing ambiguous replacement")
    updated = original.replace(find_text, replace_text, 1)
    preview = {
        "target_file": str(path),
        "find_text": find_text,
        "replace_text": replace_text,
        "matches": count,
        "applied": False,
        "run_id": run_id,
    }
    if not apply:
        return preview

    run_dir, manifest = _create_or_load_tuning_run(run_id, runs_dir)
    backup_path = _backup_file(path, run_dir)
    path.write_text(updated, encoding="utf-8")
    changed_file = {
        "path": str(path),
        "backup": str(backup_path),
        "reason": f"replace {find_text!r} with {replace_text!r}",
    }
    manifest.changed_files.append(changed_file)
    manifest.notes.append(f"Applied find/replace to {path}")
    write_json(run_dir / "manifest.json", asdict(manifest))
    preview.update({"applied": True, "run_id": manifest.run_id, "backup": str(backup_path)})
    return preview


def _create_or_load_tuning_run(run_id: str | None, runs_dir: Path) -> tuple[Path, RunManifest]:
    if run_id is None:
        return create_run(["tune_source"], ResourceBudget(), runs_dir)
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        manifest = RunManifest(
            run_id=run_id,
            command=["tune_source"],
            budget=ResourceBudget().to_record(),
            started_at="manual",
        )
        write_json(run_dir / "manifest.json", asdict(manifest))
        return run_dir, manifest
    raw = load_manifest(run_dir)
    manifest = RunManifest(
        run_id=raw["run_id"],
        command=raw.get("command", []),
        budget=raw.get("budget", {}),
        started_at=raw.get("started_at", "unknown"),
        status=raw.get("status", "running"),
        finished_at=raw.get("finished_at"),
        return_code=raw.get("return_code"),
        changed_files=raw.get("changed_files", []),
        notes=raw.get("notes", []),
    )
    return run_dir, manifest


def _backup_file(path: Path, run_dir: Path) -> Path:
    try:
        relative_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative_path = Path(path.name)
    backup_path = run_dir / "backup" / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path
