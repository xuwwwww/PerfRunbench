from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import create_run, finish_run, load_manifest, manifest_from_dict, write_json
from autotune.resource.workload_runner import run_with_budget
from autotune.source_tuner.transaction import apply_find_replace, restore_changed_files


@dataclass(frozen=True)
class SourceEdit:
    file: str
    find_text: str
    replace_text: str


def run_tuned_with_budget(
    command: list[str],
    edits: list[SourceEdit],
    budget: ResourceBudget,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    auto_restore: bool = True,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
) -> tuple[int, Path]:
    if not edits:
        raise ValueError("at least one source edit is required")
    run_dir, manifest = create_run(command, budget)
    restored: list[str] = []
    return_code = 1
    completed_by_workload_runner = False
    try:
        for edit in edits:
            apply_find_replace(
                edit.file,
                edit.find_text,
                edit.replace_text,
                run_id=manifest.run_id,
                apply=True,
            )
        manifest = manifest_from_dict(load_manifest(run_dir))
        return_code, run_dir = run_with_budget(
            command,
            budget,
            sample_interval_seconds=sample_interval_seconds,
            hard_kill=hard_kill,
            run_dir=run_dir,
            manifest=manifest,
            executor=executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
        )
        completed_by_workload_runner = True
        return return_code, run_dir
    finally:
        if auto_restore:
            restored = restore_changed_files(run_dir)
            manifest = manifest_from_dict(load_manifest(run_dir))
            manifest.notes.append(f"auto_restored_files={restored}")
            write_json(run_dir / "manifest.json", asdict(manifest))
        if not completed_by_workload_runner:
            manifest = manifest_from_dict(load_manifest(run_dir))
            finish_run(run_dir, manifest, "failed", return_code)
