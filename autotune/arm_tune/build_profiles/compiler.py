from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.build_profiles.validation import validate_compiler_argument_list
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import to_plain


@dataclass(frozen=True)
class CompilerProbeResult:
    probe_name: str
    supported: bool
    compiler: str
    arguments: list[str]
    return_code: int | None
    produced_binary_ran: bool = False
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class EffectiveProfileSignature:
    profile_id: str
    cmake_definitions: tuple[tuple[str, str], ...]
    compile_flags: tuple[str, ...] = field(default_factory=tuple)
    linker_flags: tuple[str, ...] = field(default_factory=tuple)
    backend_policy: str = "default_cpu"
    cpu_target_policy: str = "upstream_default"
    optimization_policy: str = "standard_release"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def probe_compiler_target(
    *,
    compiler: str,
    arguments: list[str],
    work_dir: str | Path,
    probe_name: str = "explicit_cpu_target",
    timeout_seconds: int = 20,
) -> CompilerProbeResult:
    args = validate_compiler_argument_list(arguments, label="compiler probe arguments")
    if not compiler or any(token in compiler for token in ("&&", "||", ";", "`", "$(", "|", ">", "<")):
        raise ArmTuneConfigError("compiler must be an executable name or path, not a shell fragment")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"{probe_name}.c"
    binary = root / f"{probe_name}.out"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8", newline="\n")
    command = [compiler, *args, str(source), "-o", str(binary)]
    try:
        build = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CompilerProbeResult(
            probe_name=probe_name,
            supported=False,
            compiler=compiler,
            arguments=args,
            return_code=None,
            reason=str(exc),
        )
    if build.returncode != 0 or not binary.exists():
        return CompilerProbeResult(
            probe_name=probe_name,
            supported=False,
            compiler=compiler,
            arguments=args,
            return_code=build.returncode,
            stdout_excerpt=_excerpt(build.stdout),
            stderr_excerpt=_excerpt(build.stderr),
            reason="compile test failed",
        )
    try:
        run = subprocess.run(
            [str(binary)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CompilerProbeResult(
            probe_name=probe_name,
            supported=False,
            compiler=compiler,
            arguments=args,
            return_code=build.returncode,
            stdout_excerpt=_excerpt(build.stdout),
            stderr_excerpt=_excerpt(build.stderr),
            reason=f"produced binary did not run: {exc}",
        )
    return CompilerProbeResult(
        probe_name=probe_name,
        supported=run.returncode == 0,
        compiler=compiler,
        arguments=args,
        return_code=build.returncode,
        produced_binary_ran=run.returncode == 0,
        stdout_excerpt=_excerpt(build.stdout),
        stderr_excerpt=_excerpt(build.stderr),
        reason="" if run.returncode == 0 else f"produced binary exited with {run.returncode}",
    )


def effective_signature(
    *,
    profile_id: str,
    cmake_definitions: dict[str, str],
    compile_flags: list[str],
    linker_flags: list[str] | None = None,
    backend_policy: str = "default_cpu",
    cpu_target_policy: str = "upstream_default",
    optimization_policy: str = "standard_release",
) -> EffectiveProfileSignature:
    return EffectiveProfileSignature(
        profile_id=profile_id,
        cmake_definitions=tuple(sorted((str(key), str(value)) for key, value in cmake_definitions.items())),
        compile_flags=tuple(str(item) for item in compile_flags),
        linker_flags=tuple(str(item) for item in (linker_flags or [])),
        backend_policy=backend_policy,
        cpu_target_policy=cpu_target_policy,
        optimization_policy=optimization_policy,
    )


def duplicate_effective_profiles(signatures: list[EffectiveProfileSignature]) -> dict[str, str]:
    seen: dict[tuple[Any, ...], str] = {}
    duplicates: dict[str, str] = {}
    for signature in signatures:
        key = (
            signature.cmake_definitions,
            signature.compile_flags,
            signature.linker_flags,
            signature.backend_policy,
            signature.cpu_target_policy,
            signature.optimization_policy,
        )
        if key in seen:
            duplicates[signature.profile_id] = seen[key]
        else:
            seen[key] = signature.profile_id
    return duplicates


def _excerpt(value: str | None, limit: int = 2000) -> str:
    text = value or ""
    return text[:limit]
