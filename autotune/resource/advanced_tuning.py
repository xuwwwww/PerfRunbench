from __future__ import annotations

import shutil
from dataclasses import dataclass, field


class AdvancedTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdvancedRunOptions:
    numa_node: int | None = None
    numa_cpu_nodes: str | None = None
    numa_memory_nodes: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def enabled(self) -> bool:
        return bool(
            self.numa_node is not None
            or self.numa_cpu_nodes
            or self.numa_memory_nodes
            or self.extra_env
        )

    def summary(self) -> list[str]:
        items: list[str] = []
        if self.numa_node is not None:
            items.append(f"numa_node={self.numa_node}")
        if self.numa_cpu_nodes:
            items.append(f"numa_cpu_nodes={self.numa_cpu_nodes}")
        if self.numa_memory_nodes:
            items.append(f"numa_memory_nodes={self.numa_memory_nodes}")
        if self.extra_env:
            items.append(
                "extra_env="
                + ",".join(f"{key}={value}" for key, value in sorted(self.extra_env.items()))
            )
        return items


ADVANCED_SYSTEM_PROFILES = {"linux-extreme-throughput"}
ADVANCED_RUNTIME_PROFILES = {"runtime-pytorch-aggressive"}


def wrap_command_with_numa(command: list[str], options: AdvancedRunOptions) -> list[str]:
    validate_advanced_run_options(options)
    if not _needs_numactl(options):
        return command
    numactl = shutil.which("numactl")
    if not numactl:
        raise AdvancedTuningError("advanced NUMA tuning requested, but numactl was not found on PATH.")
    wrapped = [numactl]
    if options.numa_node is not None:
        wrapped.extend(["--cpunodebind", str(options.numa_node), "--membind", str(options.numa_node)])
    else:
        if options.numa_cpu_nodes:
            wrapped.extend(["--cpunodebind", options.numa_cpu_nodes])
        if options.numa_memory_nodes:
            wrapped.extend(["--membind", options.numa_memory_nodes])
    return [*wrapped, *command]


def validate_advanced_confirmation(
    *,
    confirm_advanced_tuning: bool,
    tune_system_profile: str | None = None,
    runtime_env_profile: str | None = None,
    advanced_options: AdvancedRunOptions | None = None,
) -> None:
    advanced_options = advanced_options or AdvancedRunOptions()
    requested = []
    if tune_system_profile in ADVANCED_SYSTEM_PROFILES:
        requested.append(f"system_profile={tune_system_profile}")
    if runtime_env_profile in ADVANCED_RUNTIME_PROFILES:
        requested.append(f"runtime_profile={runtime_env_profile}")
    requested.extend(advanced_options.summary())
    if requested and not confirm_advanced_tuning:
        raise AdvancedTuningError(
            "advanced tuning requires explicit confirmation. "
            "Re-run with --confirm-advanced-tuning to allow: "
            + ", ".join(requested)
        )


def validate_advanced_run_options(options: AdvancedRunOptions) -> None:
    if options.numa_node is not None and options.numa_node < 0:
        raise AdvancedTuningError("numa_node must be >= 0")
    for key, value in options.extra_env.items():
        if not key or "=" in key:
            raise AdvancedTuningError(f"invalid extra env key: {key!r}")
        if value is None:
            raise AdvancedTuningError(f"invalid extra env value for {key!r}")


def parse_extra_env(entries: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise AdvancedTuningError(f"invalid --extra-env value {entry!r}; expected KEY=VALUE")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise AdvancedTuningError(f"invalid --extra-env value {entry!r}; key cannot be empty")
        result[key] = value
    return result


def _needs_numactl(options: AdvancedRunOptions) -> bool:
    return bool(
        options.numa_node is not None
        or options.numa_cpu_nodes
        or options.numa_memory_nodes
    )
