from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.llama_bench import OPTIONAL_FLAGS
from autotune.arm_tune.schemas import CpuTopology, LlamaBenchJobSpec, LlamaBenchSettings, to_plain


AFFINITY_MODES = {"scheduler-default", "compact", "spread", "reserve-one-core", "numa-local"}


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    stage: str
    threads: int
    additional_supported_options: dict[str, int] = field(default_factory=dict)
    affinity_mode: str = "scheduler-default"
    cpu_affinity: list[int] | None = None
    allowed_cpu_mask: str = "scheduler-default"
    is_baseline: bool = False
    description: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    def settings_for(self, base: LlamaBenchSettings) -> LlamaBenchSettings:
        return replace(
            base,
            threads=self.threads,
            additional_supported_options=dict(self.additional_supported_options),
            cpu_affinity=list(self.cpu_affinity) if self.cpu_affinity is not None else None,
            affinity_mode=self.affinity_mode,
        )


def baseline_candidate(job: LlamaBenchJobSpec) -> SearchCandidate:
    return SearchCandidate(
        candidate_id="baseline",
        stage="baseline",
        threads=job.benchmark.threads,
        additional_supported_options=_int_options(job.benchmark.additional_supported_options),
        affinity_mode=job.benchmark.affinity_mode,
        cpu_affinity=list(job.benchmark.cpu_affinity) if job.benchmark.cpu_affinity else None,
        allowed_cpu_mask=_mask(job.benchmark.cpu_affinity),
        is_baseline=True,
        description="configured baseline",
    )


def visible_cpu_indexes(topology: CpuTopology) -> list[int]:
    parsed = parse_cpu_list(topology.online_cpus)
    if parsed:
        return parsed
    if topology.logical_cpus and topology.logical_cpus > 0:
        return list(range(topology.logical_cpus))
    return [0]


def parse_cpu_list(text: str) -> list[int]:
    if not text or text == "unknown":
        return []
    cpus: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                return []
            if start > end or start < 0:
                return []
            cpus.update(range(start, end + 1))
            continue
        try:
            cpu = int(token)
        except ValueError:
            return []
        if cpu < 0:
            return []
        cpus.add(cpu)
    return sorted(cpus)


def generate_thread_counts(topology: CpuTopology, *, baseline_threads: int) -> list[int]:
    visible = len(visible_cpu_indexes(topology))
    candidates = {1, baseline_threads, visible}
    if visible > 1:
        candidates.add(2)
        candidates.add(max(1, visible // 2))
        candidates.add(visible - 1)
    power = 1
    while power <= visible:
        candidates.add(power)
        power *= 2
    if topology.cores_per_socket and topology.cores_per_socket > 0:
        candidates.add(topology.cores_per_socket)
    if topology.cores_per_socket and topology.sockets and topology.cores_per_socket > 0 and topology.sockets > 0:
        candidates.add(topology.cores_per_socket * topology.sockets)
    return sorted(value for value in candidates if 1 <= value <= visible)


def generate_thread_candidates(job: LlamaBenchJobSpec, topology: CpuTopology) -> list[SearchCandidate]:
    baseline = baseline_candidate(job)
    candidates = [baseline]
    for threads in generate_thread_counts(topology, baseline_threads=job.benchmark.threads):
        if threads == baseline.threads:
            continue
        candidates.append(
            SearchCandidate(
                candidate_id=f"threads-{threads}",
                stage="thread_count",
                threads=threads,
                additional_supported_options=dict(baseline.additional_supported_options),
                description=f"{threads} llama-bench threads",
            )
        )
    return candidates


def supported_option_names(supported_flags: list[str]) -> set[str]:
    supported = set(supported_flags)
    names: set[str] = set()
    for name, flags in OPTIONAL_FLAGS.items():
        if any(flag in supported for flag in flags):
            names.add(name)
    return names


def generate_batch_candidates(
    base: SearchCandidate,
    job: LlamaBenchJobSpec,
    *,
    supported_flags: list[str],
) -> list[SearchCandidate]:
    supported = supported_option_names(supported_flags)
    options = dict(base.additional_supported_options)
    changed = False
    if "threads_batch" in supported:
        value = max(1, base.threads)
        changed = changed or options.get("threads_batch") != value
        options["threads_batch"] = value
    if "batch_size" in supported:
        value = max(job.benchmark.prompt_tokens, int(options.get("batch_size", job.benchmark.prompt_tokens)))
        changed = changed or options.get("batch_size") != value
        options["batch_size"] = value
    if "ubatch_size" in supported:
        batch = int(options.get("batch_size", job.benchmark.prompt_tokens))
        value = min(batch, max(1, int(options.get("ubatch_size", batch))))
        changed = changed or options.get("ubatch_size") != value
        options["ubatch_size"] = value
    if not changed:
        return []
    label = "-".join(f"{key}{value}" for key, value in sorted(options.items()))
    return [
        SearchCandidate(
            candidate_id=f"batch-{label}",
            stage="batch_parameters",
            threads=base.threads,
            additional_supported_options=options,
            affinity_mode=base.affinity_mode,
            cpu_affinity=list(base.cpu_affinity) if base.cpu_affinity else None,
            allowed_cpu_mask=base.allowed_cpu_mask,
            description="supported llama-bench batch parameter candidate",
        )
    ]


def generate_affinity_candidates(base: SearchCandidate, topology: CpuTopology) -> list[SearchCandidate]:
    cpus = visible_cpu_indexes(topology)
    threads = min(base.threads, len(cpus))
    if threads <= 0:
        return []
    candidates: list[SearchCandidate] = []
    seen: set[str] = set()
    _add_affinity(candidates, seen, base, "compact", cpus[:threads])
    _add_affinity(candidates, seen, base, "spread", _spread(cpus, threads))
    if len(cpus) > 1:
        reserve = cpus[:-1]
        _add_affinity(candidates, seen, base, "reserve-one-core", reserve[: min(threads, len(reserve))])
    if topology.numa_nodes and topology.numa_nodes > 1:
        _add_affinity(candidates, seen, base, "numa-local", cpus[:threads])
    return candidates


def _add_affinity(
    candidates: list[SearchCandidate],
    seen: set[str],
    base: SearchCandidate,
    mode: str,
    cpus: list[int],
) -> None:
    if mode not in AFFINITY_MODES or not cpus:
        return
    mask = _mask(cpus)
    identity = f"{mode}:{mask}"
    if identity in seen:
        return
    seen.add(identity)
    candidates.append(
        SearchCandidate(
            candidate_id=f"affinity-{mode}-{mask.replace(',', '_')}",
            stage="affinity",
            threads=base.threads,
            additional_supported_options=dict(base.additional_supported_options),
            affinity_mode=mode,
            cpu_affinity=cpus,
            allowed_cpu_mask=mask,
            description=f"{mode} process CPU affinity",
        )
    )


def _spread(cpus: list[int], count: int) -> list[int]:
    if count >= len(cpus):
        return list(cpus)
    if count == 1:
        return [cpus[0]]
    selected: list[int] = []
    last_index = len(cpus) - 1
    for index in range(count):
        position = round(index * last_index / (count - 1))
        selected.append(cpus[position])
    return sorted(set(selected))


def _mask(cpus: list[int] | None) -> str:
    if not cpus:
        return "scheduler-default"
    return ",".join(str(cpu) for cpu in cpus)


def _int_options(options: dict[str, Any]) -> dict[str, int]:
    return {key: int(value) for key, value in options.items()}
