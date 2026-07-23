from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import stable_digest
from autotune.system_profiles.protocols import PROTOCOLS
from autotune.system_profiles.schemas import (
    HostCapabilityReport,
    HostStateSnapshot,
    HostSystemProfile,
    NOT_AVAILABLE,
    ProfileChange,
    SystemProfileError,
    SystemTuningPlan,
)
from autotune.system_profiles.snapshot import RootReader


PROFILE_STATES = {
    "supported",
    "unsupported",
    "partially_supported",
    "requires_root",
    "requires_reboot",
    "requires_user_snapshot",
    "unsafe",
    "duplicate_effective_profile",
}

PROHIBITED_PROFILE_IDS = {
    "disable-mitigations": "disabling CPU vulnerability mitigations is prohibited",
    "disable-security-modules": "disabling SELinux/AppArmor/security modules is prohibited",
    "panic-on-oom": "changing panic_on_oom is prohibited",
    "unsafe-overcommit": "unsafe overcommit settings are prohibited",
    "disable-watchdogs": "watchdog changes require a verified recovery reason and are prohibited here",
    "ssh-firewall-change": "SSH and firewall configuration changes are prohibited",
    "sched-fifo": "SCHED_FIFO and SCHED_RR are prohibited",
    "no-housekeeping": "profiles must preserve housekeeping CPUs",
    "offline-housekeeping": "offlining housekeeping CPUs is prohibited",
    "remove-kernel": "removing running or fallback kernels is prohibited",
    "aws-metadata-change": "AWS metadata or credential changes are prohibited",
    "hide-restore-failure": "failed restore operations must never be hidden",
}

SAFE_PROFILE_IDS = [
    "baseline-current",
    "cpufreq-performance",
    "affinity-only",
    "cpuset-partition-1h7w",
    "cpuset-partition-2h6w",
    "irq-workqueue-isolation",
    "irqbalance-controlled",
    "scheduler-batch",
    "process-priority",
    "thp-always",
    "thp-madvise",
    "thp-never",
    "thp-defrag-always",
    "thp-defrag-defer",
    "thp-defrag-defer-madvise",
    "thp-defrag-madvise",
    "thp-defrag-never",
    "compaction-conservative",
    "mmap-default",
    "mmap-mlock",
    "no-mmap",
    "no-mmap-mlock",
    "prefault-warm",
    "allocator-glibc",
    "allocator-glibc-bounded-arenas",
    "allocator-jemalloc",
    "allocator-mimalloc",
    "storage-read-ahead",
    "io-scheduler",
    "numa-policy",
    "vm-system-conservative",
    "hugetlb-bounded",
]


def collect_capabilities(snapshot: HostStateSnapshot, root: str | Path | None = None) -> HostCapabilityReport:
    reader = RootReader(root)
    caps = _capability_map(snapshot, reader)
    states = {profile_id: build_profile(profile_id, snapshot, caps, reader).eligibility_state for profile_id in SAFE_PROFILE_IDS}
    for profile_id in PROHIBITED_PROFILE_IDS:
        states[profile_id] = "unsafe"
    return HostCapabilityReport(snapshot_digest=snapshot.snapshot_digest, capabilities=caps, profile_states=states)


def build_profile(
    profile_id: str,
    snapshot: HostStateSnapshot,
    capabilities: dict[str, Any] | None = None,
    reader: RootReader | None = None,
) -> HostSystemProfile:
    if profile_id in PROHIBITED_PROFILE_IDS:
        return _profile(
            profile_id=profile_id,
            mechanism="prohibited unsafe host modification",
            scope="host",
            state="unsafe",
            reason=PROHIBITED_PROFILE_IDS[profile_id],
            protocols=[],
            changes=[],
            prohibited=[PROHIBITED_PROFILE_IDS[profile_id]],
        )
    if profile_id not in SAFE_PROFILE_IDS:
        raise SystemProfileError(f"unknown system profile: {profile_id}")
    caps = capabilities or _capability_map(snapshot, reader or RootReader(None))
    rdr = reader or RootReader(None)
    if profile_id == "baseline-current":
        return _profile(profile_id=profile_id, mechanism="capture current host state without changes", scope="read_only", state="supported", protocols=list(PROTOCOLS), changes=[])
    if profile_id == "cpufreq-performance":
        changes = _cpufreq_changes(snapshot)
        if not caps["cpufreq_available"]:
            return _unsupported(profile_id, "CPUFreq policies are not exposed", "cpu frequency governor")
        if not changes:
            return _unsupported(profile_id, "performance governor is unavailable or governor file is not writable", "cpu frequency governor")
        return _profile(
            profile_id=profile_id,
            mechanism="force exposed CPUFreq policies to the performance governor",
            scope="host_runtime",
            state="requires_root",
            protocols=["steady_state_decode", "interactive_server", "concurrent_server"],
            changes=changes,
            tradeoffs=["higher power use", "may be unavailable on virtualized hosts"],
        )
    if profile_id == "affinity-only":
        return _profile(
            profile_id=profile_id,
            mechanism="process-scoped CPU placement without host writes",
            scope="process",
            state="supported" if caps["valid_affinity_masks"] else "unsupported",
            protocols=["steady_state_decode", "interactive_server", "concurrent_server"],
            changes=[
                ProfileChange("allowed_cpu_masks", "process", "sched_setaffinity", str(caps["valid_affinity_masks"]), str(caps["valid_affinity_masks"]), requires_root=False)
            ],
            tradeoffs=["can reduce available parallelism", "must not starve housekeeping work"],
        )
    if profile_id in {"cpuset-partition-1h7w", "cpuset-partition-2h6w"}:
        housekeeping = "0" if profile_id.endswith("1h7w") else "0,1"
        workload = "1-7" if profile_id.endswith("1h7w") else "2-7"
        if caps["logical_cpus"] != 8:
            return _unsupported(profile_id, "conceptual split requires exactly 8 logical CPUs", "cgroup v2 cpuset partition")
        if not caps["cgroup_v2_cpuset"]:
            return _unsupported(profile_id, "cgroup v2 cpuset controller is not available", "cgroup v2 cpuset partition")
        changes = [
            ProfileChange("housekeeping_cpuset", "cgroup_v2", "/sys/fs/cgroup/perfrunbench-housekeeping/cpuset.cpus", NOT_AVAILABLE, housekeeping, requires_root=True),
            ProfileChange("workload_cpuset", "cgroup_v2", "/sys/fs/cgroup/perfrunbench-workload/cpuset.cpus", NOT_AVAILABLE, workload, requires_root=True),
            ProfileChange("workload_partition", "cgroup_v2", "/sys/fs/cgroup/perfrunbench-workload/cpuset.cpus.partition", NOT_AVAILABLE, "isolated", requires_root=True),
        ]
        return _profile(profile_id=profile_id, mechanism="cgroup v2 housekeeping/workload partition isolation", scope="host_runtime", state="requires_root", protocols=["steady_state_decode", "interactive_server", "concurrent_server"], changes=changes)
    if profile_id == "irq-workqueue-isolation":
        if not caps["irq_affinity_available"] and not caps["workqueue_cpumask_available"]:
            return _unsupported(profile_id, "IRQ affinity and workqueue cpumask interfaces are not available", "interrupt and workqueue isolation")
        changes = []
        if caps["irq_affinity_available"]:
            changes.append(ProfileChange("irq_affinity_housekeeping", "procfs", "/proc/irq/*/smp_affinity_list", NOT_AVAILABLE, "housekeeping_cpus", requires_root=True))
        if caps["workqueue_cpumask_available"]:
            changes.append(ProfileChange("unbound_workqueue_housekeeping", "sysfs", "/sys/devices/virtual/workqueue/cpumask", snapshot.interrupts_workqueues.get("unbound_workqueue_cpumask", NOT_AVAILABLE), "housekeeping_cpus", requires_root=True))
        return _profile(profile_id=profile_id, mechanism="move writable IRQs and unbound workqueues away from workload CPUs", scope="host_runtime", state="partially_supported", protocols=["steady_state_decode", "interactive_server", "concurrent_server"], changes=changes)
    if profile_id == "irqbalance-controlled":
        state = "partially_supported" if caps["irqbalance_observable"] else "unsupported"
        return _profile(profile_id=profile_id, mechanism="preserve, stop, or constrain irqbalance when supported", scope="host_runtime", state=state, protocols=["steady_state_decode", "interactive_server"], changes=[], reason="" if state != "unsupported" else "irqbalance state is not observable")
    if profile_id == "scheduler-batch":
        return _profile(profile_id=profile_id, mechanism="process-scoped SCHED_BATCH policy", scope="process", state="supported", protocols=["steady_state_decode"], changes=[ProfileChange("scheduler_policy", "process", "sched_setscheduler", "current", "SCHED_BATCH")])
    if profile_id == "process-priority":
        return _profile(profile_id=profile_id, mechanism="bounded process nice adjustment", scope="process", state="requires_root", protocols=["steady_state_decode", "interactive_server"], changes=[ProfileChange("nice_level", "process", "setpriority", "current", "-5", requires_root=True)], tradeoffs=["may reduce service responsiveness if misused"])
    if profile_id in {"thp-always", "thp-madvise", "thp-never"}:
        mode = profile_id.removeprefix("thp-")
        modes = caps["thp_modes"]
        if mode not in modes:
            return _unsupported(profile_id, f"THP mode {mode!r} is not reported by the kernel", "transparent huge pages")
        before = snapshot.memory.get("thp_top_level_mode", NOT_AVAILABLE)
        return _profile(profile_id=profile_id, mechanism="Transparent Huge Page mode selection with workload exercise evidence", scope="host_runtime", state="requires_root", protocols=["steady_state_decode", "warm_first_request"], changes=[ProfileChange("thp_enabled", "sysfs", "/sys/kernel/mm/transparent_hugepage/enabled", before, mode, requires_root=True)], tradeoffs=["THP may increase compaction latency", "a sysfs write alone is not exercised evidence"])
    if profile_id.startswith("thp-defrag-"):
        mode = profile_id.removeprefix("thp-defrag-").replace("-", "+")
        if mode not in caps["thp_defrag_modes"]:
            return _unsupported(profile_id, f"THP defrag mode {mode!r} is not supported", "transparent huge page defrag")
        return _profile(profile_id=profile_id, mechanism="THP defrag policy after a THP mode is selected", scope="host_runtime", state="requires_root", protocols=["steady_state_decode", "warm_first_request"], changes=[ProfileChange("thp_defrag", "sysfs", "/sys/kernel/mm/transparent_hugepage/defrag", snapshot.memory.get("thp_defrag_mode", NOT_AVAILABLE), mode, requires_root=True)])
    if profile_id == "compaction-conservative":
        current = snapshot.memory.get("compaction_settings", {}).get("vm.compaction_proactiveness", NOT_AVAILABLE) if isinstance(snapshot.memory.get("compaction_settings"), dict) else NOT_AVAILABLE
        if current == NOT_AVAILABLE:
            return _unsupported(profile_id, "vm.compaction_proactiveness is not exposed", "memory compaction")
        return _profile(profile_id=profile_id, mechanism="bounded vm.compaction_proactiveness testing", scope="host_runtime", state="requires_root", protocols=["steady_state_decode", "warm_first_request"], changes=[ProfileChange("vm.compaction_proactiveness", "procfs", "/proc/sys/vm/compaction_proactiveness", current, "20", requires_root=True)], tradeoffs=["compaction may add latency"])
    if profile_id in {"mmap-default", "mmap-mlock", "no-mmap", "no-mmap-mlock"}:
        return _profile(profile_id=profile_id, mechanism="llama.cpp model mapping option if supported by the actual binary", scope="workload_launch", state="partially_supported", protocols=["cold_model_load", "warm_first_request", "steady_state_decode"], changes=[ProfileChange("llama_mmap_mode", "process", "argv", "default", profile_id)])
    if profile_id == "prefault-warm":
        return _profile(profile_id=profile_id, mechanism="bounded prefault and warm page-cache protocol", scope="workload_protocol", state="supported", protocols=["warm_first_request", "cold_model_load"], changes=[], tradeoffs=["not a decode acceleration claim"])
    if profile_id in {"allocator-glibc", "allocator-glibc-bounded-arenas"}:
        change = ProfileChange("MALLOC_ARENA_MAX", "process_env", "env:MALLOC_ARENA_MAX", NOT_AVAILABLE, "2") if profile_id.endswith("bounded-arenas") else ProfileChange("allocator", "process_env", "runtime_allocator", "glibc", "glibc")
        return _profile(profile_id=profile_id, mechanism="glibc allocator selection with bounded arenas when requested", scope="process", state="supported", protocols=["steady_state_decode", "interactive_server"], changes=[change])
    if profile_id in {"allocator-jemalloc", "allocator-mimalloc"}:
        lib = "jemalloc" if profile_id.endswith("jemalloc") else "mimalloc"
        installed = _allocator_available(rdr, lib)
        return _profile(profile_id=profile_id, mechanism=f"{lib} LD_PRELOAD allocator evidence", scope="process", state="partially_supported" if installed else "unsupported", protocols=["steady_state_decode", "interactive_server"], changes=[ProfileChange("LD_PRELOAD", "process_env", "env:LD_PRELOAD", NOT_AVAILABLE, lib)], reason="" if installed else f"{lib} library not installed or not fixture-visible")
    if profile_id == "storage-read-ahead":
        if snapshot.storage.get("block_read_ahead") == NOT_AVAILABLE:
            return _unsupported(profile_id, "block read-ahead interface is not available", "storage read-ahead")
        return _profile(profile_id=profile_id, mechanism="bounded block read-ahead for cold model load", scope="host_runtime", state="requires_root", protocols=["cold_model_load"], changes=[ProfileChange("read_ahead_kb", "sysfs", "/sys/block/*/queue/read_ahead_kb", "current", "bounded_derived_value", requires_root=True)])
    if profile_id == "io-scheduler":
        if snapshot.storage.get("io_schedulers") == NOT_AVAILABLE:
            return _unsupported(profile_id, "device scheduler list is not available", "I/O scheduler")
        return _profile(profile_id=profile_id, mechanism="allowlisted device I/O scheduler for cold load/startup", scope="host_runtime", state="requires_root", protocols=["cold_model_load"], changes=[ProfileChange("io_scheduler", "sysfs", "/sys/block/*/queue/scheduler", "current", "supported_scheduler", requires_root=True)])
    if profile_id == "numa-policy":
        if not caps["multi_numa"]:
            return _unsupported(profile_id, "single effective NUMA node; NUMA policy is not applicable", "NUMA placement")
        return _profile(profile_id=profile_id, mechanism="explicit NUMA placement on multi-node targets", scope="process", state="supported", protocols=["steady_state_decode", "interactive_server"], changes=[ProfileChange("numa_policy", "process", "numactl", "current", "local_workload_node")])
    if profile_id == "vm-system-conservative":
        changes = []
        sysctls = snapshot.memory.get("relevant_vm_sysctls", {})
        if isinstance(sysctls, dict) and sysctls.get("vm.stat_interval", NOT_AVAILABLE) != NOT_AVAILABLE:
            changes.append(ProfileChange("vm.stat_interval", "procfs", "/proc/sys/vm/stat_interval", sysctls.get("vm.stat_interval", NOT_AVAILABLE), "10", requires_root=True))
        sched = snapshot.scheduler_cgroup.get("exposed_scheduler_sysctls", {})
        if isinstance(sched, dict) and sched.get("kernel.sched_autogroup_enabled", NOT_AVAILABLE) != NOT_AVAILABLE:
            changes.append(ProfileChange("kernel.sched_autogroup_enabled", "procfs", "/proc/sys/kernel/sched_autogroup_enabled", sched.get("kernel.sched_autogroup_enabled", NOT_AVAILABLE), "0", requires_root=True))
        return _profile(profile_id=profile_id, mechanism="small mechanism-based sysctl allowlist only", scope="host_runtime", state="requires_root" if changes else "unsupported", protocols=["steady_state_decode"], changes=changes, reason="" if changes else "no relevant allowlisted sysctls are exposed")
    if profile_id == "hugetlb-bounded":
        if snapshot.memory.get("hugetlb_page_sizes") == NOT_AVAILABLE:
            return _unsupported(profile_id, "HugeTLB pools are not exposed", "HugeTLB reservation and verified mapping")
        return _profile(profile_id=profile_id, mechanism="bounded HugeTLB pool reservation with process mapping proof", scope="host_runtime", state="partially_supported", protocols=["cold_model_load", "warm_first_request"], changes=[ProfileChange("hugetlb_pool", "sysfs", "/sys/kernel/mm/hugepages/*/nr_hugepages", "current", "bounded_safe_reservation", requires_root=True)], reason="supported_but_not_exercised until workload mappings confirm HugeTLB use")
    raise SystemProfileError(f"profile not implemented: {profile_id}")


def create_plan(profile_id: str, snapshot: HostStateSnapshot, root: str | Path | None = None, *, catalog: str = "safe-system") -> SystemTuningPlan:
    if catalog != "safe-system":
        raise SystemProfileError(f"unsupported catalog: {catalog}")
    reader = RootReader(root)
    capabilities = _capability_map(snapshot, reader)
    profile = build_profile(profile_id, snapshot, capabilities, reader)
    plan_id = stable_digest({"profile_id": profile_id, "snapshot_digest": snapshot.snapshot_digest, "changes": [change.to_dict() for change in profile.requested_changes]})[:24]
    eligibility = "eligible_for_planning" if profile.eligibility_state in {"supported", "partially_supported", "requires_root"} else "not_eligible"
    return SystemTuningPlan(
        plan_id=plan_id,
        profile=profile,
        snapshot_digest=snapshot.snapshot_digest,
        profile_state=profile.eligibility_state,
        changes=profile.requested_changes,
        eligibility=eligibility,
        protocol_ids=profile.workload_protocols,
    )


def list_profiles(snapshot: HostStateSnapshot, root: str | Path | None = None) -> list[dict[str, Any]]:
    report = collect_capabilities(snapshot, root)
    reader = RootReader(root)
    return [build_profile(profile_id, snapshot, report.capabilities, reader).to_dict() for profile_id in SAFE_PROFILE_IDS]


def staged_search_policy(snapshot: HostStateSnapshot, root: str | Path | None = None) -> dict[str, Any]:
    report = collect_capabilities(snapshot, root)
    return {
        "schema_version": "1.0",
        "cartesian_product": False,
        "stages": [
            {"stage": 0, "name": "baseline and capability discovery", "profiles": ["baseline-current"]},
            {"stage": 1, "name": "CPU frequency and process placement", "profiles": ["cpufreq-performance", "affinity-only", "scheduler-batch"]},
            {"stage": 2, "name": "cpuset, IRQ, workqueue isolation", "profiles": ["cpuset-partition-1h7w", "cpuset-partition-2h6w", "irq-workqueue-isolation", "irqbalance-controlled"]},
            {"stage": 3, "name": "THP, memory mapping, prefault", "profiles": ["thp-always", "thp-madvise", "thp-never", "prefault-warm", "mmap-default", "mmap-mlock", "no-mmap", "no-mmap-mlock"]},
            {"stage": 4, "name": "allocator", "profiles": ["allocator-glibc", "allocator-glibc-bounded-arenas", "allocator-jemalloc", "allocator-mimalloc"]},
            {"stage": 5, "name": "storage startup profiles", "profiles": ["storage-read-ahead", "io-scheduler"]},
            {"stage": 6, "name": "targeted VM/scheduler profiles", "profiles": ["compaction-conservative", "vm-system-conservative", "numa-policy", "hugetlb-bounded"]},
            {"stage": 7, "name": "reboot-required kernel profiles", "profiles": [], "phase": "future_real_validation"},
        ],
        "candidate_caps": {"steady_state_decode": 12, "cold_model_load": 8, "warm_first_request": 8, "interactive_server": 8, "concurrent_server": 6},
        "profile_states": report.profile_states,
        "carry_forward_rule": "only one eligible finalist per mechanism family after confirmation outside the noise band",
    }


def _capability_map(snapshot: HostStateSnapshot, reader: RootReader) -> dict[str, Any]:
    logical = snapshot.identity.get("logical_cpus", NOT_AVAILABLE)
    try:
        logical_count = int(logical)
    except (TypeError, ValueError):
        logical_count = 0
    thp_text = reader.read_text("/sys/kernel/mm/transparent_hugepage/enabled")
    return {
        "cpufreq_available": bool(snapshot.cpu_frequency.get("policies")),
        "cpufreq_performance_governor": bool(_cpufreq_changes(snapshot)),
        "cpuidle_available": bool(snapshot.cpu_idle.get("states")),
        "thp_modes": _modes_from_text(thp_text) or _modes_from_snapshot(snapshot.memory.get("thp_top_level_mode")),
        "thp_defrag_modes": snapshot.memory.get("thp_available_defrag_modes") or [],
        "hugetlb_available": snapshot.memory.get("hugetlb_page_sizes") != NOT_AVAILABLE,
        "multi_numa": _effective_numa_nodes(snapshot) > 1,
        "cgroup_v2_cpuset": snapshot.scheduler_cgroup.get("cgroup_version") == "v2" and bool(snapshot.scheduler_cgroup.get("cpuset_controller_available")),
        "irq_affinity_available": snapshot.interrupts_workqueues.get("irq_affinity_masks") != NOT_AVAILABLE,
        "workqueue_cpumask_available": snapshot.interrupts_workqueues.get("unbound_workqueue_cpumask") != NOT_AVAILABLE,
        "irqbalance_observable": snapshot.interrupts_workqueues.get("irqbalance_service_state") != NOT_AVAILABLE,
        "io_scheduler_available": snapshot.storage.get("io_schedulers") != NOT_AVAILABLE,
        "read_ahead_available": snapshot.storage.get("block_read_ahead") != NOT_AVAILABLE,
        "rps_xps_available": snapshot.network.get("rps_masks") != NOT_AVAILABLE or snapshot.network.get("xps_masks") != NOT_AVAILABLE,
        "grub_available": snapshot.security_recovery.get("grub_configuration_location") != NOT_AVAILABLE,
        "logical_cpus": logical_count,
        "valid_affinity_masks": _valid_affinity_masks(snapshot),
    }


def _cpufreq_changes(snapshot: HostStateSnapshot) -> list[ProfileChange]:
    changes = []
    for policy in snapshot.cpu_frequency.get("policies", []):
        if "performance" not in policy.get("available_governors", []):
            continue
        if not policy.get("governor_writable"):
            continue
        changes.append(
            ProfileChange(
                label=f"{policy['policy']}_governor",
                interface="sysfs",
                path=policy["governor_path"],
                before_value=policy.get("current_governor", NOT_AVAILABLE),
                requested_value="performance",
                requires_root=True,
            )
        )
    return changes


def _profile(
    *,
    profile_id: str,
    mechanism: str,
    scope: str,
    state: str,
    protocols: list[str],
    changes: list[ProfileChange],
    reason: str = "",
    prohibited: list[str] | None = None,
    tradeoffs: list[str] | None = None,
) -> HostSystemProfile:
    if state not in PROFILE_STATES:
        raise SystemProfileError(f"invalid profile state: {state}")
    return HostSystemProfile(
        profile_id=profile_id,
        mechanism=mechanism,
        scope=scope,
        privilege_requirement="root" if any(change.requires_root for change in changes) or state == "requires_root" else "none",
        reboot_requirement=False,
        required_capabilities=_required_caps(profile_id),
        requested_changes=changes,
        prohibited_changes=prohibited or ["unsafe sysctls", "security disablement", "empty IRQ masks", "unbounded realtime scheduling"],
        verification_checks=["before state recorded", "requested state written", "effective state read back", "rollback exact restoration"],
        rollback_plan=["reverse-order restore of original values", "verify exact original values", "mark host_state_dirty on mismatch"],
        workload_protocols=protocols,
        expected_primary_metric=PROTOCOLS[protocols[0]].primary_metric if protocols else NOT_AVAILABLE,
        possible_tradeoffs=tradeoffs or [],
        eligibility_state=state,
        state_reason=reason,
    )


def _unsupported(profile_id: str, reason: str, mechanism: str) -> HostSystemProfile:
    return _profile(profile_id=profile_id, mechanism=mechanism, scope="host_runtime", state="unsupported", protocols=[], changes=[], reason=reason)


def _required_caps(profile_id: str) -> list[str]:
    mapping = {
        "cpufreq-performance": ["cpufreq policy", "performance governor", "writable governor file"],
        "cpuset-partition-1h7w": ["cgroup v2", "cpuset controller", "8 logical CPUs"],
        "cpuset-partition-2h6w": ["cgroup v2", "cpuset controller", "8 logical CPUs"],
        "irq-workqueue-isolation": ["housekeeping/workload split", "writable IRQ or workqueue masks"],
        "hugetlb-bounded": ["HugeTLB pool", "safe reservation guard", "workload HugeTLB mapping proof"],
    }
    return mapping.get(profile_id, [])


def _modes_from_text(text: str) -> list[str]:
    if not text or text == NOT_AVAILABLE:
        return []
    return [item.strip("[]") for item in text.split()]


def _modes_from_snapshot(selected: object) -> list[str]:
    return [str(selected)] if selected and selected != NOT_AVAILABLE else []


def _effective_numa_nodes(snapshot: HostStateSnapshot) -> int:
    value = snapshot.identity.get("numa_topology", {}).get("effective_node_count")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _valid_affinity_masks(snapshot: HostStateSnapshot) -> list[str]:
    online = str(snapshot.identity.get("online_cpus", ""))
    if online == NOT_AVAILABLE:
        return []
    cpus = _expand_cpu_list(online)
    masks = []
    if cpus:
        masks.append(",".join(str(cpu) for cpu in cpus))
    if len(cpus) >= 8:
        masks.extend(["1,2,3,4,5,6,7", "2,3,4,5,6,7"])
    return list(dict.fromkeys(masks))


def _expand_cpu_list(text: str) -> list[int]:
    cpus = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                cpus.extend(range(int(start), int(end) + 1))
            except ValueError:
                return []
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                return []
    return cpus


def _allocator_available(reader: RootReader, allocator: str) -> bool:
    names = {"jemalloc": ["libjemalloc.so", "libjemalloc.so.2"], "mimalloc": ["libmimalloc.so", "libmimalloc.so.2"]}[allocator]
    for pattern in ["/usr/lib*/**/*", "/lib*/**/*", "/run/perfrunbench-fixtures/*"]:
        for path in reader.glob(pattern):
            if path.name in names:
                return True
    return False
