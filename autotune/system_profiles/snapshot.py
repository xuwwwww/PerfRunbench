from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import file_sha256, stable_digest
from autotune.system_profiles.schemas import HostStateSnapshot, NOT_AVAILABLE, versioned_digest


def collect_host_state_snapshot(root: str | Path | None = None, *, snapshot_marker: str | None = None) -> HostStateSnapshot:
    reader = RootReader(root)
    identity = _identity(reader)
    cpu_frequency = _cpu_frequency(reader)
    cpu_idle = _cpu_idle(reader)
    scheduler_cgroup = _scheduler_cgroup(reader, identity)
    interrupts_workqueues = _interrupts_workqueues(reader)
    memory = _memory(reader)
    storage = _storage(reader)
    network = _network(reader)
    security_recovery = _security_recovery(reader)
    host_fingerprint = stable_digest(
        {
            "architecture": identity["architecture"],
            "cpu_model": identity["cpu_model"],
            "logical_cpus": identity["logical_cpus"],
            "online_cpus": identity["online_cpus"],
            "kernel_release": identity["kernel_release"],
            "kernel_command_line": identity["kernel_command_line"],
            "base_page_size": identity["base_page_size"],
        }
    )
    snapshot = HostStateSnapshot(
        host_fingerprint=host_fingerprint,
        identity=identity,
        cpu_frequency=cpu_frequency,
        cpu_idle=cpu_idle,
        scheduler_cgroup=scheduler_cgroup,
        interrupts_workqueues=interrupts_workqueues,
        memory=memory,
        storage=storage,
        network=network,
        security_recovery=security_recovery,
        snapshot_marker=snapshot_marker or NOT_AVAILABLE,
    )
    payload = snapshot.to_dict()
    digest = versioned_digest(payload, ignore_keys={"captured_at_utc", "snapshot_digest"})
    return HostStateSnapshot(**{**payload, "snapshot_digest": digest})


class RootReader:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root else Path("/")
        self.is_fake = self.root != Path("/")

    def path(self, absolute_path: str) -> Path:
        stripped = absolute_path.lstrip("/")
        if self.is_fake:
            return self.root / stripped
        return Path("/") / stripped

    def read_text(self, absolute_path: str) -> str:
        path = self.path(absolute_path)
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return NOT_AVAILABLE

    def exists(self, absolute_path: str) -> bool:
        return self.path(absolute_path).exists()

    def glob(self, absolute_pattern: str) -> list[Path]:
        pattern = absolute_pattern.lstrip("/")
        base = self.root if self.is_fake else Path("/")
        return sorted(base.glob(pattern))

    def relative_abs(self, path: Path) -> str:
        if self.is_fake:
            try:
                return "/" + path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                return str(path)
        return str(path)


def _identity(reader: RootReader) -> dict[str, Any]:
    cpuinfo = reader.read_text("/proc/cpuinfo")
    online = _not_empty(reader.read_text("/sys/devices/system/cpu/online"))
    kernel_release = _not_empty(reader.read_text("/proc/sys/kernel/osrelease"))
    if kernel_release == NOT_AVAILABLE:
        kernel_release = platform.release() if not reader.is_fake else NOT_AVAILABLE
    architecture = _not_empty(reader.read_text("/proc/sys/kernel/arch"))
    if architecture == NOT_AVAILABLE:
        architecture = platform.machine() if not reader.is_fake else NOT_AVAILABLE
    logical_cpus = _logical_cpus(cpuinfo, online, fake=reader.is_fake)
    kernel_config_path = _first_existing(reader, [f"/boot/config-{kernel_release}", "/proc/config.gz"])
    kernel_config_digest = file_sha256(kernel_config_path) if kernel_config_path else NOT_AVAILABLE
    return {
        "host_fingerprint_fields": "architecture,cpu_model,logical_cpus,online_cpus,kernel_release,kernel_command_line,base_page_size",
        "architecture": architecture or NOT_AVAILABLE,
        "cpu_model": _cpu_model(cpuinfo),
        "logical_cpus": logical_cpus,
        "online_cpus": online,
        "allowed_cpu_mask": _allowed_cpu_mask(reader, online),
        "numa_topology": _numa_topology(reader),
        "kernel_release": kernel_release,
        "kernel_flavor": _kernel_flavor(kernel_release),
        "kernel_configuration_digest": kernel_config_digest,
        "boot_id": _not_empty(reader.read_text("/proc/sys/kernel/random/boot_id")),
        "base_page_size": _base_page_size(reader),
        "kernel_command_line": _not_empty(reader.read_text("/proc/cmdline")),
        "distribution_identity": _os_release(reader),
    }


def _cpu_frequency(reader: RootReader) -> dict[str, Any]:
    policies = []
    for policy_dir in reader.glob("/sys/devices/system/cpu/cpufreq/policy*"):
        policy = policy_dir.name
        base = reader.relative_abs(policy_dir)
        policies.append(
            {
                "policy": policy,
                "affected_cpus": _not_empty(reader.read_text(f"{base}/affected_cpus")),
                "current_governor": _not_empty(reader.read_text(f"{base}/scaling_governor")),
                "available_governors": _split_words(reader.read_text(f"{base}/scaling_available_governors")),
                "scaling_min_freq": _not_empty(reader.read_text(f"{base}/scaling_min_freq")),
                "scaling_max_freq": _not_empty(reader.read_text(f"{base}/scaling_max_freq")),
                "energy_performance_preference": _not_empty(reader.read_text(f"{base}/energy_performance_preference")),
                "current_frequency": _not_empty(reader.read_text(f"{base}/scaling_cur_freq")),
                "governor_path": f"{base}/scaling_governor",
                "governor_writable": reader.path(f"{base}/scaling_governor").exists(),
            }
        )
    return {
        "cpufreq_state": "available" if policies else NOT_AVAILABLE,
        "driver": _not_empty(reader.read_text("/sys/devices/system/cpu/cpufreq/policy0/scaling_driver")),
        "policies": policies,
    }


def _cpu_idle(reader: RootReader) -> dict[str, Any]:
    states = []
    for state_dir in reader.glob("/sys/devices/system/cpu/cpu0/cpuidle/state*"):
        base = reader.relative_abs(state_dir)
        states.append(
            {
                "state": state_dir.name,
                "name": _not_empty(reader.read_text(f"{base}/name")),
                "disable": _not_empty(reader.read_text(f"{base}/disable")),
                "usage": _not_empty(reader.read_text(f"{base}/usage")),
                "latency": _not_empty(reader.read_text(f"{base}/latency")),
            }
        )
    return {
        "cpuidle_driver": _not_empty(reader.read_text("/sys/devices/system/cpu/cpuidle/current_driver")),
        "states": states,
        "cpuidle_state": "available" if states else NOT_AVAILABLE,
    }


def _scheduler_cgroup(reader: RootReader, identity: dict[str, Any]) -> dict[str, Any]:
    controllers = _split_words(reader.read_text("/sys/fs/cgroup/cgroup.controllers"))
    return {
        "cgroup_version": "v2" if controllers else NOT_AVAILABLE,
        "cpuset_controller_available": "cpuset" in controllers,
        "controllers": controllers,
        "root_partition": _not_empty(reader.read_text("/sys/fs/cgroup/cpuset.cpus.partition")),
        "existing_partitions": _collect_cgroup_partitions(reader),
        "scheduler_load_balancing_state": _not_empty(reader.read_text("/proc/sys/kernel/sched_domain/cpu0/domain0/flags")),
        "process_affinity": identity.get("allowed_cpu_mask", NOT_AVAILABLE),
        "scheduler_policy": NOT_AVAILABLE,
        "nice_level": NOT_AVAILABLE,
        "exposed_scheduler_sysctls": _sysctl_values(reader, ["kernel/sched_autogroup_enabled", "kernel/sched_rt_runtime_us"]),
    }


def _interrupts_workqueues(reader: RootReader) -> dict[str, Any]:
    irq_masks = {}
    for path in reader.glob("/proc/irq/*/smp_affinity_list"):
        irq_masks[path.parent.name] = _not_empty(path.read_text(encoding="utf-8", errors="replace").strip())
    workqueues = {}
    for path in reader.glob("/sys/devices/virtual/workqueue/*/cpumask"):
        workqueues[path.parent.name] = _not_empty(path.read_text(encoding="utf-8", errors="replace").strip())
    return {
        "irqbalance_service_state": _not_empty(reader.read_text("/run/perfrunbench-fixtures/irqbalance_state")),
        "interrupts_snapshot": _not_empty(reader.read_text("/proc/interrupts")),
        "irq_affinity_masks": irq_masks or NOT_AVAILABLE,
        "managed_irq_evidence": _not_empty(reader.read_text("/run/perfrunbench-fixtures/managed_irqs")),
        "softirq_counters": _not_empty(reader.read_text("/proc/softirqs")),
        "unbound_workqueue_cpumask": _not_empty(reader.read_text("/sys/devices/virtual/workqueue/cpumask")),
        "visible_workqueue_cpumasks": workqueues or NOT_AVAILABLE,
    }


def _memory(reader: RootReader) -> dict[str, Any]:
    meminfo = _key_value_lines(reader.read_text("/proc/meminfo"))
    thp_root = "/sys/kernel/mm/transparent_hugepage"
    hugepages = {}
    for path in reader.glob("/sys/kernel/mm/hugepages/hugepages-*"):
        base = reader.relative_abs(path)
        hugepages[path.name] = {
            "nr_hugepages": _not_empty(reader.read_text(f"{base}/nr_hugepages")),
            "free_hugepages": _not_empty(reader.read_text(f"{base}/free_hugepages")),
        }
    return {
        "total_memory": meminfo.get("MemTotal", NOT_AVAILABLE),
        "available_memory": meminfo.get("MemAvailable", NOT_AVAILABLE),
        "swap": {"total": meminfo.get("SwapTotal", NOT_AVAILABLE), "free": meminfo.get("SwapFree", NOT_AVAILABLE)},
        "base_page_size": _base_page_size(reader),
        "thp_top_level_mode": _selected_mode(reader.read_text(f"{thp_root}/enabled")),
        "available_thp_sizes": [path.name for path in reader.glob(f"{thp_root}/hugepages-*")] or NOT_AVAILABLE,
        "per_size_thp_mode": _per_size_thp(reader),
        "thp_defrag_mode": _selected_mode(reader.read_text(f"{thp_root}/defrag")),
        "thp_available_defrag_modes": _available_modes(reader.read_text(f"{thp_root}/defrag")),
        "khugepaged_settings": _directory_values(reader, f"{thp_root}/khugepaged"),
        "hugetlb_page_sizes": hugepages or NOT_AVAILABLE,
        "compaction_settings": _sysctl_values(reader, ["vm/compaction_proactiveness", "vm/compact_unevictable_allowed"]),
        "numa_balancing_state": _not_empty(reader.read_text("/proc/sys/kernel/numa_balancing")),
        "relevant_vm_sysctls": _sysctl_values(reader, ["vm/swappiness", "vm/stat_interval", "vm/zone_reclaim_mode"]),
        "vmstat": _not_empty(reader.read_text("/proc/vmstat")),
        "buddyinfo": _not_empty(reader.read_text("/proc/buddyinfo")),
        "pagetypeinfo": _not_empty(reader.read_text("/proc/pagetypeinfo")),
    }


def _storage(reader: RootReader) -> dict[str, Any]:
    schedulers = {}
    read_ahead = {}
    for path in reader.glob("/sys/block/*/queue/scheduler"):
        schedulers[path.parent.parent.name] = {
            "current": _selected_mode(path.read_text(encoding="utf-8", errors="replace").strip()),
            "available": _available_modes(path.read_text(encoding="utf-8", errors="replace").strip()),
        }
    for path in reader.glob("/sys/block/*/queue/read_ahead_kb"):
        read_ahead[path.parent.parent.name] = _not_empty(path.read_text(encoding="utf-8", errors="replace").strip())
    return {
        "model_device": _not_empty(reader.read_text("/run/perfrunbench-fixtures/model_device")),
        "filesystem": _not_empty(reader.read_text("/run/perfrunbench-fixtures/model_filesystem")),
        "mount_options": _not_empty(reader.read_text("/run/perfrunbench-fixtures/model_mount_options")),
        "io_schedulers": schedulers or NOT_AVAILABLE,
        "block_read_ahead": read_ahead or NOT_AVAILABLE,
        "device_topology": _not_empty(reader.read_text("/run/perfrunbench-fixtures/block_topology")),
        "model_file_mmap_eligibility": _not_empty(reader.read_text("/run/perfrunbench-fixtures/model_mmap_eligibility")),
    }


def _network(reader: RootReader) -> dict[str, Any]:
    interfaces = {}
    for iface_dir in reader.glob("/sys/class/net/*"):
        if iface_dir.name == "lo":
            continue
        base = reader.relative_abs(iface_dir)
        queues = [path.name for path in reader.glob(f"{base}/queues/*")]
        interfaces[iface_dir.name] = {"queues": queues, "carrier": _not_empty(reader.read_text(f"{base}/carrier"))}
    return {
        "interfaces": interfaces or NOT_AVAILABLE,
        "queue_counts": {name: len(item["queues"]) for name, item in interfaces.items()} if interfaces else NOT_AVAILABLE,
        "irq_mapping": _not_empty(reader.read_text("/run/perfrunbench-fixtures/network_irq_mapping")),
        "rps_masks": _queue_masks(reader, "rps_cpus"),
        "xps_masks": _queue_masks(reader, "xps_cpus"),
        "queue_backlog_counters": _not_empty(reader.read_text("/proc/net/softnet_stat")),
    }


def _security_recovery(reader: RootReader) -> dict[str, Any]:
    return {
        "root_privilege": os.geteuid() == 0 if hasattr(os, "geteuid") and not reader.is_fake else NOT_AVAILABLE,
        "writable_interfaces": _not_empty(reader.read_text("/run/perfrunbench-fixtures/writable_interfaces")),
        "lockdown_state": _not_empty(reader.read_text("/sys/kernel/security/lockdown")),
        "grub_configuration_location": _first_existing_text(reader, ["/etc/default/grub", "/boot/grub/grub.cfg"]),
        "available_kernels": _available_kernels(reader),
        "currently_selected_kernel_flavor": _kernel_flavor(_not_empty(reader.read_text("/proc/sys/kernel/osrelease"))),
        "snapshot_marker_supplied_by_user": _not_empty(reader.read_text("/run/perfrunbench-fixtures/snapshot_marker")),
        "boot_change_authorization_state": _not_empty(reader.read_text("/run/perfrunbench-fixtures/boot_change_authorized")),
    }


def _first_existing(reader: RootReader, paths: list[str]) -> Path | None:
    for item in paths:
        path = reader.path(item)
        if path.exists():
            return path
    return None


def _first_existing_text(reader: RootReader, paths: list[str]) -> str:
    for item in paths:
        if reader.exists(item):
            return item
    return NOT_AVAILABLE


def _not_empty(value: str | None) -> str:
    if value is None:
        return NOT_AVAILABLE
    text = str(value).strip()
    return text if text and text != NOT_AVAILABLE else NOT_AVAILABLE


def _split_words(value: str) -> list[str]:
    if not value or value == NOT_AVAILABLE:
        return []
    return [item for item in value.split() if item]


def _cpu_model(cpuinfo: str) -> str:
    if cpuinfo == NOT_AVAILABLE:
        return NOT_AVAILABLE
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, value = [item.strip() for item in line.split(":", 1)]
        if key in {"model name", "Hardware", "Processor", "CPU part", "CPU implementer"} and value:
            return value
    return NOT_AVAILABLE


def _logical_cpus(cpuinfo: str, online: str, *, fake: bool) -> int | str:
    processors = len(re.findall(r"^processor\s*:", cpuinfo, flags=re.MULTILINE)) if cpuinfo != NOT_AVAILABLE else 0
    if processors:
        return processors
    parsed = _count_cpu_list(online)
    if parsed:
        return parsed
    if not fake:
        return os.cpu_count() or NOT_AVAILABLE
    return NOT_AVAILABLE


def _allowed_cpu_mask(reader: RootReader, online: str) -> str:
    fixture = _not_empty(reader.read_text("/run/perfrunbench-fixtures/allowed_cpu_mask"))
    if fixture != NOT_AVAILABLE:
        return fixture
    if not reader.is_fake and hasattr(os, "sched_getaffinity"):
        try:
            return ",".join(str(item) for item in sorted(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
        except OSError:
            pass
    return online


def _numa_topology(reader: RootReader) -> dict[str, Any]:
    online = _not_empty(reader.read_text("/sys/devices/system/node/online"))
    nodes = []
    for node_dir in reader.glob("/sys/devices/system/node/node*"):
        base = reader.relative_abs(node_dir)
        nodes.append({"node": node_dir.name, "cpulist": _not_empty(reader.read_text(f"{base}/cpulist"))})
    return {"online_nodes": online, "nodes": nodes, "effective_node_count": len(nodes) if nodes else NOT_AVAILABLE}


def _kernel_flavor(release: str) -> str:
    if not release or release == NOT_AVAILABLE:
        return NOT_AVAILABLE
    lowered = release.lower()
    if "64k" in lowered:
        return "aws-64k" if "aws" in lowered else "64k"
    if "aws" in lowered:
        return "aws-4k"
    return "generic"


def _base_page_size(reader: RootReader) -> str:
    fixture = _not_empty(reader.read_text("/run/perfrunbench-fixtures/base_page_size"))
    if fixture != NOT_AVAILABLE:
        return fixture
    if not reader.is_fake and hasattr(os, "sysconf"):
        try:
            return str(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            pass
    return NOT_AVAILABLE


def _os_release(reader: RootReader) -> dict[str, str]:
    text = reader.read_text("/etc/os-release")
    if text == NOT_AVAILABLE:
        return {"state": NOT_AVAILABLE}
    result = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result or {"state": NOT_AVAILABLE}


def _collect_cgroup_partitions(reader: RootReader) -> list[dict[str, str]]:
    partitions = []
    for path in reader.glob("/sys/fs/cgroup/**/cpuset.cpus.partition"):
        partitions.append({"path": reader.relative_abs(path), "state": _not_empty(path.read_text(encoding="utf-8", errors="replace").strip())})
    return partitions


def _sysctl_values(reader: RootReader, names: list[str]) -> dict[str, str]:
    values = {}
    for name in names:
        values[name.replace("/", ".")] = _not_empty(reader.read_text(f"/proc/sys/{name}"))
    return values


def _directory_values(reader: RootReader, absolute_path: str) -> dict[str, str] | str:
    base = reader.path(absolute_path)
    if not base.exists() or not base.is_dir():
        return NOT_AVAILABLE
    values = {}
    for child in sorted(base.iterdir()):
        if child.is_file():
            try:
                values[child.name] = _not_empty(child.read_text(encoding="utf-8", errors="replace").strip())
            except OSError:
                values[child.name] = NOT_AVAILABLE
    return values or NOT_AVAILABLE


def _key_value_lines(text: str) -> dict[str, str]:
    if text == NOT_AVAILABLE:
        return {}
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _selected_mode(text: str) -> str:
    if not text or text == NOT_AVAILABLE:
        return NOT_AVAILABLE
    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        return match.group(1)
    return text.split()[0] if text.split() else NOT_AVAILABLE


def _available_modes(text: str) -> list[str]:
    if not text or text == NOT_AVAILABLE:
        return []
    return [item.strip("[]") for item in text.split()]


def _per_size_thp(reader: RootReader) -> dict[str, str] | str:
    values = {}
    for path in reader.glob("/sys/kernel/mm/transparent_hugepage/hugepages-*/enabled"):
        values[path.parent.name] = _selected_mode(path.read_text(encoding="utf-8", errors="replace").strip())
    return values or NOT_AVAILABLE


def _queue_masks(reader: RootReader, filename: str) -> dict[str, str] | str:
    values = {}
    for path in reader.glob(f"/sys/class/net/*/queues/*/{filename}"):
        values[reader.relative_abs(path)] = _not_empty(path.read_text(encoding="utf-8", errors="replace").strip())
    return values or NOT_AVAILABLE


def _available_kernels(reader: RootReader) -> list[str] | str:
    kernels = [path.name.removeprefix("vmlinuz-") for path in reader.glob("/boot/vmlinuz-*")]
    return kernels or NOT_AVAILABLE


def _count_cpu_list(text: str) -> int:
    if not text or text == NOT_AVAILABLE:
        return 0
    count = 0
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                count += int(end) - int(start) + 1
            except ValueError:
                return 0
        else:
            try:
                int(part)
            except ValueError:
                return 0
            count += 1
    return count
