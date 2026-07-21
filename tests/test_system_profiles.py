from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autotune.cli import main as autotune_main
from autotune.recommendation.bundle import create_recommendation_bundle, validate_recommendation_bundle
from autotune.recommendation.schemas import atomic_write_json
from autotune.system_profiles.boot import build_boot_profile
from autotune.system_profiles.catalog import (
    PROHIBITED_PROFILE_IDS,
    collect_capabilities,
    create_plan,
    list_profiles,
    staged_search_policy,
)
from autotune.system_profiles.lease import apply_system_profile_plan, rollback_system_profile, verify_system_profile
from autotune.system_profiles.protocols import list_protocols
from autotune.system_profiles.schemas import NOT_AVAILABLE
from autotune.system_profiles.snapshot import RootReader, _queue_masks, collect_host_state_snapshot


class SystemProfilesTest(unittest.TestCase):
    def test_complete_snapshot_records_capabilities_and_stable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            first = collect_host_state_snapshot(root, snapshot_marker="manual-snapshot-ok")
            second = collect_host_state_snapshot(root, snapshot_marker="manual-snapshot-ok")

            self.assertEqual(first.snapshot_digest, second.snapshot_digest)
            self.assertEqual(first.identity["architecture"], "aarch64")
            self.assertEqual(first.identity["cpu_model"], "Neoverse-V1")
            self.assertEqual(first.identity["logical_cpus"], 8)
            self.assertEqual(first.cpu_frequency["policies"][0]["current_governor"], "powersave")
            self.assertEqual(first.memory["thp_top_level_mode"], "madvise")
            self.assertEqual(first.memory["thp_defrag_mode"], "madvise")

            report = collect_capabilities(first, root)
            self.assertTrue(report.capabilities["cpufreq_performance_governor"])
            self.assertTrue(report.capabilities["cgroup_v2_cpuset"])
            self.assertTrue(report.capabilities["irq_affinity_available"])
            self.assertEqual(report.profile_states["cpuset-partition-1h7w"], "requires_root")

    def test_unavailable_fields_remain_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = collect_host_state_snapshot(root)

            self.assertEqual(snapshot.identity["cpu_model"], NOT_AVAILABLE)
            self.assertEqual(snapshot.cpu_frequency["cpufreq_state"], NOT_AVAILABLE)
            self.assertEqual(snapshot.memory["total_memory"], NOT_AVAILABLE)
            self.assertEqual(snapshot.storage["model_device"], NOT_AVAILABLE)

            profiles = {item["profile_id"]: item for item in list_profiles(snapshot, root)}
            self.assertEqual(profiles["cpufreq-performance"]["eligibility_state"], "unsupported")
            self.assertEqual(profiles["numa-policy"]["eligibility_state"], "unsupported")

    def test_volatile_network_queue_masks_remain_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = root / "sys/class/net/eth0/queues/rx-0/rps_cpus"
            mask.parent.mkdir(parents=True)
            mask.write_text("ff\n", encoding="utf-8")
            original_read_text = Path.read_text

            def flaky_read_text(path: Path, *args: object, **kwargs: object) -> str:
                if path == mask:
                    raise FileNotFoundError(str(path))
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", flaky_read_text):
                masks = _queue_masks(RootReader(root), "rps_cpus")

            self.assertEqual(masks["/sys/class/net/eth0/queues/rx-0/rps_cpus"], NOT_AVAILABLE)

    def test_profile_catalog_rejects_unsafe_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)
            for profile_id in PROHIBITED_PROFILE_IDS:
                profile = create_plan(profile_id, snapshot, root).profile
                self.assertEqual(profile.eligibility_state, "unsafe")
                self.assertTrue(profile.prohibited_changes)

    def test_cpufreq_apply_verify_and_rollback_are_lease_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)
            plan = create_plan("cpufreq-performance", snapshot, root)
            governor = root / "sys/devices/system/cpu/cpufreq/policy0/scaling_governor"

            result = apply_system_profile_plan(plan, snapshot, root=root, lease_minutes=15)

            self.assertEqual(result.apply_status, "applied")
            self.assertEqual(governor.read_text(encoding="utf-8").strip(), "performance")
            verification = verify_system_profile(plan, result.lease, root=root)
            self.assertEqual(verification["verification_status"], "passed")

            rollback = rollback_system_profile(snapshot, result.lease, root=root)

            self.assertEqual(rollback.rollback_status, "passed")
            self.assertEqual(governor.read_text(encoding="utf-8").strip(), "powersave")
            self.assertFalse(rollback.host_state_dirty)

    def test_apply_refuses_real_system_without_fake_root(self) -> None:
        snapshot = collect_host_state_snapshot()
        plan = create_plan("baseline-current", snapshot)
        with self.assertRaisesRegex(Exception, "refuses real system writes"):
            apply_system_profile_plan(plan, snapshot)

    def test_dirty_host_refusal_marks_dirty_and_does_not_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)
            plan = create_plan("cpufreq-performance", snapshot, root)
            governor = root / "sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
            governor.write_text("ondemand\n", encoding="utf-8")

            result = apply_system_profile_plan(plan, snapshot, root=root)

            self.assertEqual(result.apply_status, "failed")
            self.assertTrue(result.host_state_dirty)
            self.assertEqual(governor.read_text(encoding="utf-8").strip(), "ondemand")

    def test_partial_apply_rolls_back_reverse_order_and_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root, second_policy=True)
            snapshot = collect_host_state_snapshot(root)
            plan = create_plan("cpufreq-performance", snapshot, root)
            second = root / "sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
            second.unlink()

            result = apply_system_profile_plan(plan, snapshot, root=root)
            rollback = rollback_system_profile(snapshot, result.lease, root=root)

            self.assertEqual(result.apply_status, "failed")
            self.assertTrue(result.host_state_dirty)
            self.assertEqual(rollback.rollback_status, "passed")
            self.assertEqual((root / "sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read_text(encoding="utf-8").strip(), "powersave")

    def test_thp_hugetlb_and_numa_states_are_capability_driven(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)

            thp = create_plan("thp-always", snapshot, root).profile
            hugetlb = create_plan("hugetlb-bounded", snapshot, root).profile
            numa = create_plan("numa-policy", snapshot, root).profile

            self.assertEqual(thp.eligibility_state, "requires_root")
            self.assertEqual(hugetlb.eligibility_state, "partially_supported")
            self.assertIn("supported_but_not_exercised", hugetlb.state_reason)
            self.assertEqual(numa.eligibility_state, "unsupported")

    def test_staged_search_keeps_protocols_separate_and_avoids_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)
            policy = staged_search_policy(snapshot, root)
            protocols = {item["protocol_id"]: item for item in list_protocols()}

            self.assertFalse(policy["cartesian_product"])
            self.assertEqual(policy["stages"][0]["profiles"], ["baseline-current"])
            self.assertIn("cold_model_load", protocols["steady_state_decode"]["cannot_combine_with"])
            self.assertNotEqual(protocols["steady_state_decode"]["primary_metric"], protocols["cold_model_load"]["primary_metric"])

    def test_boot_profile_requires_kernel_config_fallback_snapshot_and_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root, snapshot_marker="manual-snapshot-ok")

            plan = build_boot_profile("aws-4k-nohz-rcu-irq", snapshot, root=root, allow_boot_change=True)
            advanced = build_boot_profile("aws-4k-domain-isolation", snapshot, root=root, allow_boot_change=True)

            self.assertEqual(plan.eligibility_state, "requires_reboot")
            self.assertIn("nohz_full=2-7", plan.generated_command_line)
            self.assertEqual(advanced.eligibility_state, "requires_reboot")
            self.assertTrue(advanced.less_flexible_boot_isolation)

    def test_boot_profile_rejects_missing_support_and_missing_snapshot_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root, kernel_config=False)
            snapshot = collect_host_state_snapshot(root)

            missing_support = build_boot_profile("aws-4k-nohz-rcu-irq", snapshot, root=root)

            self.assertEqual(missing_support.eligibility_state, "unsupported")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_root(root)
            snapshot = collect_host_state_snapshot(root)

            missing_marker = build_boot_profile("aws-4k-nohz-rcu-irq", snapshot, root=root)

            self.assertEqual(missing_marker.eligibility_state, "requires_user_snapshot")

    def test_cli_fake_e2e_doctor_snapshot_apply_verify_rollback_boot_plan_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            _write_fixture_root(root)
            snapshot_path = Path(tmp) / "host-state.json"
            lease_path = Path(tmp) / "lease.json"

            self.assertEqual(_run_cli(["system", "doctor", "--root", str(root), "--json"])["status"], "passed")
            self.assertEqual(
                _run_cli_code(["system", "snapshot", "--root", str(root), "--output", str(snapshot_path), "--snapshot-marker", "marker"]),
                0,
            )
            plan_payload = _run_cli(["system", "plan", "--root", str(root), "--catalog", "safe-system", "--json"])
            self.assertFalse(plan_payload["staged_search_policy"]["cartesian_product"])
            self.assertEqual(_run_cli_code(["system", "show-profile", "cpufreq-performance", "--root", str(root), "--json"]), 0)
            self.assertEqual(
                _run_cli_code([
                    "system",
                    "apply",
                    "--profile",
                    "cpufreq-performance",
                    "--lease-minutes",
                    "5",
                    "--snapshot",
                    str(snapshot_path),
                    "--lease-output",
                    str(lease_path),
                    "--root",
                    str(root),
                    "--json",
                ]),
                0,
            )
            self.assertEqual(
                _run_cli_code(["system", "verify", "--profile", "cpufreq-performance", "--snapshot", str(snapshot_path), "--lease", str(lease_path), "--root", str(root), "--json"]),
                0,
            )
            self.assertEqual(
                _run_cli_code(["system", "rollback", "--snapshot", str(snapshot_path), "--lease", str(lease_path), "--root", str(root), "--json"]),
                0,
            )
            self.assertEqual(_run_cli_code(["system", "show-profile", "cpufreq-performance", "--root", str(root), "--json"]), 0)
            self.assertEqual(_run_cli_code(["system", "boot-plan", "--profile", "aws-4k-default", "--root", str(root), "--json"]), 0)
            compare = _run_cli(["system", "compare", "--run-id", "fixture-run", "--json"])
            self.assertEqual(compare["score_combination"], "prohibited")

    def test_recommendation_bundle_rejects_dirty_host_profile_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            passport = Path(tmp) / "passport.json"
            atomic_write_json(passport, {"schema_version": "1.0", "claim_eligibility_state": "eligible_preliminary"})
            bundle = create_recommendation_bundle(
                source_run_id="20260721_000000_000000",
                source_passport_path=passport,
                status="eligible_preliminary",
                objective={"metric": "throughput", "direction": "maximize"},
                baseline_configuration={"launch_spec": {"schema_version": "1.0", "argv": ["python", "-c", "print(1)"]}},
                selected_configuration={"configuration_id": "selected", "launch_spec": {"schema_version": "1.0", "argv": ["python", "-c", "print(1)"]}},
                statistical_evidence={"status": "recommended"},
                provenance={"result_origin": "measured"},
            )
            bundle["host_profile_evidence"] = {"host_state_dirty": True}

            validation = validate_recommendation_bundle(bundle)

            self.assertEqual(validation["status"], "fail")
            self.assertTrue(any(item["code"] == "dirty_host_state" for item in validation["guard_results"]))


def _run_cli(argv: list[str]) -> dict[str, object]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = autotune_main(argv)
    if code != 0:
        raise AssertionError(f"CLI failed with {code}: {buffer.getvalue()}")
    return json.loads(buffer.getvalue())


def _run_cli_code(argv: list[str]) -> int:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return autotune_main(argv)


def _write_fixture_root(root: Path, *, kernel_config: bool = True, second_policy: bool = False) -> None:
    files = {
        "proc/sys/kernel/arch": "aarch64\n",
        "proc/sys/kernel/osrelease": "6.17.0-1019-aws\n",
        "proc/sys/kernel/random/boot_id": "11111111-2222-3333-4444-555555555555\n",
        "proc/cmdline": "BOOT_IMAGE=/boot/vmlinuz-6.17.0-1019-aws root=LABEL=cloudimg-rootfs ro\n",
        "proc/cpuinfo": "\n".join(f"processor\t: {i}\nmodel name\t: Neoverse-V1" for i in range(8)) + "\n",
        "proc/meminfo": "MemTotal:       16384000 kB\nMemAvailable:  12000000 kB\nSwapTotal:             0 kB\nSwapFree:              0 kB\nAnonHugePages:    2048 kB\n",
        "proc/vmstat": "thp_fault_alloc 10\nthp_collapse_alloc 2\ncompact_stall 0\n",
        "proc/interrupts": " 32: 1 2 3 4 GICv3 test\n",
        "proc/softirqs": "NET_RX: 1 2 3 4\n",
        "proc/sys/vm/compaction_proactiveness": "20\n",
        "proc/sys/vm/stat_interval": "1\n",
        "proc/sys/vm/swappiness": "60\n",
        "proc/sys/vm/zone_reclaim_mode": "0\n",
        "proc/sys/kernel/numa_balancing": "0\n",
        "proc/sys/kernel/sched_autogroup_enabled": "1\n",
        "sys/devices/system/cpu/online": "0-7\n",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "cppc_cpufreq\n",
        "sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0-7\n",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_governor": "powersave\n",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_available_governors": "performance powersave\n",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq": "1000000\n",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq": "3000000\n",
        "sys/devices/system/cpu/cpu0/cpuidle/state0/name": "WFI\n",
        "sys/devices/system/cpu/cpu0/cpuidle/state0/disable": "0\n",
        "sys/devices/system/cpu/cpu0/cpuidle/state0/usage": "10\n",
        "sys/devices/system/cpu/cpu0/cpuidle/state0/latency": "1\n",
        "sys/devices/system/cpu/cpuidle/current_driver": "arm_idle\n",
        "sys/devices/system/node/online": "0\n",
        "sys/devices/system/node/node0/cpulist": "0-7\n",
        "sys/fs/cgroup/cgroup.controllers": "cpuset cpu memory io\n",
        "sys/fs/cgroup/cpuset.cpus.partition": "root\n",
        "sys/kernel/mm/transparent_hugepage/enabled": "always [madvise] never\n",
        "sys/kernel/mm/transparent_hugepage/defrag": "always defer defer+madvise [madvise] never\n",
        "sys/kernel/mm/transparent_hugepage/khugepaged/pages_to_scan": "4096\n",
        "sys/kernel/mm/transparent_hugepage/hugepages-2048kB/enabled": "always [madvise] never\n",
        "sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages": "0\n",
        "sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages": "0\n",
        "proc/irq/32/smp_affinity_list": "0-7\n",
        "sys/devices/virtual/workqueue/cpumask": "0-7\n",
        "sys/block/nvme0n1/queue/scheduler": "none [mq-deadline]\n",
        "sys/block/nvme0n1/queue/read_ahead_kb": "128\n",
        "etc/os-release": "ID=ubuntu\nVERSION_ID=\"24.04\"\n",
        "boot/vmlinuz-6.17.0-1019-aws": "",
        "boot/vmlinuz-6.17.0-1018-aws": "",
        "run/perfrunbench-fixtures/base_page_size": "4096\n",
        "run/perfrunbench-fixtures/allowed_cpu_mask": "0,1,2,3,4,5,6,7\n",
        "run/perfrunbench-fixtures/irqbalance_state": "running\n",
        "run/perfrunbench-fixtures/snapshot_marker": "manual-snapshot-ok\n",
        "run/perfrunbench-fixtures/boot_change_authorized": "false\n",
    }
    if second_policy:
        files.update(
            {
                "sys/devices/system/cpu/cpufreq/policy4/affected_cpus": "4-7\n",
                "sys/devices/system/cpu/cpufreq/policy4/scaling_governor": "powersave\n",
                "sys/devices/system/cpu/cpufreq/policy4/scaling_available_governors": "performance powersave\n",
            }
        )
    if kernel_config:
        files["boot/config-6.17.0-1019-aws"] = "CONFIG_NO_HZ_FULL=y\nCONFIG_RCU_NOCB_CPU=y\n"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
