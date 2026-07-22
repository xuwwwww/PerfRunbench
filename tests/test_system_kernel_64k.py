from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from autotune.cli import main as autotune_main
from autotune.recommendation.schemas import atomic_write_json
from autotune.system_profiles.boot_readiness import collect_boot_inventory, build_one_time_boot_plan, verify_one_time_boot_plan
from autotune.system_profiles.kernel_64k import (
    build_grub_default_plan,
    build_kernel_cleanup_plan,
    build_kernel_experiment_plan,
    build_kernel_package_transaction_plan,
    parse_grub_menu_entries,
    qualify_generated_grub_entry,
    render_temporary_grub_entry_from_generated,
    validate_phase21c2b_comparison,
    verify_kernel_experiment_plan,
    verify_kernel_package_transaction_plan,
)
from tests.test_system_profiles import _write_phase21c0_boot_fixture


class SystemKernel64KTest(unittest.TestCase):
    def test_grub_parser_preserves_generated_paths_and_rejects_guesses(self) -> None:
        text = "\n".join(
            [
                "submenu 'Advanced options for Ubuntu' --id 'gnulinux-advanced-root' {",
                "menuentry 'Ubuntu, with Linux 6.17.0-1019-aws' --id 'gnulinux-6.17.0-1019-aws-advanced-root' {",
                "  insmod gzio",
                "  search --no-floppy --fs-uuid --set=root 1111-2222",
                "  linux /vmlinuz-6.17.0-1019-aws root=LABEL=cloudimg-rootfs ro",
                "  initrd /initrd.img-6.17.0-1019-aws",
                "}",
                "menuentry 'Ubuntu, with Linux 6.17.0-1019-aws (recovery mode)' --id 'recovery' {",
                "  linux /vmlinuz-6.17.0-1019-aws single recovery",
                "  initrd /initrd.img-6.17.0-1019-aws",
                "}",
                "}",
            ]
        )

        entries = parse_grub_menu_entries(text)
        qualified = qualify_generated_grub_entry(entries, kernel_release="6.17.0-1019-aws", require_initrd=True)

        self.assertEqual(qualified["status"], "qualified")
        self.assertEqual(qualified["entry"]["linux_path"], "/vmlinuz-6.17.0-1019-aws")
        self.assertEqual(qualified["entry"]["initrd_path"], "/initrd.img-6.17.0-1019-aws")
        self.assertIn("search --no-floppy", " ".join(qualified["entry"]["loader_prelude"]))

        duplicate = (
            text.replace("(recovery mode)", "duplicate alternate")
            .replace("--id 'recovery'", "--id 'duplicate-entry'")
            .replace(" single recovery", " root=LABEL=cloudimg-rootfs ro")
        )
        duplicate_result = qualify_generated_grub_entry(parse_grub_menu_entries(duplicate), kernel_release="6.17.0-1019-aws", require_initrd=True)
        self.assertIn("ambiguous_duplicate_kernel_entries", duplicate_result["issues"])

        malformed = "menuentry 'bad' --id 'bad' {\n  linux\n}\n"
        malformed_result = qualify_generated_grub_entry(parse_grub_menu_entries(malformed), kernel_release="6.17.0-1019-aws", require_initrd=True)
        self.assertIn("generated_kernel_entry_missing", malformed_result["issues"])

    def test_temporary_entry_preserves_generated_initrdfail_conditional(self) -> None:
        text = "\n".join(
            [
                "menuentry 'Ubuntu' --id 'Ubuntu' {",
                "    recordfail",
                "    search --no-floppy --fs-uuid --set=root 1111-2222",
                "    if [ \"${initrdfail}\" = 1 ]; then",
                "    echo 'GRUB_FORCE_PARTUUID set, initrdless boot failed. Attempting with initrd.'",
                "    linux /vmlinuz-6.17.0-1019-aws-64k root=PARTUUID=abc ro",
                "    initrd /initrd.img-6.17.0-1019-aws-64k",
                "    else",
                "    linux /vmlinuz-6.17.0-1019-aws-64k root=PARTUUID=abc ro initrd=\\EFI\\ubuntu\\initrd.img-6.17.0-1019-aws-64k",
                "    fi",
                "}",
            ]
        )
        entry = parse_grub_menu_entries(text)[0]

        rendered = render_temporary_grub_entry_from_generated(
            title="PerfRunbench Phase 21C2B 64k-nohz",
            entry_id="perfrunbench-phase21c2b-64k-nohz",
            source_entry=entry,
            linux_arguments="root=PARTUUID=abc ro nohz_full=2-7 rcu_nocbs=2-7 irqaffinity=0-1",
        )

        self.assertIn("if [ \"${initrdfail}\" = 1 ]; then", rendered)
        self.assertIn("else", rendered)
        self.assertIn("    fi", rendered)
        self.assertEqual(rendered.count("nohz_full=2-7"), 2)
        self.assertIn("initrd /initrd.img-6.17.0-1019-aws-64k", rendered)

    def test_phase21c2a_plan_uses_generated_root_relative_path_not_boot_guess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_phase21c0_boot_fixture(root)
            (root / "boot/grub/grub.cfg").write_text(
                "\n".join(
                    [
                        "menuentry 'Ubuntu, with Linux 6.17.0-1019-aws' --id 'gnulinux-6.17.0-1019-aws-advanced-root' {",
                        "  search --no-floppy --partuuid --set=root abcdef",
                        "  linux /vmlinuz-6.17.0-1019-aws root=LABEL=cloudimg-rootfs ro",
                        "  initrd /initrd.img-6.17.0-1019-aws",
                        "}",
                        "menuentry 'Ubuntu, with Linux 6.17.0-1018-aws' --id 'gnulinux-6.17.0-1018-aws-advanced-root' {",
                        "  search --no-floppy --partuuid --set=root abcdef",
                        "  linux /vmlinuz-6.17.0-1018-aws root=LABEL=cloudimg-rootfs ro",
                        "  initrd /initrd.img-6.17.0-1018-aws",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            inventory = collect_boot_inventory(root)

            plan = build_one_time_boot_plan("4k-nohz-rcu-irq", inventory)
            raw = plan.to_dict()

            self.assertEqual(plan.candidate_vmlinuz_path, "/vmlinuz-6.17.0-1019-aws")
            self.assertEqual(raw["temporary_entry_plan"]["candidate_vmlinuz_path"], "/vmlinuz-6.17.0-1019-aws")
            self.assertNotIn("linux /boot/vmlinuz-6.17.0-1019-aws", raw["temporary_entry_plan"]["menuentry_text"])
            self.assertEqual(raw["temporary_entry_plan"]["path_source"], "generated_grub_menu_entry")
            self.assertFalse(raw["temporary_entry_plan"]["path_guess_allowed"])
            self.assertEqual(verify_one_time_boot_plan(raw)["status"], "passed")

            guessed = json.loads(json.dumps(raw))
            guessed["temporary_entry_plan"]["path_source"] = "guessed_from_boot_filename"
            guessed["temporary_entry_plan"]["path_guess_allowed"] = True
            verification = verify_one_time_boot_plan(guessed)

            self.assertIn("temporary_entry_not_based_on_generated_grub_entry", verification["issues"])
            self.assertIn("guessed_kernel_path_allowed", verification["issues"])

    def test_kernel_package_transaction_accepts_only_safe_official_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_phase21c0_boot_fixture(root)
            inventory = collect_boot_inventory(root).to_dict()
            simulation = "\n".join(
                [
                    "Inst linux-image-aws-64k (6.17.0-1020.20 Ubuntu:24.04 [arm64])",
                    "Inst linux-modules-aws-64k (6.17.0-1020.20 Ubuntu:24.04 [arm64])",
                ]
            )

            plan = build_kernel_package_transaction_plan(inventory, apt_simulation_text=simulation)

            self.assertEqual(plan.eligibility_state, "planning_ready")
            self.assertFalse(plan.kernel_or_boot_state_modified)
            self.assertIn("linux-image-aws-64k=6.17.0-1020.20", plan.real_install_command)
            self.assertIn("linux-image-aws-64k", plan.rollback_plan.remove_packages)
            self.assertNotIn("autoremove", " ".join(plan.rollback_plan.remove_packages))
            self.assertEqual(verify_kernel_package_transaction_plan(plan.to_dict())["status"], "passed")

    def test_kernel_package_transaction_rejects_unsafe_apt_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_phase21c0_boot_fixture(root)
            inventory = collect_boot_inventory(root).to_dict()
            unsafe = "\n".join(
                [
                    "WARNING: The following packages cannot be authenticated!",
                    "Inst linux-image-aws-64k (6.17.0-1020.20 Ubuntu:24.04 [arm64])",
                    "Inst libc6 [2.39] (2.40 Ubuntu:24.04 [arm64])",
                    "Remv linux-image-6.17.0-1019-aws [6.17.0-1019.19]",
                    "Inst grub-efi-arm64 [2.12] (2.13 Ubuntu:24.04 [arm64])",
                ]
            )

            plan = build_kernel_package_transaction_plan(inventory, apt_simulation_text=unsafe)

            self.assertEqual(plan.eligibility_state, "blocked")
            self.assertIn("unsigned_or_untrusted_packages", plan.blockers)
            self.assertIn("kernel_removal_rejected", plan.blockers)
            self.assertIn("unrelated_upgrade_rejected", plan.blockers)
            self.assertIn("bootloader_replacement", plan.blockers)

    def test_resolved_default_safety_pin_distinguishes_text_from_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_phase21c0_boot_fixture(root)
            _add_installed_64k_fixture(root)
            inventory = collect_boot_inventory(root).to_dict()
            inventory["grub_environment"]["saved_entry"] = "Ubuntu, with Linux 6.17.0-1020-aws-64k"
            inventory["current_default_boot_entry"] = "Ubuntu, with Linux 6.17.0-1020-aws-64k"

            plan = build_grub_default_plan(inventory, safe_entry="gnulinux-6.17.0-1019-aws-advanced-root")

            self.assertEqual(plan.status, "planning_ready")
            self.assertTrue(plan.configuration_text_unchanged)
            self.assertFalse(plan.resolved_default_boot_target_unchanged)
            self.assertTrue(plan.safety_pin_required)
            self.assertFalse(plan.safety_pin_plan["apply_allowed_in_phase21c2b0"])
            self.assertEqual(plan.safety_pin_plan["safe_kernel"], "6.17.0-1019-aws")

    def test_g4_g5_g6_g7_plans_and_comparison_identities_are_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_phase21c0_boot_fixture(root)
            _add_installed_64k_fixture(root)
            inventory = collect_boot_inventory(root).to_dict()

            g4 = build_kernel_experiment_plan("G4", inventory)
            g5 = build_kernel_experiment_plan("G5", inventory)
            g6 = build_kernel_experiment_plan("G6", inventory)
            g7_blocked = build_kernel_experiment_plan("G7", inventory)
            g7_ready = build_kernel_experiment_plan(
                "G7",
                inventory,
                allow_domain_isolation_interaction=True,
                completed_profiles=["64k-default", "64k-nohz-rcu-irq", "64k-managed-irq-isolation"],
            )

            self.assertEqual(g4["eligibility_state"], "planning_ready")
            self.assertIn("nohz_full=2-7", g5["temporary_entry_plan"]["menuentry_text"])
            self.assertIn("isolcpus=managed_irq,2-7", g6["temporary_entry_plan"]["menuentry_text"])
            self.assertEqual(g7_blocked["eligibility_state"], "blocked")
            self.assertIn("domain_isolation_interaction_authorization_missing", g7_blocked["blockers"])
            self.assertTrue(g7_ready["advanced_interaction_candidate"])
            self.assertTrue(g7_ready["prior_4k_domain_isolation_regressed"])
            self.assertEqual(verify_kernel_experiment_plan(g7_ready)["status"], "passed")
            self.assertEqual(validate_phase21c2b_comparison(g4["comparison_identities"])["status"], "passed")

    def test_kernel_package_cleanup_and_cli_fake_e2e_are_plan_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            _write_phase21c0_boot_fixture(root)
            before = collect_boot_inventory(root).to_dict()
            _add_installed_64k_fixture(root)
            after = collect_boot_inventory(root).to_dict()
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            package_plan_path = Path(tmp) / "package-plan.json"
            experiment_plan_path = Path(tmp) / "experiment-plan.json"
            sim_path = Path(tmp) / "apt-sim.txt"
            atomic_write_json(before_path, before)
            atomic_write_json(after_path, after)
            sim_path.write_text("Inst linux-image-aws-64k (6.17.0-1020.20 Ubuntu:24.04 [arm64])\n", encoding="utf-8")

            cleanup = build_kernel_cleanup_plan(before, after)

            self.assertIn("linux-image-6.17.0-1020-aws-64k", cleanup.removed_packages)
            self.assertFalse(cleanup.kernel_or_boot_state_modified)

            self.assertEqual(
                _run_cli_code([
                    "system",
                    "kernel-package-plan",
                    "--inventory",
                    str(before_path),
                    "--apt-simulation",
                    str(sim_path),
                    "--output",
                    str(package_plan_path),
                    "--json",
                ]),
                0,
            )
            self.assertEqual(_run_cli_code(["system", "kernel-package-plan", "verify", "--plan", str(package_plan_path), "--json"]), 0)
            default_payload = _run_cli([
                "system",
                "grub-default-plan",
                "--inventory",
                str(after_path),
                "--safe-entry",
                "gnulinux-6.17.0-1019-aws-advanced-root",
                "--json",
            ])
            self.assertFalse(default_payload["kernel_or_boot_state_modified"])
            self.assertEqual(
                _run_cli_code([
                    "system",
                    "kernel-experiment-plan",
                    "--profile",
                    "G6",
                    "--inventory",
                    str(after_path),
                    "--output",
                    str(experiment_plan_path),
                    "--json",
                ]),
                0,
            )
            self.assertEqual(_run_cli_code(["system", "kernel-experiment-plan", "verify", "--plan", str(experiment_plan_path), "--json"]), 0)
            cleanup_payload = _run_cli(["system", "kernel-cleanup-plan", "--before", str(before_path), "--after", str(after_path), "--json"])
            self.assertEqual(cleanup_payload["kernel_or_boot_state_modified"], False)


def _add_installed_64k_fixture(root: Path) -> None:
    release = "6.17.0-1020-aws-64k"
    config = (root / "run/perfrunbench-fixtures/candidate_64k_kernel_config").read_text(encoding="utf-8")
    for relative, content in {
        f"boot/vmlinuz-{release}": "",
        f"boot/initrd.img-{release}": "",
        f"boot/config-{release}": config,
        f"boot/System.map-{release}": "",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "lib/modules" / release).mkdir(parents=True, exist_ok=True)
    (root / "run/perfrunbench-fixtures/dpkg_kernel_inventory").write_text(
        "\n".join(
            [
                "linux-image-6.17.0-1019-aws\t6.17.0-1019.19\tarm64\tii",
                "linux-image-6.17.0-1018-aws\t6.17.0-1018.18\tarm64\tii",
                f"linux-image-{release}\t6.17.0-1020.20\tarm64\tii",
                "",
            ]
        ),
        encoding="utf-8",
    )
    original = (root / "boot/grub/grub.cfg").read_text(encoding="utf-8")
    addition = "\n".join(
        [
            f"menuentry 'Ubuntu, with Linux {release}' --id 'gnulinux-{release}-advanced-root' {{",
            "  insmod gzio",
            "  search --no-floppy --fs-uuid --set=root 1111-2222",
            f"  linux /boot/vmlinuz-{release} root=LABEL=cloudimg-rootfs ro",
            f"  initrd /boot/initrd.img-{release}",
            "}",
            "",
        ]
    )
    (root / "boot/grub/grub.cfg").write_text(addition + original, encoding="utf-8")


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
