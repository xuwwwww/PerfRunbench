from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from autotune.arm_tune.config import parse_target_spec
from autotune.arm_tune.errors import TargetExecutionError
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmCpuIdentity,
    ArmHostCapabilities,
    CommandResult,
    CpuTopology,
    HostInfo,
    HostInspectionResult,
    ProductionValidation,
    RemotePrerequisiteResult,
    TargetSpec,
    WorkerResult,
)
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.arm_tune.targets.ssh import SshArmTarget


class FakeRunner:
    def __init__(self, stdout: str, *, stdout_call: int = 3) -> None:
        self.stdout = stdout
        self.stdout_call = stdout_call
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        self.commands.append(command)
        if len(self.commands) == self.stdout_call:
            return CommandResult(command=command, return_code=0, stdout=self.stdout)
        return CommandResult(command=command, return_code=0)


class MissingModuleRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        self.commands.append(command)
        if "-m" in command:
            return CommandResult(command=command, return_code=1, stderr="No module named autotune")
        return CommandResult(command=command, return_code=0)


class ArmTuneTargetsTest(unittest.TestCase):
    def test_local_target_validation_uses_inspection_result(self) -> None:
        inspection = _inspection(valid=False, system="Windows", machine="amd64")
        with patch("autotune.arm_tune.targets.local.collect_host_inspection", return_value=inspection):
            validation = LocalArmTarget(TargetSpec(type="local")).validate_production_target()

        self.assertFalse(validation.valid)
        self.assertEqual(validation.observed_system, "Windows")

    def test_ssh_worker_command_uses_strict_host_key_checking_by_default(self) -> None:
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh", "host": "arm-host", "user": "ubuntu"}})
        target = SshArmTarget(spec)

        command = target.build_worker_command("~/armtune-runs/job.json", "~/armtune-runs/run")

        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("ubuntu@arm-host", command)
        self.assertIn("python3", command)
        self.assertIn("-m", command)
        self.assertIn("autotune.cli", command)
        self.assertIn("--job-spec", command)

    def test_ssh_inspect_uses_fixed_worker_command_with_fake_runner(self) -> None:
        payload = WorkerResult(operation="inspect_host", status="ok", result=_inspection().to_dict()).to_dict()
        runner = FakeRunner(json.dumps(payload))
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh", "host": "arm-host"}})
        target = SshArmTarget(spec, runner=runner)

        result = target.inspect()

        self.assertEqual(result.schema_version, "1.0")
        self.assertEqual(len(runner.commands), 3)
        self.assertEqual(runner.commands[0][-3:], ["mkdir", "-p", "~/armtune-runs/inspect-host"])
        self.assertIn("scp", runner.commands[1][0])
        self.assertIn("worker", runner.commands[2])
        self.assertIn("-m", runner.commands[2])
        self.assertIn("autotune.cli", runner.commands[2])

    def test_ssh_prerequisites_use_structured_python_module_invocation(self) -> None:
        payload = WorkerResult(
            operation="check_prerequisites",
            status="ok",
            result=_remote_prerequisites().to_dict(),
        ).to_dict()
        runner = FakeRunner(json.dumps(payload), stdout_call=4)
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh", "host": "arm-host"}})
        target = SshArmTarget(spec, runner=runner)

        result = target.check_prerequisites()

        self.assertTrue(result.python_available)
        self.assertTrue(result.module_importable)
        self.assertEqual(len(runner.commands), 4)
        self.assertEqual(runner.commands[0][-2:], ["python3", "--version"])
        self.assertEqual(
            runner.commands[3][-9:],
            [
                "python3",
                "-m",
                "autotune.cli",
                "arm",
                "worker",
                "--job-spec",
                "~/armtune-runs/prerequisites/job.json",
                "--run-dir",
                "~/armtune-runs/prerequisites",
            ],
        )

    def test_ssh_prerequisites_report_missing_remote_module(self) -> None:
        spec = parse_target_spec({"schema_version": "1.0", "target": {"type": "ssh", "host": "arm-host"}})
        target = SshArmTarget(spec, runner=MissingModuleRunner())

        with self.assertRaises(TargetExecutionError) as error:
            target.check_prerequisites()

        self.assertIn("remote module prerequisite failed", str(error.exception))

    def test_ssh_redacted_command_hides_identity_file_path_on_failure(self) -> None:
        spec = parse_target_spec(
            {
                "schema_version": "1.0",
                "target": {"type": "ssh", "host": "arm-host", "identity_file": "~/.ssh/id_ed25519"},
            }
        )
        target = SshArmTarget(spec)

        text = target._redacted_command(target.build_worker_command("~/job.json", "~/run"))

        self.assertNotIn("id_ed25519", text)
        self.assertIn("<identity-file>", text)

    def test_ssh_serialized_job_does_not_include_identity_file_path(self) -> None:
        payload = WorkerResult(operation="inspect_host", status="ok", result=_inspection().to_dict()).to_dict()
        runner = FakeRunner(json.dumps(payload))
        spec = parse_target_spec(
            {
                "schema_version": "1.0",
                "target": {"type": "ssh", "host": "arm-host", "identity_file": "~/.ssh/id_ed25519"},
            }
        )
        target = SshArmTarget(spec, runner=runner)
        written_payloads: list[dict[str, object]] = []

        def capture_json(path: object, payload: dict[str, object]) -> object:
            written_payloads.append(payload)
            return path

        with patch("autotune.arm_tune.targets.ssh.atomic_write_json", side_effect=capture_json):
            target.inspect()

        serialized = json.dumps(written_payloads, sort_keys=True)
        self.assertNotIn("id_ed25519", serialized)
        self.assertNotIn(".ssh", serialized)


def _inspection(*, valid: bool = True, system: str = "Linux", machine: str = "aarch64") -> HostInspectionResult:
    host = HostInfo(
        architecture=ArchitectureInfo(
            system=system,
            machine=machine,
            normalized_machine=machine,
            kernel_release="6.8.0",
            is_linux=system == "Linux",
            is_arm64=machine == "aarch64",
        ),
        cpu_identity=ArmCpuIdentity(model_name="Neoverse-V2", source="fixture"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="fixture"),
        topology=CpuTopology(logical_cpus=1, source="fixture"),
    )
    validation = ProductionValidation(
        valid=valid,
        reason="fixture",
        observed_system=system,
        observed_machine=machine,
    )
    return HostInspectionResult(target_type="fixture", host=host, production_validation=validation)


def _remote_prerequisites() -> RemotePrerequisiteResult:
    inspection = _inspection()
    return RemotePrerequisiteResult(
        python_executable="/usr/bin/python3",
        python_available=True,
        module="autotune.cli",
        module_importable=True,
        worker_protocol_compatible=True,
        remote_workdir_writable=True,
        linux=True,
        arm64=True,
        host=inspection.host,
        production_validation=inspection.production_validation,
    )


if __name__ == "__main__":
    unittest.main()
