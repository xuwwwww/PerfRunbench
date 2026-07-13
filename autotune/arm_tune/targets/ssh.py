from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from autotune.arm_tune.benchmark.llama_bench import ensure_benchmark_operation
from autotune.arm_tune.errors import TargetExecutionError
from autotune.arm_tune.schemas import (
    ArmBenchmarkResult,
    CommandResult,
    HostInspectionResult,
    LlamaBenchJobSpec,
    ProductionValidation,
    RemotePrerequisiteResult,
    TargetSpec,
    WorkerJobSpec,
)
from autotune.arm_tune.targets.base import ArmTuneTarget, CommandRunner
from autotune.arm_tune.targets.protocol import atomic_write_json, safe_remote_path, validate_remote_identity, validate_remote_module


class SubprocessCommandRunner:
    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(command=command, return_code=124, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
        except OSError as exc:
            return CommandResult(command=command, return_code=127, stderr=str(exc))
        return CommandResult(command=command, return_code=result.returncode, stdout=result.stdout or "", stderr=result.stderr or "")


class SshArmTarget(ArmTuneTarget):
    def __init__(self, spec: TargetSpec, *, runner: CommandRunner | None = None) -> None:
        super().__init__(spec)
        self.runner = runner or SubprocessCommandRunner()

    def inspect(self) -> HostInspectionResult:
        with tempfile.TemporaryDirectory(prefix="armtune-ssh-") as temp_dir:
            remote_job, remote_run_dir = self._upload_job(
                temp_dir=temp_dir,
                operation="inspect_host",
                run_label="inspect-host",
            )
            result = self._run_checked(
                self.build_worker_command(remote_job, remote_run_dir),
                timeout_seconds=self.spec.command_timeout_seconds,
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TargetExecutionError("remote worker did not return valid JSON") from exc
        worker_payload = payload.get("result", payload)
        return HostInspectionResult.from_dict(worker_payload)

    def validate_production_target(self) -> ProductionValidation:
        return self.inspect().production_validation

    def run_benchmark(self, job: LlamaBenchJobSpec, local_artifact_dir: str | Path) -> ArmBenchmarkResult:
        local_dir = Path(local_artifact_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_run_dir = safe_remote_path(f"{self.spec.remote_workdir.rstrip('/')}/{job.run_id}/armtune/benchmark")
        remote_job = safe_remote_path(f"{remote_run_dir}/benchmark_job.json")
        remote_job_spec = replace(
            job,
            target_type="ssh",
            target_run_directory=remote_run_dir,
            remote_module=self.spec.remote_module,
        )
        ensure_benchmark_operation(remote_job_spec)
        with tempfile.TemporaryDirectory(prefix="armtune-ssh-benchmark-") as temp_dir:
            local_job = Path(temp_dir) / "benchmark_job.json"
            atomic_write_json(local_job, remote_job_spec.to_dict())
            self._run_checked(self.build_ssh_command(["mkdir", "-p", remote_run_dir]), timeout_seconds=self.spec.connect_timeout_seconds)
            self._run_checked(self.build_scp_upload_command(local_job, remote_job), timeout_seconds=self.spec.connect_timeout_seconds)
            result = self._run_checked(
                self.build_worker_command(remote_job, remote_run_dir),
                timeout_seconds=self.spec.command_timeout_seconds,
            )
            self._run_checked(
                self.build_scp_download_command(f"{remote_run_dir}/.", local_dir),
                timeout_seconds=self.spec.command_timeout_seconds,
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TargetExecutionError("remote benchmark worker did not return valid JSON") from exc
        if payload.get("status") != "ok":
            raise TargetExecutionError(f"remote benchmark worker failed: {payload.get('error')}")
        return ArmBenchmarkResult.from_dict(payload["result"])

    def check_prerequisites(self) -> RemotePrerequisiteResult:
        self._run_checked(self.build_python_version_command(), timeout_seconds=self.spec.connect_timeout_seconds)
        with tempfile.TemporaryDirectory(prefix="armtune-ssh-") as temp_dir:
            remote_job, remote_run_dir = self._upload_job(
                temp_dir=temp_dir,
                operation="check_prerequisites",
                run_label="prerequisites",
            )
            try:
                result = self._run_checked(
                    self.build_worker_command(remote_job, remote_run_dir),
                    timeout_seconds=self.spec.command_timeout_seconds,
                )
            except TargetExecutionError as exc:
                raise TargetExecutionError(
                    f"remote module prerequisite failed for {self.spec.remote_module!r}: {exc}"
                ) from exc
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TargetExecutionError("remote prerequisite worker did not return valid JSON") from exc
        if payload.get("status") != "ok":
            raise TargetExecutionError(f"remote prerequisite worker failed: {payload.get('error')}")
        return RemotePrerequisiteResult.from_dict(payload["result"])

    def build_python_version_command(self) -> list[str]:
        return self.build_ssh_command([safe_remote_path(self.spec.remote_python), "--version"])

    def build_worker_command(self, remote_job_spec: str, remote_run_dir: str) -> list[str]:
        return self.build_ssh_command(
            [
                safe_remote_path(self.spec.remote_python),
                "-m",
                validate_remote_module(self.spec.remote_module),
                "arm",
                "worker",
                "--job-spec",
                safe_remote_path(remote_job_spec),
                "--run-dir",
                safe_remote_path(remote_run_dir),
            ]
        )

    def build_ssh_command(self, remote_argv: list[str]) -> list[str]:
        destination = self._destination()
        command = [
            self.spec.ssh_executable,
            "-o",
            "BatchMode=yes",
            "-o",
            f"StrictHostKeyChecking={'yes' if self.spec.strict_host_key_checking else 'no'}",
            "-o",
            f"ConnectTimeout={self.spec.connect_timeout_seconds}",
            "-p",
            str(self.spec.port),
        ]
        if self.spec.identity_file:
            command.extend(["-i", self.spec.identity_file])
        command.append(destination)
        command.extend(remote_argv)
        return command

    def _upload_job(self, *, temp_dir: str, operation: str, run_label: str) -> tuple[str, str]:
        job = WorkerJobSpec(operation=operation, target_type="ssh", remote_module=self.spec.remote_module)
        local_job = Path(temp_dir) / "job.json"
        atomic_write_json(local_job, job.to_dict())
        remote_run_dir = safe_remote_path(f"{self.spec.remote_workdir.rstrip('/')}/{run_label}")
        remote_job = safe_remote_path(f"{remote_run_dir}/job.json")
        self._run_checked(self.build_ssh_command(["mkdir", "-p", remote_run_dir]), timeout_seconds=self.spec.connect_timeout_seconds)
        self._run_checked(self.build_scp_upload_command(local_job, remote_job), timeout_seconds=self.spec.connect_timeout_seconds)
        return remote_job, remote_run_dir

    def build_scp_upload_command(self, local_path: str | Path, remote_path: str) -> list[str]:
        destination = f"{self._destination()}:{safe_remote_path(remote_path)}"
        command = [
            self.spec.scp_executable,
            "-o",
            "BatchMode=yes",
            "-o",
            f"StrictHostKeyChecking={'yes' if self.spec.strict_host_key_checking else 'no'}",
            "-o",
            f"ConnectTimeout={self.spec.connect_timeout_seconds}",
            "-P",
            str(self.spec.port),
        ]
        if self.spec.identity_file:
            command.extend(["-i", self.spec.identity_file])
        command.extend([str(local_path), destination])
        return command

    def build_scp_download_command(self, remote_path: str, local_path: str | Path) -> list[str]:
        source = f"{self._destination()}:{safe_remote_path(remote_path)}"
        command = [
            self.spec.scp_executable,
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            f"StrictHostKeyChecking={'yes' if self.spec.strict_host_key_checking else 'no'}",
            "-o",
            f"ConnectTimeout={self.spec.connect_timeout_seconds}",
            "-P",
            str(self.spec.port),
        ]
        if self.spec.identity_file:
            command.extend(["-i", self.spec.identity_file])
        command.extend([source, str(local_path)])
        return command

    def _destination(self) -> str:
        if not self.spec.host:
            raise TargetExecutionError("ssh target requires host")
        host = validate_remote_identity(self.spec.host, label="target.host")
        if self.spec.user:
            user = validate_remote_identity(self.spec.user, label="target.user")
            return f"{user}@{host}"
        return host

    def _run_checked(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        result = self.runner.run(command, timeout_seconds=timeout_seconds)
        if result.return_code != 0:
            raise TargetExecutionError(f"target command failed with exit code {result.return_code}: {self._redacted_command(command)}")
        return result

    def _redacted_command(self, command: list[str]) -> str:
        redacted: list[str] = []
        skip_next = False
        for part in command:
            if skip_next:
                redacted.append("<identity-file>")
                skip_next = False
                continue
            redacted.append(part)
            if part == "-i":
                skip_next = True
        return " ".join(redacted)
