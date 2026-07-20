from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RuntimeAdapter(ABC):
    """Narrow runtime boundary used by ArmTune benchmark modes."""

    name: str

    @abstractmethod
    def inspect_runtime(self, job: Any, run_dir: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate_configuration(self, job: Any, inspection: Any | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, job: Any, run_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, job: Any, run_dir: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def collect_artifacts(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def parse_result(self, run_dir: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, run_dir: Path) -> Any:
        raise NotImplementedError


class ServerRuntimeAdapter(RuntimeAdapter):
    @abstractmethod
    def start_server(self, job: Any, run_dir: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def wait_until_ready(self, server: Any, job: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run_request(self, server: Any, request: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def stop_server(self, server: Any, job: Any) -> Any:
        raise NotImplementedError
