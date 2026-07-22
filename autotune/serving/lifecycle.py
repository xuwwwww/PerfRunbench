from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune.schemas import to_plain
from autotune.serving import SERVING_SCHEMA_VERSION


@dataclass(frozen=True)
class ServingLifecyclePreflight:
    port: int
    port_free: bool
    stale_process_detected: bool
    can_start: bool
    issues: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def check_server_lifecycle_preflight(
    *,
    host: str = "127.0.0.1",
    port: int,
    process_listing: str = "",
) -> ServingLifecyclePreflight:
    port_free = _port_free(host, port)
    stale = "llama-server" in process_listing or "fake_llama_server" in process_listing
    issues: list[str] = []
    if not port_free:
        issues.append("unknown_listener_occupies_port")
    if stale:
        issues.append("stale_llama_server_process_detected")
    return ServingLifecyclePreflight(
        port=port,
        port_free=port_free,
        stale_process_detected=stale,
        can_start=port_free and not stale,
        issues=issues,
    )


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False
