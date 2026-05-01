from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.executor_capabilities import collect_executor_capabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local executor capabilities for resource guarding.")
    parser.add_argument(
        "--probe-docker",
        action="store_true",
        help="Run docker info to check whether the Docker daemon is reachable.",
    )
    parser.add_argument(
        "--probe-systemd",
        action="store_true",
        help="Run a harmless systemd transient scope probe to detect whether sudo is required.",
    )
    parser.add_argument(
        "--check-sudo-cache",
        action="store_true",
        help="Check whether sudo credentials are cached without prompting for a password.",
    )
    args = parser.parse_args()

    capabilities = collect_executor_capabilities(
        probe_docker=args.probe_docker,
        probe_systemd=args.probe_systemd,
        check_sudo_cache=args.check_sudo_cache,
    )
    print(json.dumps(capabilities, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
