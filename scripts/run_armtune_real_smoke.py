from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotune.arm_tune.config import load_benchmark_config
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.cli import main as autotune_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explicitly acknowledged ArmTune real-Arm smoke benchmark.")
    parser.add_argument("--config", required=True, help="ArmTune benchmark config with user-supplied binary and model paths.")
    parser.add_argument("--acknowledge-real-benchmark", action="store_true", help="Required acknowledgement for preliminary measured data.")
    args = parser.parse_args(argv)
    if not args.acknowledge_real_benchmark:
        print("Refusing to run: pass --acknowledge-real-benchmark to opt in.", file=sys.stderr)
        return 2
    config_path = Path(args.config)
    spec, _job = load_benchmark_config(config_path)
    if spec.type != "local":
        print("This scaffold only validates local real-Arm smoke runs; use the normal CLI for SSH targets.", file=sys.stderr)
        return 2
    inspection = LocalArmTarget(spec).inspect()
    if not inspection.production_validation.valid:
        print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
        print("Refusing to run: local target is not Linux Arm64.", file=sys.stderr)
        return 2
    print("Preliminary measured data: real Arm smoke run, not a public performance claim.")
    return autotune_main(["arm", "benchmark", "--config", str(config_path)])


if __name__ == "__main__":
    raise SystemExit(main())
