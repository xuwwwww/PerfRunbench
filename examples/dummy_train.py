from __future__ import annotations

from pathlib import Path
import time


def read_batch_size(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("batch_size:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("batch_size not found")


batch_size = read_batch_size(Path("examples/train_config.yaml"))
payload = [0] * (batch_size * 1000)
time.sleep(0.2)
print(f"dummy training completed with batch_size={batch_size}, payload={len(payload)}")

