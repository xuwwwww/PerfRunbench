from __future__ import annotations

import json
import os
import time


threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
delay_seconds = 0.30 if threads == 1 else 0.06
time.sleep(delay_seconds)

print(
    json.dumps(
        {
            "schema_version": "1.0",
            "result_origin": "test_fixture",
            "fixture_name": "submission_cli_demo",
            "completed": True,
        },
        sort_keys=True,
    )
)
