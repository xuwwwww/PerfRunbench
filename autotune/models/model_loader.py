from __future__ import annotations


def load_model_metadata(config: dict) -> dict:
    return config.get("model", {})

