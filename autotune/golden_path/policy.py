from __future__ import annotations


RETUNE_STATES = {"hard_drift", "expired", "missing", "corrupt"}
DRIFT_STATES = {"soft_drift", "hard_drift", "expired", "missing"}


def should_retune(state: str, policy: str) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "on_hard_drift_or_missing":
        return state in RETUNE_STATES
    raise ValueError(f"unsupported retune_policy: {policy}")


def should_canary(state: str, policy: str, *, newly_tuned: bool = False) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "on_new_or_drift":
        return newly_tuned or state in DRIFT_STATES
    raise ValueError(f"unsupported canary_policy: {policy}")

