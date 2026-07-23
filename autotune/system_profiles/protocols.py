from __future__ import annotations

from autotune.system_profiles.schemas import BenchmarkProtocol


PROTOCOLS = {
    "steady_state_decode": BenchmarkProtocol(
        protocol_id="steady_state_decode",
        primary_metric="decode_tokens_per_second",
        secondary_metrics=["variability", "context_switches", "cpu_migrations", "page_faults", "dtlb_cache_evidence"],
        constraints=["measured provenance", "sufficient samples", "no parser failures"],
        cannot_combine_with=["cold_model_load", "warm_first_request", "interactive_server", "concurrent_server"],
    ),
    "cold_model_load": BenchmarkProtocol(
        protocol_id="cold_model_load",
        primary_metric="model_load_time",
        secondary_metrics=["major_faults", "minor_faults", "bytes_read", "peak_rss"],
        constraints=["cold-cache protocol label", "no ranking by warm decode"],
        cannot_combine_with=["steady_state_decode", "warm_first_request", "interactive_server", "concurrent_server"],
    ),
    "warm_first_request": BenchmarkProtocol(
        protocol_id="warm_first_request",
        primary_metric="ttft_ms",
        secondary_metrics=["end_to_end_latency_ms", "faults", "rss"],
        constraints=["warm model protocol label", "request completion required"],
        cannot_combine_with=["steady_state_decode", "cold_model_load", "interactive_server", "concurrent_server"],
    ),
    "interactive_server": BenchmarkProtocol(
        protocol_id="interactive_server",
        primary_metric="ttft_median_and_p95",
        secondary_metrics=["end_to_end_latency_median_and_p95", "stream_parse_failures"],
        constraints=["zero request failures", "cleanup passed"],
        cannot_combine_with=["steady_state_decode", "cold_model_load", "warm_first_request", "concurrent_server"],
    ),
    "concurrent_server": BenchmarkProtocol(
        protocol_id="concurrent_server",
        primary_metric="aggregate_output_tokens_per_second",
        secondary_metrics=["p95_latency", "failed_requests", "memory_headroom", "quality_guard_status"],
        constraints=["zero failures", "memory headroom", "quality guard when configured"],
        cannot_combine_with=["steady_state_decode", "cold_model_load", "warm_first_request", "interactive_server"],
    ),
}


def list_protocols() -> list[dict[str, object]]:
    return [PROTOCOLS[key].to_dict() for key in sorted(PROTOCOLS)]
