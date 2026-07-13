from __future__ import annotations

import re

from autotune.arm_tune.schemas import LlamaBenchParseResult, ParsedMetric


METRIC_PATTERN = re.compile(r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
TEST_TOKEN_PATTERN = re.compile(r"(?P<kind>pp|tg|prompt|prefill|decode|gen|generation)[^0-9]*(?P<tokens>\d+)?", re.IGNORECASE)


def parse_llama_bench_output(text: str) -> LlamaBenchParseResult:
    lines = text.splitlines()
    pipe_result = _parse_pipe_table(lines)
    if pipe_result.metrics or pipe_result.parse_status == "malformed_output":
        return pipe_result
    if pipe_result.parse_status == "unsupported_format":
        return pipe_result
    whitespace_result = _parse_whitespace_table(lines)
    if whitespace_result.metrics or whitespace_result.parse_status == "malformed_output":
        return whitespace_result
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return LlamaBenchParseResult(parse_status="no_metrics", parser_strategy="none", message="output was empty")
    return LlamaBenchParseResult(
        parse_status="no_metrics",
        parser_strategy="none",
        unparsed_lines=non_empty[:20],
        message="no llama-bench metrics were found",
    )


def required_metric_values(result: LlamaBenchParseResult) -> tuple[float | None, float | None]:
    prefill = _first_metric(result, "prefill_tokens_per_second")
    decode = _first_metric(result, "decode_tokens_per_second")
    return prefill, decode


def _first_metric(result: LlamaBenchParseResult, name: str) -> float | None:
    for metric in result.metrics:
        if metric.name == name:
            return metric.value
    return None


def _parse_pipe_table(lines: list[str]) -> LlamaBenchParseResult:
    header: list[str] | None = None
    metrics: list[ParsedMetric] = []
    unparsed: list[str] = []
    malformed = False
    saw_pipe = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            if stripped:
                unparsed.append(stripped)
            continue
        saw_pipe = True
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if any("test" == cell or cell.endswith(" test") for cell in lowered) and any("t/s" in cell or "tok/s" in cell for cell in lowered):
            header = lowered
            continue
        if header is None or _is_separator_row(cells):
            continue
        row = {header[index]: cells[index] for index in range(min(len(header), len(cells)))}
        test = _find_column(row, "test")
        throughput_text = _find_throughput(row)
        if not test or not throughput_text:
            unparsed.append(stripped)
            continue
        metric_name = _metric_name_from_test(test)
        if metric_name is None:
            unparsed.append(stripped)
            continue
        value = _metric_value(throughput_text)
        if value is None:
            malformed = True
            unparsed.append(stripped)
            continue
        metrics.append(_metric(metric_name, value, "pipe_table", stripped))
    if saw_pipe and header is None:
        return LlamaBenchParseResult(
            parse_status="unsupported_format",
            metrics=[],
            parser_strategy="pipe_table",
            unparsed_lines=unparsed[:20],
            message="pipe table did not contain supported llama-bench metric columns",
        )
    return _result_from_metrics(metrics, "pipe_table", unparsed, malformed)


def _parse_whitespace_table(lines: list[str]) -> LlamaBenchParseResult:
    metrics: list[ParsedMetric] = []
    unparsed: list[str] = []
    malformed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("test") or lower.startswith("---") or lower.startswith("#") or "warning" in lower:
            unparsed.append(stripped)
            continue
        parts = stripped.split()
        if not parts:
            continue
        metric_name = _metric_name_from_test(parts[0])
        if metric_name is None:
            unparsed.append(stripped)
            continue
        value = _metric_value(" ".join(parts[1:]))
        if value is None:
            malformed = True
            unparsed.append(stripped)
            continue
        metrics.append(_metric(metric_name, value, "whitespace_table", stripped))
    return _result_from_metrics(metrics, "whitespace_table", unparsed, malformed)


def _result_from_metrics(
    metrics: list[ParsedMetric],
    strategy: str,
    unparsed: list[str],
    malformed: bool,
) -> LlamaBenchParseResult:
    if malformed and not metrics:
        return LlamaBenchParseResult(
            parse_status="malformed_output",
            metrics=[],
            parser_strategy=strategy,
            unparsed_lines=unparsed[:20],
            message="metric-looking output contained malformed numeric values",
        )
    if not metrics:
        return LlamaBenchParseResult(
            parse_status="no_metrics",
            metrics=[],
            parser_strategy=strategy,
            unparsed_lines=unparsed[:20],
            message="no metrics parsed",
        )
    names = {metric.name for metric in metrics}
    status = "parsed" if {"prefill_tokens_per_second", "decode_tokens_per_second"}.issubset(names) and not malformed else "partially_parsed"
    if malformed:
        status = "malformed_output" if not metrics else "partially_parsed"
    return LlamaBenchParseResult(
        parse_status=status,
        metrics=metrics,
        parser_strategy=strategy,
        unparsed_lines=unparsed[:20],
        message="metrics parsed" if status == "parsed" else "only some required metrics were parsed",
    )


def _metric(name: str, value: float, strategy: str, raw_line: str) -> ParsedMetric:
    return ParsedMetric(
        name=name,
        value=value,
        unit="tokens/s",
        source="stdout",
        parser_strategy=strategy,
        raw_source_line=raw_line,
    )


def _metric_name_from_test(test: str) -> str | None:
    match = TEST_TOKEN_PATTERN.search(test)
    if not match:
        return None
    kind = match.group("kind").lower()
    if kind in {"pp", "prompt", "prefill"}:
        return "prefill_tokens_per_second"
    if kind in {"tg", "decode", "gen", "generation"}:
        return "decode_tokens_per_second"
    return None


def _metric_value(text: str) -> float | None:
    match = METRIC_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group("value"))
    except ValueError:
        return None


def _find_column(row: dict[str, str], suffix: str) -> str:
    for key, value in row.items():
        if key == suffix or key.endswith(f" {suffix}"):
            return value
    return ""


def _find_throughput(row: dict[str, str]) -> str:
    for key, value in row.items():
        if "t/s" in key or "tok/s" in key or "tokens/s" in key:
            return value
    return ""


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells if cell.strip())
