from __future__ import annotations

import math
import re
from collections import Counter

from autotune.arm_tune.schemas import LlamaBenchParseResult, ParsedMetric


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_PATTERN = re.compile(r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
BAD_VALUE_PATTERN = re.compile(r"(?i)\b(?:nan|inf|-inf|\+inf)\b|\d+,\d+")
TEST_TOKEN_PATTERN = re.compile(r"(?P<kind>pp|tg|prompt|prefill|decode|gen|generation)[^0-9]*(?P<tokens>\d+)?", re.IGNORECASE)


def parse_llama_bench_output(text: str) -> LlamaBenchParseResult:
    return _parse_stream(text, source="stdout")


def parse_llama_bench_streams(stdout: str, stderr: str) -> LlamaBenchParseResult:
    stdout_result = _parse_stream(stdout, source="stdout")
    if stdout_result.metrics or stdout_result.parse_status in {"malformed_output", "ambiguous_output"}:
        return stdout_result
    stderr_result = _parse_stream(stderr, source="stderr")
    if stderr_result.metrics or stderr_result.parse_status in {"malformed_output", "ambiguous_output"}:
        return stderr_result
    if stdout_result.parse_status == "unsupported_format":
        return stdout_result
    return stdout_result


def required_metric_values(result: LlamaBenchParseResult) -> tuple[float | None, float | None]:
    if result.parse_status == "ambiguous_output":
        return None, None
    prefill = _single_metric(result, "prefill_tokens_per_second")
    decode = _single_metric(result, "decode_tokens_per_second")
    return prefill, decode


def _single_metric(result: LlamaBenchParseResult, name: str) -> float | None:
    values = [metric.value for metric in result.metrics if metric.name == name and metric.validation_status == "passed"]
    return values[0] if len(values) == 1 else None


def _parse_stream(text: str, *, source: str) -> LlamaBenchParseResult:
    lines = text.splitlines()
    pipe_result = _parse_pipe_table(lines, source=source)
    if pipe_result.metrics or pipe_result.parse_status in {"malformed_output", "ambiguous_output"}:
        return pipe_result
    if pipe_result.parse_status == "unsupported_format":
        return pipe_result
    whitespace_result = _parse_whitespace_table(lines, source=source)
    if whitespace_result.metrics or whitespace_result.parse_status in {"malformed_output", "ambiguous_output"}:
        return whitespace_result
    non_empty = [_clean_line(line).strip() for line in lines if _clean_line(line).strip()]
    if not non_empty:
        return LlamaBenchParseResult(parse_status="no_metrics", parser_strategy="none", message=f"{source} was empty")
    return LlamaBenchParseResult(
        parse_status="no_metrics",
        parser_strategy="none",
        unparsed_lines=non_empty[:20],
        message=f"no llama-bench metrics were found in {source}",
    )


def _parse_pipe_table(lines: list[str], *, source: str) -> LlamaBenchParseResult:
    header: list[str] | None = None
    metrics: list[ParsedMetric] = []
    unparsed: list[str] = []
    malformed = False
    saw_pipe = False
    for line_number, line in enumerate(lines, start=1):
        stripped = _clean_line(line).strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            if stripped:
                unparsed.append(stripped)
            continue
        saw_pipe = True
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if any(cell == "test" or cell.endswith(" test") for cell in lowered) and any(_is_throughput_header(cell) for cell in lowered):
            header = lowered
            continue
        if header is None or _is_separator_row(cells):
            continue
        row = {header[index]: cells[index] for index in range(min(len(header), len(cells)))}
        metric = _metric_from_row(
            test=_find_column(row, "test"),
            throughput_text=_find_throughput(row),
            strategy="pipe_table",
            source=source,
            raw_line=stripped,
            line_number=line_number,
        )
        if metric is None:
            if _looks_like_metric_row(row):
                malformed = True
            unparsed.append(stripped)
            continue
        metrics.append(metric)
    if saw_pipe and header is None:
        return LlamaBenchParseResult(
            parse_status="unsupported_format",
            metrics=[],
            parser_strategy="pipe_table",
            unparsed_lines=unparsed[:20],
            message="pipe table did not contain supported llama-bench metric columns",
        )
    return _result_from_metrics(metrics, "pipe_table", unparsed, malformed)


def _parse_whitespace_table(lines: list[str], *, source: str) -> LlamaBenchParseResult:
    header: list[str] | None = None
    metrics: list[ParsedMetric] = []
    unparsed: list[str] = []
    malformed = False
    for line_number, line in enumerate(lines, start=1):
        stripped = _clean_line(line).strip()
        if not stripped:
            continue
        lower = stripped.lower()
        parts = stripped.split()
        lowered = [part.lower() for part in parts]
        if _is_warning_or_comment(lower):
            unparsed.append(stripped)
            continue
        if "test" in lowered and any(_is_throughput_header(part) for part in lowered):
            header = lowered
            unparsed.append(stripped)
            continue
        if lower.startswith("---"):
            unparsed.append(stripped)
            continue
        if header:
            metric = _metric_from_header_parts(header, parts, source=source, raw_line=stripped, line_number=line_number)
        else:
            metric = _metric_from_row(
                test=parts[0] if parts else "",
                throughput_text=" ".join(parts[1:]),
                strategy="whitespace_table",
                source=source,
                raw_line=stripped,
                line_number=line_number,
            )
        if metric is None:
            if parts and _metric_name_from_test(parts[0]):
                malformed = True
            unparsed.append(stripped)
            continue
        metrics.append(metric)
    return _result_from_metrics(metrics, "whitespace_table", unparsed, malformed)


def _metric_from_header_parts(
    header: list[str],
    parts: list[str],
    *,
    source: str,
    raw_line: str,
    line_number: int,
) -> ParsedMetric | None:
    if len(parts) < len(header):
        return None
    row = {header[index]: parts[index] for index in range(min(len(header), len(parts)))}
    return _metric_from_row(
        test=_find_column(row, "test"),
        throughput_text=_find_throughput(row),
        strategy="whitespace_table",
        source=source,
        raw_line=raw_line,
        line_number=line_number,
    )


def _metric_from_row(
    *,
    test: str,
    throughput_text: str,
    strategy: str,
    source: str,
    raw_line: str,
    line_number: int,
) -> ParsedMetric | None:
    if not test or not throughput_text:
        return None
    metric_name = _metric_name_from_test(test)
    if metric_name is None:
        return None
    value = _metric_value(throughput_text)
    if value is None:
        return None
    return _metric(metric_name, value, strategy, source, raw_line, line_number)


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
    counts = Counter(metric.name for metric in metrics)
    if any(count > 1 for count in counts.values()):
        return LlamaBenchParseResult(
            parse_status="ambiguous_output",
            metrics=metrics,
            parser_strategy=strategy,
            unparsed_lines=unparsed[:20],
            message="multiple rows matched the same required metric; no fastest row was selected",
        )
    names = set(counts)
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


def _metric(name: str, value: float, strategy: str, source: str, raw_line: str, line_number: int) -> ParsedMetric:
    return ParsedMetric(
        name=name,
        value=value,
        unit="tokens/s",
        source=source,
        parser_strategy=strategy,
        raw_source_line=raw_line,
        raw_line_reference=line_number,
        validation_status="passed",
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
    if BAD_VALUE_PATTERN.search(text):
        return None
    match = METRIC_PATTERN.search(text)
    if not match:
        return None
    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    if value < 0 or math.isnan(value) or math.isinf(value):
        return None
    return value


def _find_column(row: dict[str, str], suffix: str) -> str:
    for key, value in row.items():
        if key == suffix or key.endswith(f" {suffix}"):
            return value
    return ""


def _find_throughput(row: dict[str, str]) -> str:
    for key, value in row.items():
        if _is_throughput_header(key):
            return value
    return ""


def _looks_like_metric_row(row: dict[str, str]) -> bool:
    return bool(_find_column(row, "test") and _metric_name_from_test(_find_column(row, "test")))


def _is_throughput_header(text: str) -> bool:
    return "t/s" in text or "tok/s" in text or "tokens/s" in text


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells if cell.strip())


def _is_warning_or_comment(text: str) -> bool:
    return text.startswith("#") or "warning" in text


def _clean_line(line: str) -> str:
    return ANSI_PATTERN.sub("", line)
