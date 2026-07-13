from __future__ import annotations

import unittest
from pathlib import Path

from autotune.arm_tune.benchmark.parser import parse_llama_bench_output, parse_llama_bench_streams, required_metric_values


FIXTURES = Path(__file__).parent / "fixtures" / "arm_tune" / "llama_bench_outputs"


class LlamaBenchParserTest(unittest.TestCase):
    def test_parses_current_pipe_table_fixture(self) -> None:
        result = parse_llama_bench_output(_fixture("current_table.txt"))

        prefill, decode = required_metric_values(result)
        self.assertEqual(result.parse_status, "parsed")
        self.assertEqual(result.parser_strategy, "pipe_table")
        self.assertEqual(prefill, 321.50)
        self.assertEqual(decode, 87.25)
        self.assertEqual(result.metrics[0].unit, "tokens/s")
        self.assertEqual(result.metrics[0].source, "stdout")
        self.assertEqual(result.metrics[0].validation_status, "passed")
        self.assertIsNotNone(result.metrics[0].raw_line_reference)
        self.assertIn("pp128", result.metrics[0].raw_source_line)

    def test_parses_alternate_whitespace_fixture(self) -> None:
        result = parse_llama_bench_output(_fixture("alternate_whitespace.txt"))

        prefill, decode = required_metric_values(result)
        self.assertEqual(result.parse_status, "parsed")
        self.assertEqual(result.parser_strategy, "whitespace_table")
        self.assertEqual(prefill, 300.25)
        self.assertEqual(decode, 75.50)

    def test_parses_variable_whitespace_and_unknown_columns(self) -> None:
        result = parse_llama_bench_output(_fixture("variable_whitespace_unknown_columns.txt"))

        self.assertEqual(result.parse_status, "parsed")
        self.assertEqual(len(result.metrics), 2)

    def test_malformed_numeric_does_not_become_zero(self) -> None:
        result = parse_llama_bench_output(_fixture("malformed_numeric.txt"))

        self.assertEqual(result.parse_status, "malformed_output")
        self.assertEqual(result.metrics, [])

    def test_successful_output_with_no_metrics_is_not_parsed(self) -> None:
        result = parse_llama_bench_output(_fixture("no_metrics.txt"))

        self.assertEqual(result.parse_status, "no_metrics")
        self.assertEqual(result.metrics, [])

    def test_partial_table_reports_partial_parse(self) -> None:
        result = parse_llama_bench_output(_fixture("partial_table.txt"))

        prefill, decode = required_metric_values(result)
        self.assertEqual(result.parse_status, "partially_parsed")
        self.assertEqual(prefill, 321.50)
        self.assertIsNone(decode)

    def test_warning_lines_before_table_are_preserved_as_unparsed(self) -> None:
        result = parse_llama_bench_output(_fixture("warnings_before_table.txt"))

        self.assertEqual(result.parse_status, "parsed")
        self.assertTrue(any("warning" in line for line in result.unparsed_lines))

    def test_unsupported_table_reports_unsupported_format(self) -> None:
        result = parse_llama_bench_output(_fixture("unsupported_table.txt"))

        self.assertEqual(result.parse_status, "unsupported_format")
        self.assertEqual(result.metrics, [])

    def test_crlf_output_parses_like_lf(self) -> None:
        result = parse_llama_bench_output(_fixture("current_table.txt").replace("\n", "\r\n"))

        self.assertEqual(result.parse_status, "parsed")

    def test_tab_separated_columns_parse(self) -> None:
        result = parse_llama_bench_output(_fixture("tab_separated.txt"))

        prefill, decode = required_metric_values(result)
        self.assertEqual(result.parse_status, "parsed")
        self.assertEqual(prefill, 301.0)
        self.assertEqual(decode, 76.0)

    def test_reordered_columns_ansi_and_warning_lines_parse(self) -> None:
        result = parse_llama_bench_output(_fixture("reordered_columns_ansi_warnings.txt"))

        self.assertEqual(result.parse_status, "parsed")
        self.assertTrue(any("warning" in line for line in result.unparsed_lines))

    def test_duplicate_required_rows_are_ambiguous(self) -> None:
        result = parse_llama_bench_output(_fixture("duplicate_rows.txt"))

        prefill, decode = required_metric_values(result)
        self.assertEqual(result.parse_status, "ambiguous_output")
        self.assertIsNone(prefill)
        self.assertIsNone(decode)
        self.assertEqual(len([metric for metric in result.metrics if metric.name == "prefill_tokens_per_second"]), 2)

    def test_nan_negative_and_locale_decimals_are_malformed_not_zero(self) -> None:
        for name in ["bad_values.txt", "locale_decimal.txt"]:
            result = parse_llama_bench_output(_fixture(name))
            self.assertEqual(result.parse_status, "malformed_output")
            self.assertEqual(result.metrics, [])

    def test_metrics_present_only_in_stderr_are_marked_stderr(self) -> None:
        result = parse_llama_bench_streams("", _fixture("stderr_only.txt"))

        self.assertEqual(result.parse_status, "parsed")
        self.assertEqual({metric.source for metric in result.metrics}, {"stderr"})

    def test_successful_exit_empty_stdout_is_no_metrics(self) -> None:
        result = parse_llama_bench_streams("", "")

        self.assertEqual(result.parse_status, "no_metrics")

    def test_invalid_utf8_replacement_does_not_create_zero_metric(self) -> None:
        text = b"| model | test | t/s |\n| --- | --- | --- |\n| fake | pp128 | \xff |\n".decode("utf-8", errors="replace")
        result = parse_llama_bench_output(text)

        self.assertEqual(result.parse_status, "malformed_output")
        self.assertEqual(result.metrics, [])


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
