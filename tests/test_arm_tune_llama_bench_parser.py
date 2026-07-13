from __future__ import annotations

import unittest
from pathlib import Path

from autotune.arm_tune.benchmark.parser import parse_llama_bench_output, required_metric_values


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


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
