from __future__ import annotations

from collections import defaultdict

from autotune.arm_tune.quality.diagnostics import failure_category_histogram
from autotune.arm_tune.quality.schemas import QualityAggregateResult, QualityCaseResult


def aggregate_quality_results(results: list[QualityCaseResult]) -> QualityAggregateResult:
    grouped: dict[str, list[QualityCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.case_id].append(result)
    total_cases = len(grouped)
    required_cases = sum(1 for case_results in grouped.values() if any(item.required for item in case_results))
    passed_cases = 0
    failed_cases = 0
    failure_reasons: list[str] = []
    critical_failures = 0
    json_valid = 0
    json_total = 0
    evidence_valid = 0
    evidence_total = 0
    required_passed = 0
    for case_id, case_results in grouped.items():
        case_required = any(item.required for item in case_results)
        case_passed = all(item.status == "passed" for item in case_results)
        if case_passed:
            passed_cases += 1
            if case_required:
                required_passed += 1
        else:
            failed_cases += 1
            failure_reasons.append(f"{case_id}: case failed")
        for result in case_results:
            critical_failures += sum(1 for item in result.validator_results if item.critical and item.status != "passed")
            for item in result.validator_results:
                if item.validator_type == "valid_json":
                    json_total += 1
                    if item.status == "passed":
                        json_valid += 1
                if item.validator_type == "evidence_line_ids":
                    evidence_total += 1
                    if item.status == "passed":
                        evidence_valid += 1
            failure_reasons.extend(result.failure_reasons)
    if total_cases == 0:
        status = "not_evaluated"
    elif any(result.status == "execution_failed" for result in results):
        status = "execution_failed"
    elif critical_failures:
        status = "failed"
    elif required_passed == required_cases:
        status = "passed"
    else:
        status = "incomplete"
    return QualityAggregateResult(
        total_cases=total_cases,
        required_cases=required_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        critical_validator_failures=critical_failures,
        json_validity_rate=_rate(json_valid, json_total),
        required_case_pass_rate=_rate(required_passed, required_cases),
        evidence_citation_correctness=_rate(evidence_valid, evidence_total),
        quality_gate_status=status,
        failure_reasons=sorted(set(reason for reason in failure_reasons if reason)),
        failure_category_histogram=failure_category_histogram(results),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 6)
