import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import run_evals  # noqa: E402


def test_golden_questions_pass_on_offline_engine():
    failures = []
    for case in run_evals.CASES:
        ok, errs, _, _ = run_evals.run_case(case, "offline")
        if not ok:
            failures.append((case.id, errs))
    assert not failures, failures


def test_number_matching_helpers():
    assert run_evals.has_number("Revenue was $844.6K this year", 844624)
    assert run_evals.has_number("total 1,234,567.89", 1234567.89)
    assert not run_evals.has_number("total 1,000", 5000)
