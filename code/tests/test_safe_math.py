from __future__ import annotations

import pytest

from shared.safe_math import UnsafeExpression, evaluate_arithmetic


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(10 - 4) / 3", 2.0),
        ("7 // 3", 2),
        ("7 % 3", 1),
        ("2 ** 10", 1024),
        ("-1.5 + 2", 0.5),
    ],
)
def test_evaluate_arithmetic(expression: str, expected: int | float) -> None:
    assert evaluate_arithmetic(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "(1).__class__.__mro__",
        "[1, 2, 3]",
        "abs(-1)",
        "True + 1",
        "2 ** 1000",
        "9" * 129,
    ],
)
def test_evaluate_arithmetic_rejects_code_and_resource_abuse(expression: str) -> None:
    with pytest.raises(UnsafeExpression):
        evaluate_arithmetic(expression)


def test_evaluate_arithmetic_reports_invalid_math() -> None:
    with pytest.raises(ZeroDivisionError):
        evaluate_arithmetic("1 / 0")
