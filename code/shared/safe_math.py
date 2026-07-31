"""Small arithmetic evaluator for examples that accept untrusted tool input."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

Number = int | float

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 128
_MAX_AST_NODES = 64
_MAX_ABS_RESULT = 1e100
_MAX_ABS_EXPONENT = 12


class UnsafeExpression(ValueError):
    """Raised when an arithmetic expression exceeds the supported grammar."""


def evaluate_arithmetic(expression: str) -> Number:
    """Evaluate bounded numeric arithmetic without executing Python code.

    Supported syntax: numeric literals, parentheses, ``+ - * / // %`` and
    bounded ``**``. Names, calls, attributes, subscriptions and containers are
    rejected.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpression("expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise UnsafeExpression("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise UnsafeExpression("invalid arithmetic syntax") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise UnsafeExpression("expression is too complex")

    result = _evaluate_node(tree.body)
    _check_result(result)
    return result


def _evaluate_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpression("only real numeric literals are allowed")
        _check_result(value)
        return value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
        _check_result(value)
        return value

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_ABS_EXPONENT:
                raise UnsafeExpression("exponent is too large")
            value = operator.pow(left, right)
        else:
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise UnsafeExpression("operator is not allowed")
            value = operation(left, right)

        if isinstance(value, complex):
            raise UnsafeExpression("complex results are not allowed")
        _check_result(value)
        return value

    raise UnsafeExpression("only numeric arithmetic is allowed")


def _check_result(value: Number) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise UnsafeExpression("result must be finite")
    if abs(value) > _MAX_ABS_RESULT:
        raise UnsafeExpression("result magnitude is too large")
