from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from synthesize_vl.sft.schema import Difficulty

Operator = Literal["+", "-", "*", "/"]


@dataclass(frozen=True, slots=True)
class Number:
    value: int


@dataclass(frozen=True, slots=True)
class Binary:
    op: Operator
    left: Expr
    right: Expr
    value: int


Expr = Number | Binary

# Minimum operations, maximum operations,
# maximum input operand magnitude, maximum intermediate magnitude.
LEVELS: dict[Difficulty, tuple[int, int, int, int]] = {
    "easy": (1, 2, 10, 100),
    "medium": (3, 4, 30, 1000),
    "hard": (5, 6, 50, 10000),
    "ood": (7, 8, 100, 100000),
}


def calculate(a: int, b: int, op: Operator) -> int:
    if op == "+":
        return a + b

    if op == "-":
        return a - b

    if op == "*":
        return a * b

    if op == "/":
        if b == 0 or a % b:
            raise ValueError("Division must be exact")

        return a // b

    raise ValueError(f"Unknown arithmetic operator: {op}")


def sample_expression(rng: np.random.Generator, difficulty: Difficulty) -> Expr:
    if difficulty not in LEVELS:
        raise ValueError("Unknown difficulty")

    min_ops, max_ops, max_operand, max_intermediate = LEVELS[difficulty]

    expr: Expr = Number(int(rng.integers(-max_operand, max_operand + 1)))

    operations = int(rng.integers(min_ops, max_ops + 1))

    for _ in range(operations):
        for _attempt in range(128):
            op: Operator = ("+", "-", "*", "/")[int(rng.integers(4))]

            if op == "/":
                possible = [d for d in range(2, min(abs(expr.value), max_operand) + 1) if expr.value % d == 0]

                if not possible:
                    continue

                divisor = possible[int(rng.integers(len(possible)))]

                if rng.random() < 0.5:
                    divisor = -divisor

                left, right = expr, Number(divisor)

            else:
                operand = Number(int(rng.integers(-max_operand, max_operand + 1)))
                left, right = (expr, operand) if rng.random() < 0.5 else (operand, expr)

            try:
                result = calculate(left.value, right.value, op)
            except ValueError:
                continue

            if abs(result) > max_intermediate:
                continue

            expr = Binary(op=op, left=left, right=right, value=result)

            break

        else:
            raise RuntimeError("Failed to construct a bounded expression")

    return expr


def render_expression(expr: Expr) -> str:
    """Fully parenthesized expression for the problem prompt"""

    if isinstance(expr, Number):
        return str(expr.value)

    return f"({render_expression(expr.left)} {expr.op} {render_expression(expr.right)})"


def verify_expression(expr: Expr) -> tuple[int, tuple[str, ...]]:
    """independently reevaluate the AST in postorder."""

    if isinstance(expr, Number):
        return expr.value, ()

    left, left_steps = verify_expression(expr.left)
    right, right_steps = verify_expression(expr.right)

    result = calculate(left, right, expr.op)

    if result != expr.value:
        raise ValueError("Stored expression value is incorrect")

    step = f"{left} {expr.op} {right} = {result}"

    return (
        result,
        left_steps + right_steps + (step,),
    )
