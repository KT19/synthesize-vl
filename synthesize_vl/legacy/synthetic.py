from dataclasses import dataclass

import jax
from jax import random

OPERATORS = ("+", "-", "*")


@dataclass(frozen=True)
class Example:
    kind: str
    prompt: str
    answer: str


@dataclass(frozen=True)
class Instruction:
    operator: str
    operand: int


# Utilities
def sample_int(key: jax.Array, low: int, high: int) -> int:
    """Sample an integer in the inclusive range [low, high]"""
    return int(random.randint(key, (), low, high + 1))


def apply_operator(value: int, operator: str, operand: int) -> int:
    if operator == "+":
        return value + operand
    if operator == "-":
        return value - operand
    if operator == "*":
        return value * operand

    raise ValueError(f"Unsupported operator: {operator}")


def execute_program(start: int, instructions: tuple[Instruction, ...]) -> int:
    """Interpret a tiny, safe subset of arithmetic programs."""
    value = start

    for instruction in instructions:
        value = apply_operator(value, instruction.operator, instruction.operand)

    return value


# Math generator
def make_math_example(key: jax.Array) -> Example:
    key_a, key_b, key_op = random.split(key, 3)

    a = sample_int(key_a, -9, 9)
    b = sample_int(key_b, -9, 9)
    op = OPERATORS[sample_int(key_op, 0, 2)]

    return Example(
        kind="math",
        prompt=f"Compute: {a} {op} {b}\nAnswer:",
        answer=str(apply_operator(a, op, b)),
    )


# Code generator
def make_code_example(key: jax.Array) -> Example:
    key_start, key_a, key_b, key_op_a, key_op_b = random.split(key, 5)

    start = sample_int(key_start, 0, 9)
    a = sample_int(key_a, 1, 5)
    b = sample_int(key_b, 1, 5)

    op_a = OPERATORS[sample_int(key_op_a, 0, 2)]
    op_b = OPERATORS[sample_int(key_op_b, 0, 2)]

    instructions = (Instruction(op_a, a), Instruction(op_b, b))

    answer = execute_program(start, instructions=instructions)

    lines = [f"x = {start}"]

    for instruction in instructions:
        lines.append(f"x = x {instruction.operator} {instruction.operand}")

    lines.append("print(x)")
    program = "\n".join(lines)

    return Example(
        kind="code",
        prompt=(f"What does this program print?\n```python\n{program}\n```\nOutput:"),
        answer=str(answer),
    )


# Dataset Generator
def make_dataset(seed: int, count: int) -> list[Example]:
    """Generate reproducible examples indexed by example number."""
    root_key = random.key(seed)
    examples = []

    for index in range(count):
        example_key = random.fold_in(root_key, index)

        if index % 2 == 0:
            examples.append(make_math_example(example_key))
        else:
            examples.append(make_code_example(example_key))

    return examples
