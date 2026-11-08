import ast

from jax import random

from synthesize_vl.legacy.synthetic import (
    Example,
    Instruction,
    apply_operator,
    execute_program,
    make_code_example,
    make_dataset,
    make_math_example,
)


def test_math_operators():
    assert apply_operator(3, "+", 4) == 7
    assert apply_operator(3, "-", 4) == -1
    assert apply_operator(3, "*", 4) == 12


def test_program_interpreter():
    steps = (Instruction("+", 4), Instruction("*", 3))

    assert execute_program(2, steps) == 18


def test_math_example():
    example = make_math_example(random.key(0))

    assert isinstance(example, Example)

    expression = example.prompt.removeprefix("Compute: ").removesuffix("\nAnswer:")

    a, op, b = expression.split()

    assert int(example.answer) == apply_operator(int(a), op, int(b))


def test_code_is_valid_python():
    example = make_code_example(random.key(1))

    code = example.prompt.split("```python\n", 1)[1].split("\n```", 1)[0]

    ast.parse(code)
    assert example.answer.lstrip("-").isdigit()


def test_reproducibility():
    assert make_dataset(42, 12) == make_dataset(42, 12)
    assert make_dataset(42, 12) != make_dataset(43, 12)


def test_dataset_has_both_modalities():
    examples = make_dataset(42, 6)

    assert [example.kind for example in examples] == [
        "math",
        "code",
        "math",
        "code",
        "math",
        "code",
    ]
