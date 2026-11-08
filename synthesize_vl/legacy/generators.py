from typing import Literal

import numpy as np

from synthesize_vl.data.schema import PretrainingDocument, Split
from synthesize_vl.legacy.programs import (
    RegisterInstruction,
    StackInstruction,
    execute_register,
    execute_stack,
)

FAMILIES: tuple[str, ...] = ("register_trace", "stack_trace", "grid_walk", "byte_cycle")

_SPLIT_IDS: dict[str, int] = {"train": 11, "validation": 23, "test": 37}


def _register_trace(rng: np.random.Generator) -> bytes:
    initial = int(rng.integers(-9, 10))

    opcodes: tuple[Literal["ADD", "SUB", "MUL"], ...] = ("ADD", "SUB", "MUL")

    program = tuple(
        RegisterInstruction(
            opcode=opcodes[int(rng.integers(0, len(opcodes)))],
            argument=int(rng.integers(1, 5)),
        )
        for _ in range(int(rng.integers(3, 6)))
    )

    states = execute_register(initial, program)
    rows = [f"reg={initial}"]

    rows.extend(
        f"{ins.opcode} {ins.argument} -> {value}" for ins, value in zip(program, states)
    )

    rows.append(f"halt={states[-1]}")

    return ("\n".join(rows) + "\n").encode("ascii")


def _stack_trace(rng: np.random.Generator) -> bytes:
    a, b, c = (int(x) for x in rng.integers(-5, 6, size=3))
    binary_ops: tuple[Literal["ADD", "MUL"], ...] = ("ADD", "MUL")

    op1 = binary_ops[int(rng.integers(0, 2))]
    op2 = binary_ops[int(rng.integers(0, 2))]

    program = (
        StackInstruction("PUSH", a),
        StackInstruction("PUSH", b),
        StackInstruction(op1),
        StackInstruction("PUSH", c),
        StackInstruction(op2),
    )

    states = execute_stack(program)

    rows = [
        (
            f"{ins.opcode}{' ' + str(ins.argument) if ins.argument is not None else ''} -> {list(stack)}"
        )
        for ins, stack in zip(program, states)
    ]

    rows.append(f"halt={states[-1][-1]}")

    return ("\n".join(rows) + "\n").encode("ascii")


def _grid_walk(rng: np.random.Generator) -> bytes:
    x, y = (int(v) for v in rng.integers(0, 8, size=2))

    rows = [f"pos=({x},{y})"]
    directions = ("U", "D", "L", "R")

    for _ in range(int(rng.integers(4, 7))):
        direction = directions[int(rng.integers(0, 4))]

        dx, dy = {
            "U": (0, 1),
            "D": (0, -1),
            "L": (-1, 0),
            "R": (1, 0),
        }[direction]

        x = min(7, max(0, x + dx))
        y = min(7, max(0, y + dy))

        rows.append(f"{direction} -> ({x},{y})")

    rows.append(f"halt=({x},{y})")

    return ("\n".join(rows) + "\n").encode("ascii")


def _byte_cycle(rng: np.random.Generator) -> bytes:
    period_length = int(rng.integers(2, 9))

    pattern = rng.integers(0, 256, size=period_length, dtype=np.uint8).tobytes()

    length = int(rng.integers(48, 105))

    return (pattern * (length // period_length + 1))[:length]


def generate_document(*, seed: int, split: Split, index: int) -> PretrainingDocument:
    """Generate one document independently of sampling order."""

    if seed < 0 or index < 0:
        raise ValueError("seed and index must be nonnegative")

    if split not in _SPLIT_IDS:
        raise ValueError(f"Unknown split: {split}")

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, _SPLIT_IDS[split], index])
    )

    family_index = index % len(FAMILIES)
    family = FAMILIES[family_index]

    generators = (_register_trace, _stack_trace, _grid_walk, _byte_cycle)

    payload = generators[family_index](rng)

    return PretrainingDocument(
        sample_id=f"{split}:{index:012d}", split=split, family=family, payload=payload
    )


def generate_documents(
    *, seed: int, split: Split, start: int, count: int
) -> list[PretrainingDocument]:
    """Build only the requested batch of documents."""

    if start < 0 or count < 1:
        raise ValueError("start must be >= 0 and count must be >= 1")

    return [
        generate_document(seed=seed, split=split, index=index)
        for index in range(start, start + count)
    ]
