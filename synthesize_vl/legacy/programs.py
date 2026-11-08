from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RegisterInstruction:
    opcode: Literal["ADD", "SUB", "MUL"]
    argument: int


def execute_register(
    initial: int, program: Sequence[RegisterInstruction]
) -> tuple[int, ...]:
    """Return the register value after each instruction"""
    value = initial
    states: list[int] = []

    for instruction in program:
        if instruction.opcode == "ADD":
            value += instruction.argument
        elif instruction.opcode == "SUB":
            value -= instruction.argument
        elif instruction.opcode == "MUL":
            value *= instruction.argument
        else:
            raise ValueError(f"Unknown opcode: {instruction.opcode}")

        states.append(value)

    return tuple(states)


@dataclass(frozen=True, slots=True)
class StackInstruction:
    opcode: Literal["PUSH", "ADD", "MUL"]
    argument: int | None = None


def execute_stack(program: Sequence[StackInstruction]) -> tuple[tuple[int, ...], ...]:
    """Return a snapshot after each validated stack instruction."""

    stack: list[int] = []
    states: list[tuple[int, ...]] = []

    for instruction in program:
        if instruction.opcode == "PUSH":
            if instruction.argument is None:
                raise ValueError("PUSH requires an argument")

            stack.append(instruction.argument)

        elif instruction.opcode in ("ADD", "MUL"):
            if len(stack) < 2:
                raise ValueError("Stack underflow")

            right = stack.pop()
            left = stack.pop()

            stack.append(left + right if instruction.opcode == "ADD" else left * right)

        else:
            raise ValueError(f"Unknown opcode: {instruction.opcode}")

        states.append(tuple(stack))

    return tuple(states)
