from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Opcode = Literal["SET", "ADD", "SUB", "MUL", "XOR", "COPY", "EMIT"]


@dataclass(frozen=True, slots=True)
class Operation:
    opcode: Opcode
    register: int
    argument: int = 0


@dataclass(frozen=True, slots=True)
class Repeat:
    count: int
    body: tuple[Instruction, ...]


@dataclass(frozen=True, slots=True)
class IfLess:
    register: int
    threshold: int
    body: tuple[Instruction, ...]


Instruction: TypeAlias = Operation | Repeat | IfLess


@dataclass(frozen=True, slots=True)
class Program:
    initial: tuple[int, int]
    body: tuple[Instruction, ...]


@dataclass(frozen=True, slots=True)
class VMLimits:
    modulus: int = 65536
    max_steps: int = 256
    max_output_bytes: int = 128
    max_depth: int = 3
    max_repeat: int = 5

    def __post_init__(self) -> None:
        if self.modulus < 2:
            raise ValueError("modulus must be >= 2")

        if min(self.max_steps, self.max_output_bytes, self.max_repeat) < 1:
            raise ValueError("VM execution limits must be positive")

        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative")


@dataclass(frozen=True, slots=True)
class VMResult:
    registers: tuple[int, int]
    output: bytes
    states: tuple[tuple[int, int], ...]
    steps: int


def execute_program(program: Program, limits: VMLimits | None = None) -> VMResult:
    """Execute with strict step, depth and output bounds."""
    if limits is None:
        limits = VMLimits()

    if len(program.initial) != 2:
        raise ValueError("VM requires exactly two registers")

    if any(not 0 <= x < limits.modulus for x in program.initial):
        raise ValueError("Initial registers must be within modulus")

    registers = list(program.initial)
    emitted = bytearray()
    states: list[tuple[int, int]] = []
    steps = 0

    def visit(instructions: tuple[Instruction, ...], depth: int) -> None:
        nonlocal steps

        if depth > limits.max_depth:
            raise ValueError("Maximum nesting depth exceeds")

        for instruction in instructions:
            steps += 1

            if steps > limits.max_steps:
                raise ValueError("Maximum execution steps exceeded")

            if isinstance(instruction, Operation):
                r = instruction.register
                v = instruction.argument

                if r not in (0, 1):
                    raise ValueError("Invalid register index")

                current = registers[r]

                if instruction.opcode == "SET":
                    registers[r] = v % limits.modulus

                elif instruction.opcode == "ADD":
                    registers[r] = (current + v) % limits.modulus

                elif instruction.opcode == "SUB":
                    registers[r] = (current - v) % limits.modulus

                elif instruction.opcode == "MUL":
                    registers[r] = (current * v) % limits.modulus

                elif instruction.opcode == "XOR":
                    registers[r] = (current ^ (v % limits.modulus)) % limits.modulus

                elif instruction.opcode == "COPY":
                    if v not in (0, 1):
                        raise ValueError("COPY requires source register 0 or 1")
                    registers[r] = registers[v]

                elif instruction.opcode == "EMIT":
                    if len(emitted) >= limits.max_output_bytes:
                        raise ValueError("Maximum output bytes exceeded")

                    emitted.append(current & 0xFF)

                else:
                    raise ValueError("Unknown opcode")

                states.append((registers[0], registers[1]))

            elif isinstance(instruction, Repeat):
                if not 1 <= instruction.count <= limits.max_repeat:
                    raise ValueError("Invalid repeat count")

                if not instruction.body:
                    raise ValueError("Repeat body cannot be empty")

                for _ in range(instruction.count):
                    visit(instruction.body, depth + 1)

            elif isinstance(instruction, IfLess):
                if instruction.register not in (0, 1):
                    raise ValueError("Invalid branch register")

                if not (0 <= instruction.threshold < limits.modulus):
                    raise ValueError("Branch threshold out of range")

                if not instruction.body:
                    raise ValueError("Branch body cannot be empty")

                if registers[instruction.register] < instruction.threshold:
                    visit(instruction.body, depth + 1)

            else:
                raise TypeError("Unsupported VM instruction")

    visit(program.body, 0)

    return VMResult(
        registers=(registers[0], registers[1]),
        output=bytes(emitted),
        states=tuple(states),
        steps=steps,
    )


def _instruction_lines(
    instructions: tuple[Instruction, ...], depth: int = 0
) -> list[str]:
    result: list[str] = []
    pad = " " * depth

    for ins in instructions:
        if isinstance(ins, Operation):
            if ins.opcode == "EMIT":
                line = f"EMIT{ins.register}"
            elif ins.opcode == "COPY":
                line = f"COPY{ins.register},{ins.argument}"
            else:
                line = f"{ins.opcode}{ins.register},{ins.argument}"

            result.append(pad + line)

        elif isinstance(ins, Repeat):
            result.append(pad + f"R{ins.count}" + "{")
            result.extend(_instruction_lines(ins.body, depth + 1))
            result.append(pad + "}")

        elif isinstance(ins, IfLess):
            result.append(pad + f"L{ins.register}<{ins.threshold}" + "{")
            result.extend(_instruction_lines(ins.body, depth + 1))
            result.append(pad + "}")

    return result


def serialize_execution(
    program: Program, result: VMResult, *, format: Literal["trace", "binary"] = "trace"
) -> bytes:
    """Encode a program and its verified execution."""

    header = [f"VM r={program.initial[0]},{program.initial[1]}", "P:"]

    header.extend(_instruction_lines(program.body))

    prefix = ("\n".join(header) + "\n").encode("ascii")

    if format == "trace":
        states = ";".join(f"{a},{b}" for a, b in result.states)
        trailer = f"T:{states}\nH:{result.registers[0]},{result.registers[1]}\nO:{result.output.hex()}\n"

        return prefix + trailer.encode("ascii")

    if format == "binary":
        return prefix + b"O:\n" + result.output

    raise ValueError("Unknown VM serialization format")
