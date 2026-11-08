from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from synthesize_vl.data.schema import PretrainingDocument
from synthesize_vl.pretraining.vm import (
    IfLess,
    Instruction,
    Opcode,
    Operation,
    Program,
    Repeat,
    VMLimits,
    execute_program,
    serialize_execution,
)

VMProfile = Literal[
    "mixed",
    "foundation",
    "intermediate",
    "advanced",
    "heldout_length",
    "heldout_values",
]

TrainingVMProfile = Literal["mixed", "foundation", "intermediate", "advanced"]


@dataclass(frozen=True, slots=True)
class Difficulty:
    name: str
    min_statements: int
    max_statements: int
    operand_min: int
    operand_max: int
    repeat_probability: float
    branch_probability: float
    max_depth: int
    max_repeat: int

    def __post_init__(self) -> None:
        if not (1 <= self.min_statements <= self.max_statements):
            raise ValueError("Invalid program length range")

        if not (0 <= self.operand_min <= self.operand_max < 65536):
            raise ValueError("Invalid operand range")

        if (
            not 0 <= self.repeat_probability < 1
            or not 0 <= self.branch_probability < 1
            or not (self.repeat_probability + self.branch_probability < 1)
        ):
            raise ValueError("Invalid control flow probabilities")

        if self.max_depth < 0 or self.max_repeat < 1:
            raise ValueError("Invalid control flow bounds")

        if self.repeat_probability > 0 and self.max_repeat < 2:
            raise ValueError("Repeats require max_repeat >= 2")


DIFFICULTIES: dict[str, Difficulty] = {
    "foundation": Difficulty("foundation", 3, 5, 0, 9, 0.0, 0.0, 0, 1),
    "intermediate": Difficulty("intermediate", 5, 8, 10, 99, 0.12, 0.12, 1, 3),
    "advanced": Difficulty("advanced", 9, 13, 100, 999, 0.16, 0.14, 2, 4),
    "heldout_length": Difficulty("heldout_length", 18, 23, 10, 99, 0.10, 0.12, 2, 4),
    "heldout_values": Difficulty("heldout_values", 9, 13, 1000, 9999, 0.16, 0.14, 2, 4),
}

_TRAIN_PROFILES = ("foundation", "intermediate", "advanced")
_SPLIT_IDS = {"train": 101, "validation": 211, "test": 307}


def _pick_profile(rng: np.random.Generator, profile: VMProfile) -> str:
    if profile == "mixed":
        return str(
            rng.choice(
                _TRAIN_PROFILES,
                p=(0.40, 0.35, 0.25),
            )
        )

    if profile not in DIFFICULTIES:
        raise ValueError(f"Unknown VM profile: {profile}")

    return profile


def _operand(rng: np.random.Generator, difficulty: Difficulty) -> int:
    magnitude = int(rng.integers(difficulty.operand_min, difficulty.operand_max + 1))

    return -magnitude if rng.random() < 0.35 else magnitude


def _operation(rng: np.random.Generator, difficulty: Difficulty) -> Operation:
    register = int(rng.integers(0, 2))

    opcode = cast(
        Opcode,
        str(
            rng.choice(
                ("SET", "ADD", "SUB", "MUL", "XOR", "COPY", "EMIT"),
                p=(0.13, 0.21, 0.18, 0.12, 0.11, 0.10, 0.15),
            )
        ),
    )

    if opcode == "COPY":
        return Operation("COPY", register, 1 - register)

    if opcode == "EMIT":
        return Operation("EMIT", register)

    return Operation(opcode, register, _operand(rng, difficulty))


def _statement(
    rng: np.random.Generator, difficulty: Difficulty, depth: int
) -> Instruction:
    if depth < difficulty.max_depth:
        probability = float(rng.random())

        if probability < difficulty.repeat_probability:
            body_size = int(rng.integers(1, 3))

            return Repeat(
                count=int(rng.integers(2, difficulty.max_repeat + 1)),
                body=tuple(
                    _statement(rng, difficulty, depth=depth + 1)
                    for _ in range(body_size)
                ),
            )

        if probability < (
            difficulty.repeat_probability + difficulty.branch_probability
        ):
            return IfLess(
                register=int(rng.integers(0, 2)),
                threshold=int(
                    rng.integers(difficulty.operand_min, difficulty.operand_max + 1)
                ),
                body=(_statement(rng, difficulty, depth + 1),),
            )

    return _operation(rng, difficulty)


def generate_program(rng: np.random.Generator, difficulty: Difficulty) -> Program:
    """Sample a program with an observable final output."""

    count = int(rng.integers(difficulty.min_statements, difficulty.max_statements + 1))

    initial = (
        int(rng.integers(difficulty.operand_min, difficulty.operand_max + 1)),
        int(rng.integers(difficulty.operand_min, difficulty.operand_max + 1)),
    )

    body = tuple(_statement(rng, difficulty, 0) for _ in range(count))

    return Program(initial=initial, body=body + (Operation("EMIT", 0),))


def generate_vm_document(
    *,
    seed: int,
    split: Literal["train", "validation", "test"],
    index: int,
    profile: VMProfile = "mixed",
    max_payload_bytes: int = 511,
    max_attempts: int = 64,
) -> PretrainingDocument:
    """Generate a deterministic document without truncation"""

    if seed < 0 or index < 0:
        raise ValueError("seed/index must be nonnegative")

    if split not in _SPLIT_IDS:
        raise ValueError("Unknown split")

    if max_payload_bytes < 1 or max_attempts < 1:
        raise ValueError("invalid document budget or attempt limit")

    selector = np.random.default_rng(
        np.random.SeedSequence([seed, _SPLIT_IDS[split], index, 991])
    )
    chosen = _pick_profile(selector, profile)

    difficulty = DIFFICULTIES[chosen]

    limits = VMLimits(
        max_steps=512,
        max_output_bytes=128,
        max_depth=difficulty.max_depth,
        max_repeat=difficulty.max_repeat,
    )

    for attempt in range(max_attempts):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, _SPLIT_IDS[split], index, attempt, 7193])
        )

        program = generate_program(rng, difficulty)

        try:
            result = execute_program(program, limits)
        except ValueError:
            continue

        mode: Literal["trace", "binary"] = "binary" if index % 5 == 0 else "trace"

        payload = serialize_execution(
            program,
            result,
            format=mode,
        )

        if 0 < len(payload) <= max_payload_bytes:
            return PretrainingDocument(
                sample_id=(f"vm:{split}:{index:012d}:{chosen}"),
                split=split,
                family=f"vm_{chosen}",
                payload=payload,
            )

    raise ValueError(
        f"Unable to fit {chosen} sample {index} "
        f"within {max_payload_bytes} bytes "
        f"after {max_attempts} attempts; "
        "increase --max-length or simplify the difficulty profile"
    )


def generate_vm_documents(
    *,
    seed: int,
    split: Literal["train", "validation", "test"],
    start: int,
    count: int,
    profile: VMProfile = "mixed",
    max_payload_bytes: int = 511,
) -> list[PretrainingDocument]:
    if start < 0 or count < 1:
        raise ValueError("Invalid document range")

    return [
        generate_vm_document(
            seed=seed,
            split=split,
            index=index,
            profile=profile,
            max_payload_bytes=max_payload_bytes,
        )
        for index in range(start, start + count)
    ]
