import re
from dataclasses import dataclass
from typing import Literal

from PIL import Image

TaskType = Literal["text_arithmetic", "visual_arithmetic"]
Difficulty = Literal["easy", "medium", "hard", "ood"]


@dataclass(frozen=True, slots=True)
class SFTExample:
    sample_id: str
    split: Literal["train", "validation", "test"]
    task: TaskType
    difficulty: Difficulty
    prompt: str
    response: str
    oracle_answer: int
    oracle_steps: tuple[str, ...]
    show_work: bool
    image: Image.Image | None = None


def format_response(*, answer: int, steps: tuple[str, ...], show_work: bool) -> str:
    """One stable output format across both tasks"""
    if show_work:
        if not steps:
            raise ValueError("A worked solution must have reasoning steps")

        return "Work:\n" + "\n".join(steps) + f"\nAnswer: {answer}\n"

    return f"Answer: {answer}\n"


_FINAL_ANSWER_PATTERN = re.compile(r"(?:^|\n)Answer:[ \t]*([+-]?\d+)[ \t]*\n?\Z")


def parse_final_answer(output: str | bytes) -> int | None:
    """Extract a final integer only from a well-formed answer line"""

    if isinstance(output, bytes):
        try:
            output = output.decode("utf-8")
        except UnicodeDecodeError:
            return None

    match = _FINAL_ANSWER_PATTERN.search(output)  # type: ignore

    if match is None:
        return None

    return int(match.group(1))


def exact_answer_reward(output: str | bytes, oracle_answer: int) -> float:
    prediction = parse_final_answer(output)

    return float(prediction is not None and prediction == oracle_answer)


def validate_sft_example(example: SFTExample) -> None:
    """Check formatting and oracle consistency."""
    if not example.prompt:
        raise ValueError("Empty prompt")

    if example.image is not None:
        if example.task != "visual_arithmetic":
            raise ValueError("Unexpected image on text task")

        if example.image.size != (64, 64):
            raise ValueError("Unexpected image resolution")

    elif example.task == "visual_arithmetic":
        raise ValueError("Visual example has not image")

    expected_response = format_response(
        answer=example.oracle_answer,
        steps=example.oracle_steps,
        show_work=example.show_work,
    )

    if example.response != expected_response:
        raise ValueError("Response differs from oracle trace")

    if parse_final_answer(example.response) != example.oracle_answer:
        raise ValueError("Final answer verification failed")
