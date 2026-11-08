from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.sft.arithmetic import (
    render_expression,
    sample_expression,
    verify_expression,
)
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.schema import (
    Difficulty,
    SFTExample,
    TaskType,
    format_response,
    validate_sft_example,
)
from synthesize_vl.sft.visual_arithmetic import sample_visual_arithmetic
from synthesize_vl.tokenizer import ByteTokenizer

Split = Literal["train", "validation", "test"]

_SPLIT_IDS = {"train": 2101, "validation": 2203, "test": 2309}


def generate_sft_example(
    config: SFTDataConfig,
    *,
    split: Split,
    index: int,
    task_override: TaskType | None = None,
    difficulty_override: Difficulty | None = None,
    cot_override: bool | None = None,
) -> SFTExample:
    """Generate one verified example without mutable RNG state."""
    if split not in _SPLIT_IDS:
        raise ValueError("Unknown dataset split")

    if index < 0:
        raise ValueError("Index must be nonnegative")

    rng = np.random.default_rng(np.random.SeedSequence([config.seed, _SPLIT_IDS[split], index, 21]))

    sampled_task: TaskType = "text_arithmetic" if rng.random() < config.text_probability else "visual_arithmetic"

    task = sampled_task if task_override is None else task_override

    sampled_difficulty: Difficulty = ("easy", "medium", "hard")[
        int(rng.choice(3, p=config.difficulty_weights.probabilities()))
    ]

    difficulty = sampled_difficulty if difficulty_override is None else difficulty_override

    if difficulty not in ("easy", "medium", "hard", "ood"):
        raise ValueError("Unknown difficulty")

    # OOD problems must never enter the training stream
    if difficulty == "ood" and split == "train":
        raise ValueError("OOD difficulty is reserved for evaluation")

    sampled_cot = bool(rng.random() < config.cot_probability)

    show_work = sampled_cot if cot_override is None else cot_override
    instruction = "Show your steps." if show_work else "Give only the answer."

    if task == "text_arithmetic":
        expression = sample_expression(rng, difficulty)
        answer, steps = verify_expression(expression)
        prompt = f"Question: Compute {render_expression(expression)}.\n{instruction}\n"

        image = None

    else:
        problem = sample_visual_arithmetic(rng, difficulty)

        answer = problem.answer
        steps = problem.steps

        prompt = f"Question: In the image, {problem.prompt_rules}. What is the total score?\n{instruction}\n"

        image = problem.image

    response = format_response(answer=answer, steps=steps, show_work=show_work)
    example = SFTExample(
        sample_id=f"sft:{split}:{index:012d}:{task}:{difficulty}:{'cot' if show_work else 'direct'}",
        split=split,
        task=task,
        difficulty=difficulty,
        prompt=prompt,
        response=response,
        oracle_answer=answer,
        oracle_steps=steps,
        show_work=show_work,
        image=image,
    )

    validate_sft_example(example)

    return example


def generate_sft_examples(
    config: SFTDataConfig,
    *,
    split: Split,
    start: int,
    count: int,
    task_override: TaskType | None = None,
    difficulty_override: Difficulty | None = None,
    cot_override: bool | None = None,
) -> list[SFTExample]:
    if start < 0 or count < 1:
        raise ValueError("Invalid sample range")

    return [
        generate_sft_example(
            config,
            split=split,
            index=index,
            task_override=task_override,
            difficulty_override=difficulty_override,
            cot_override=cot_override,
        )
        for index in range(start, start + count)
    ]


def pack_sft_examples(examples: Sequence[SFTExample], *, max_length: int = 1024) -> TrainingBatch:
    """Pack SFT responses with answer-only loss mask."""
    if not examples:
        raise ValueError("Cannot pack an empty batch")

    if max_length < 2:
        raise ValueError("Invalid context length")

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    masks: list[list[bool]] = []
    images: list[np.ndarray] = []
    has_image: list[bool] = []

    for example in examples:
        validate_sft_example(example)

        visual = example.image is not None
        prompt_bytes = example.prompt.encode("utf-8")
        response_bytes = example.response.encode("utf-8")

        sequence = (
            [ByteTokenizer.BOS]
            + ([ByteTokenizer.IMAGE] if visual else [])
            + list(prompt_bytes)
            + list(response_bytes)
            + [ByteTokenizer.EOS]
        )

        valid_length = len(sequence) - 1

        if valid_length > max_length:
            raise ValueError(f"{example.sample_id} needs {valid_length} text tokens, capacity={max_length}")

        supervision_start = int(visual) + len(prompt_bytes)

        padding = max_length - valid_length

        inputs.append(sequence[:-1] + [ByteTokenizer.PAD] * padding)
        targets.append(sequence[1:] + [ByteTokenizer.PAD] * padding)
        masks.append([position >= supervision_start for position in range(valid_length)] + [False] * padding)

        if visual:
            image_array = np.asarray(example.image.convert("RGB"), dtype=np.float32) / 255.0  # type: ignore
            if image_array.shape != (64, 64, 3):
                raise ValueError("Invalid visual image shape")

        else:
            image_array = np.zeros((64, 64, 3), dtype=np.float32)

        images.append(image_array)
        has_image.append(visual)

    return TrainingBatch(
        input_ids=jnp.asarray(inputs, dtype=jnp.int32),
        target_ids=jnp.asarray(targets, dtype=jnp.int32),
        loss_mask=jnp.asarray(masks, dtype=jnp.bool_),
        images=jnp.asarray(np.stack(images), dtype=jnp.float32),
        has_image=jnp.asarray(has_image, dtype=jnp.bool_),
    )


def make_sft_batch(config: SFTDataConfig, *, split: Split, start: int) -> TrainingBatch:
    examples = generate_sft_examples(config, split=split, start=start, count=config.batch_size)

    return pack_sft_examples(examples, max_length=config.max_text_length)
