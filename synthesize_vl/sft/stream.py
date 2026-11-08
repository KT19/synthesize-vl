from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import (
    generate_sft_example,
    generate_sft_examples,
    pack_sft_examples,
)

LEVELS = ("easy", "medium", "hard", "ood")
TASKS = ("text_arithmetic", "visual_arithmetic")


def next_sft_batch(
    config: SFTDataConfig, cursor: StreamCursor
) -> tuple[TrainingBatch, StreamCursor]:
    examples = generate_sft_examples(
        config, split="train", start=cursor.next_index, count=config.batch_size
    )

    batch = pack_sft_examples(examples, max_length=config.max_text_length)

    input_bytes = sum(
        len(example.prompt.encode("utf-8")) + len(example.response.encode("utf-8"))
        for example in examples
    )
    targets = sum(len(example.response.encode("utf-8")) + 1 for example in examples)

    updated = StreamCursor(
        next_index=cursor.next_index + len(examples),
        updates=cursor.updates + 1,
        documents_seen=cursor.documents_seen + len(examples),
        bytes_seen=cursor.bytes_seen + input_bytes,
        targets_seen=cursor.targets_seen + targets,
    )

    return batch, updated


def make_sft_validation_batches(
    config: SFTDataConfig, *, examples_per_group: int
) -> dict[str, TrainingBatch]:
    if examples_per_group < 1:
        raise ValueError("examples_per_group must be positive")

    batches: dict[str, TrainingBatch] = {}

    for task_index, task in enumerate(TASKS):
        for level_index, difficulty in enumerate(LEVELS):
            examples = [
                generate_sft_example(
                    config,
                    split="validation",
                    index=100000 + task_index * 100000 + i,
                    task_override=task,
                    difficulty_override=difficulty,
                    cot_override=True,
                )
                for i in range(examples_per_group)
            ]

            batches[f"{task}/{difficulty}"] = pack_sft_examples(
                examples, max_length=config.max_text_length
            )

    return batches
