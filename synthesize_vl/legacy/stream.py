from dataclasses import dataclass
from typing import Literal

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.legacy.generators import generate_documents
from synthesize_vl.legacy.pretraining import pack_pretraining_documents
from synthesize_vl.pretraining.vm_curriculum import (
    TrainingVMProfile,
    generate_vm_documents,
)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    seed: int = 17
    batch_size: int = 8
    max_length: int = 128

    generator: Literal["legacy", "vm"] = "legacy"
    vm_profile: TrainingVMProfile = "mixed"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.generator not in ("legacy", "vm"):
            raise ValueError("Unknown pretraining generator")
        if self.vm_profile not in ("mixed", "foundation", "intermediate", "advanced"):
            raise ValueError("Held-out VM profiles cannot be selected for training")


def next_training_batch(
    config: StreamConfig, cursor: StreamCursor
) -> tuple[TrainingBatch, StreamCursor]:
    """Create the next batch without mutating the cursor"""
    if config.generator == "legacy":
        documents = generate_documents(
            seed=config.seed,
            split="train",
            start=cursor.next_index,
            count=config.batch_size,
        )
    else:
        documents = generate_vm_documents(
            seed=config.seed,
            split="train",
            start=cursor.next_index,
            count=config.batch_size,
            profile=config.vm_profile,
            max_payload_bytes=config.max_length - 1,
        )
    batch = pack_pretraining_documents(documents, max_length=config.max_length)
    byte_count = sum(len(document.payload) for document in documents)
    supervised_count = sum(len(document.payload) + 1 for document in documents)

    advanced = StreamCursor(
        next_index=cursor.next_index + len(documents),
        updates=cursor.updates + 1,
        documents_seen=cursor.documents_seen + len(documents),
        bytes_seen=cursor.bytes_seen + byte_count,
        targets_seen=cursor.targets_seen + supervised_count,
    )

    return batch, advanced
