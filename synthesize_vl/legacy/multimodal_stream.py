from dataclasses import dataclass
from typing import Literal

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.legacy.multimodal_pretraining import (
    MixedDocument,
    pack_mixed_documents,
)
from synthesize_vl.pretraining.visual import generate_visual_document
from synthesize_vl.pretraining.vm_curriculum import (
    TrainingVMProfile,
    generate_vm_document,
)

TrainingVisualProfile = Literal["mixed", "foundation", "intermediate", "advanced"]


@dataclass(frozen=True, slots=True)
class MixedStreamConfig:
    seed: int = 17
    batch_size: int = 4
    max_length: int = 512
    vision_slots: int = 4
    vm_profile: TrainingVMProfile = "mixed"
    visual_profile: TrainingVisualProfile = "mixed"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.batch_size < 1:
            raise ValueError("Invalid seed or batch size")

        if self.max_length < 32:
            raise ValueError("Mixed pretraining requries max_length >= 32")

        if not 0 <= self.vision_slots <= 8:
            raise ValueError("vision_slots must be between 0 and 8")

        if self.vm_profile not in ("mixed", "foundation", "intermediate", "advanced"):
            raise ValueError("Invalid training VM profile")

        if self.visual_profile not in (
            "mixed",
            "foundation",
            "intermediate",
            "advanced",
        ):
            raise ValueError("Invalid visual traiing profile")


def is_visual_index(index: int, vision_slots: int) -> bool:
    if index < 0 or not 0 <= vision_slots <= 8:
        raise ValueError("Invalid index or vision_slots")

    # Multiplication by 5 permutes the integers modulo 8
    return (index * 5 + 3) % 8 < vision_slots


def next_multimodal_batch(
    config: MixedStreamConfig, cursor: StreamCursor
) -> tuple[TrainingBatch, StreamCursor]:
    documents: list[MixedDocument] = []

    for index in range(cursor.next_index, cursor.next_index + config.batch_size):
        if is_visual_index(index, config.vision_slots):
            document = generate_visual_document(
                seed=config.seed,
                split="train",
                index=index,
                profile=config.visual_profile,
            )
        else:
            document = generate_vm_document(
                seed=config.seed,
                split="train",
                index=index,
                profile=config.vm_profile,
                max_payload_bytes=config.max_length - 1,
            )

        documents.append(document)

    batch = pack_mixed_documents(documents, max_length=config.max_length)

    text_bytes = 0
    supervised_targets = 0

    for document in documents:
        if hasattr(document, "payload"):
            text_bytes += len(document.payload)  # type: ignore
            supervised_targets += len(document.payload) + 1  # type: ignore
        else:
            text_bytes += len(document.prefix) + len(document.completion)  # type: ignore
            supervised_targets += len(document.completion) + 1  # type: ignore

    advanced = StreamCursor(
        next_index=cursor.next_index + len(documents),
        updates=cursor.updates + 1,
        documents_seen=cursor.documents_seen + len(documents),
        bytes_seen=cursor.bytes_seen + text_bytes,
        targets_seen=cursor.targets_seen + supervised_targets,
    )

    return batch, advanced
