from typing import cast

import numpy as np

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.pretraining.config import DataConfig
from synthesize_vl.pretraining.joint_data import (
    JointDocument,
    from_visual_document,
    from_vm_document,
    pack_joint_documents,
)
from synthesize_vl.pretraining.tasks import ARMS, TaskArm, sample_arm
from synthesize_vl.pretraining.visual import (
    VisualFamily,
    VisualProfile,
    generate_visual_document,
)
from synthesize_vl.pretraining.vm_curriculum import VMProfile, generate_vm_document


def generate_arm_document(
    arm: TaskArm,
    *,
    seed: int,
    split: str,
    index: int,
    max_length: int,
    vm_execute_probability: float,
    force_vm_execution: bool = False,
) -> JointDocument:
    """Generate an oracle-verified document for one task arm."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")

    if arm.modality == "vm":
        original = generate_vm_document(
            seed=seed,
            split=split,
            index=index,
            profile=cast(VMProfile, arm.profile),
            max_payload_bytes=max_length - 1,
        )

        mode_rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, index, {"train": 1, "validation": 2, "test": 3}[split], 1902]
            )
        )

        execute_only = force_vm_execution or (
            mode_rng.random() < vm_execute_probability
        )

        return from_vm_document(original, execute_only=execute_only)

    if arm.family is None:
        raise ValueError("Visual task lacks a family")

    visual = generate_visual_document(
        seed=seed,
        split=split,
        index=index,
        profile=cast(VisualProfile, arm.profile),
        family=cast(VisualFamily, arm.family),
    )

    return from_visual_document(visual)


def next_task_batch(
    *, data: DataConfig, max_length: int, cursor: StreamCursor
) -> tuple[TrainingBatch, StreamCursor]:
    documents: list[JointDocument] = []

    for index in range(cursor.next_index, cursor.next_index + data.batch_size):
        arm = sample_arm(seed=data.seed, index=index)

        documents.append(
            generate_arm_document(
                arm,
                seed=data.seed,
                split="train",
                index=index,
                max_length=max_length,
                vm_execute_probability=data.vm_execute_probability,
            )
        )

    batch = pack_joint_documents(documents, max_length=max_length)
    input_bytes = sum(len(doc.prefix) + len(doc.completion) for doc in documents)

    supervised_targets = sum(len(doc.completion) + 1 for doc in documents)

    advanced = StreamCursor(
        next_index=cursor.next_index + len(documents),
        updates=cursor.updates + 1,
        documents_seen=cursor.documents_seen + len(documents),
        bytes_seen=cursor.bytes_seen + input_bytes,
        targets_seen=cursor.targets_seen + supervised_targets,
    )

    return batch, advanced


def make_task_validation(
    *, data: DataConfig, max_length: int
) -> dict[str, TrainingBatch]:
    """Fixed, balanced validation set for every task arm."""

    groups: dict[str, TrainingBatch] = {}

    for arm_index, arm in enumerate(ARMS):
        documents = [
            generate_arm_document(
                arm,
                seed=data.seed,
                split="validation",
                index=300000 + arm_index * 10000 + i,
                max_length=max_length,
                vm_execute_probability=1.0,
                force_vm_execution=True,
            )
            for i in range(data.validation_per_arm)
        ]

        groups[arm.name] = pack_joint_documents(documents, max_length=max_length)

    return groups
