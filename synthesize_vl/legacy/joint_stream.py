from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.legacy.multimodal_stream import MixedStreamConfig, is_visual_index
from synthesize_vl.pretraining.joint_data import (
    JointDocument,
    from_visual_document,
    from_vm_document,
    pack_joint_documents,
)
from synthesize_vl.pretraining.visual import generate_visual_document
from synthesize_vl.pretraining.vm_curriculum import generate_vm_document


def make_joint_document(
    config: MixedStreamConfig, *, index: int, split: str = "train"
) -> JointDocument:
    if split not in ("train", "validation", "test"):
        raise ValueError("Invalid data split")

    if is_visual_index(index, config.vision_slots):
        visual = generate_visual_document(
            seed=config.seed, split=split, index=index, profile=config.visual_profile
        )

        return from_visual_document(visual)

    vm = generate_vm_document(
        seed=config.seed,
        split=split,
        index=index,
        profile=config.vm_profile,
        max_payload_bytes=config.max_length - 1,
    )

    # Alternate full-sequence modeling
    return from_vm_document(vm, execute_only=(index % 2 == 1))


def next_joint_batch(
    config: MixedStreamConfig, cursor: StreamCursor
) -> tuple[TrainingBatch, StreamCursor]:
    documents = [
        make_joint_document(config, index=index)
        for index in range(cursor.next_index, cursor.next_index + config.batch_size)
    ]

    batch = pack_joint_documents(documents, max_length=config.max_length)

    bytes_seen = sum(len(doc.prefix) + len(doc.completion) for doc in documents)

    targets_seen = sum(len(doc.completion) + 1 for doc in documents)

    advanced = StreamCursor(
        next_index=cursor.next_index + len(documents),
        updates=cursor.updates + 1,
        documents_seen=cursor.documents_seen + len(documents),
        bytes_seen=cursor.bytes_seen + bytes_seen,
        targets_seen=cursor.targets_seen + targets_seen,
    )

    return batch, advanced
