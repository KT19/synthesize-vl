import numpy as np
from PIL import Image

from synthesize_vl.data.schema import PretrainingDocument, StreamCursor
from synthesize_vl.legacy.joint_stream import next_joint_batch
from synthesize_vl.legacy.multimodal_stream import MixedStreamConfig
from synthesize_vl.pretraining.joint_data import (
    JointDocument,
    from_vm_document,
    pack_joint_documents,
)


def test_vm_trace_to_execution_target() -> None:
    original = PretrainingDocument(
        sample_id="vm:test:001",
        split="test",
        family="vm_foundation",
        payload=(b"VM r=2,0\nP:\nADD0,3\nEMIT0\nT:5,0;5,0\nH:5,0\nO:05\n"),
    )

    converted = from_vm_document(original, execute_only=True)

    assert converted.mode == "vm_execute"
    assert converted.prefix.endswith(b"\nH:")
    assert converted.completion == b"5,0\nO:05\n"

    assert b"\nT:" not in (converted.prefix + converted.completion)


def test_binary_vm_execution_target() -> None:
    original = PretrainingDocument(
        sample_id="vm:test:002",
        split="test",
        family="vm_foundation",
        payload=(b"VMr=0,0\nP:\nEMIT0\nO:\n" + bytes([0, 128, 255])),
    )

    converted = from_vm_document(original, execute_only=True)

    assert converted.prefix.endswith(b"O:\n")
    assert converted.completion == bytes([0, 128, 255])

    full = from_vm_document(original, execute_only=False)

    assert full.mode == "vm_full"
    assert full.prefix == b""
    assert full.completion == original.payload


def test_binary_output_can_contain_trace_markers() -> None:
    raw_output = b"\nT:\nH:\x00"
    original = PretrainingDocument(
        sample_id="vm:test:003",
        split="test",
        family="vm_foundation",
        payload=b"VM r=0,0\nP:\nEMIT0\nO:\n" + raw_output,
    )

    converted = from_vm_document(original, execute_only=True)

    assert converted.prefix == b"VM r=0,0\nP:\nEMIT0\nO:\n"
    assert converted.completion == raw_output


def sample_joint_documents() -> list[JointDocument]:
    return [
        JointDocument(
            sample_id="unit:vm",
            family="vm_unit",
            mode="vm_full",
            prefix=b"",
            completion=b"ABC",
        ),
        JointDocument(
            sample_id="unit:visual",
            family="visual_unit",
            mode="visual",
            prefix=b"count=",
            completion=b"2\n",
            image=Image.new(
                "RGB",
                (64, 64),
                color=(220, 50, 65),
            ),
        ),
    ]


def test_joint_packing_alignment() -> None:
    batch = pack_joint_documents(sample_joint_documents(), max_length=32)

    assert batch.input_ids.shape == (2, 32)
    assert batch.images.shape == (2, 64, 64, 3)

    np.testing.assert_array_equal(batch.has_image, [False, True])

    # VM: A,B,C and EOS
    assert int(batch.loss_mask[0].sum()) == 4
    # Visual: "2\n" and EOS
    assert int(batch.loss_mask[1].sum()) == 3


def test_joint_stream_reproducibility() -> None:
    config = MixedStreamConfig(
        seed=17,
        batch_size=8,
        max_length=512,
        vision_slots=4,
        vm_profile="foundation",
        visual_profile="foundation",
    )

    first, advanced = next_joint_batch(config, StreamCursor())
    replay, replay_cursor = next_joint_batch(config, StreamCursor())

    assert advanced == replay_cursor

    np.testing.assert_array_equal(first.input_ids, replay.input_ids)
    np.testing.assert_array_equal(first.images, replay.images)

    assert int(first.has_image.sum()) == 4
    assert advanced.documents_seen == 8
    assert advanced.next_index == 8

    assert advanced.targets_seen == int(first.loss_mask.sum())
