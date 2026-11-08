from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from PIL import Image

from synthesize_vl.data.schema import PretrainingDocument, TrainingBatch
from synthesize_vl.pretraining.visual import VisualDocument
from synthesize_vl.tokenizer import ByteTokenizer

TaskMode = Literal["vm_full", "vm_execute", "visual"]


@dataclass(frozen=True, slots=True)
class JointDocument:
    sample_id: str
    family: str
    mode: TaskMode
    prefix: bytes
    completion: bytes
    image: Image.Image | None = None


def from_vm_document(
    document: PretrainingDocument, *, execute_only: bool
) -> JointDocument:
    """Convert a verified VM output ot a training example."""
    if not execute_only:
        return JointDocument(
            sample_id=document.sample_id,
            family=document.family + "_full",
            mode="vm_full",
            prefix=b"",
            completion=document.payload,
        )

    payload = document.payload

    binary_source, binary_marker, binary_result = payload.partition(b"\nO:\n")

    if binary_marker:
        prefix = binary_source + binary_marker
        completion = binary_result
    else:
        # Text trace:
        # VM ... \nP:\n ... \nT: ...
        # Remove T so the model must calculate the result
        source, trace_marker, trace_and_result = payload.partition(b"\nT:")

        if not trace_marker:
            raise ValueError("Unsupported VM execution document")

        _, final_state_marker, result = trace_and_result.partition(b"\nH:")

        if not final_state_marker:
            raise ValueError("VM trace lacks a verified final state")

        prefix = source + b"\nH:"
        completion = result

    if not completion:
        raise ValueError("VM execution target is empty")

    # Program syntax is ASCII enve when the output is binary
    prefix.decode("ascii")

    return JointDocument(
        sample_id=document.sample_id,
        family=document.family + "_execute",
        mode="vm_execute",
        prefix=prefix,
        completion=completion,
    )


def from_visual_document(document: VisualDocument) -> JointDocument:
    return JointDocument(
        sample_id=document.sample_id,
        family=document.family,
        mode="visual",
        prefix=document.prefix,
        completion=document.completion,
        image=document.image,
    )


def pack_joint_documents(
    documents: Sequence[JointDocument], *, max_length: int
) -> TrainingBatch:
    """
    Pack VM and visual examples into one TrainingBatch.

    Full VM: supervise every payload byte + EOS
    Execute VM: supervise verified result + EOS
    Visual: supervise verified completion + EOS
    """
    if not documents:
        raise ValueError("Emtpy joint document list")

    if max_length < 2:
        raise ValueError("Invalid max_length")

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    masks: list[list[bool]] = []
    pixels: list[np.ndarray] = []
    visual_flags: list[bool] = []

    for document in documents:
        if not document.completion:
            raise ValueError("Empty completion")

        visual = document.image is not None

        if visual != (document.mode == "visual"):
            raise ValueError("Document modality does not match task mode")

        sequence = (
            [ByteTokenizer.BOS]
            + ([ByteTokenizer.IMAGE] if visual else [])
            + list(document.prefix)
            + list(document.completion)
            + [ByteTokenizer.EOS]
        )

        valid_length = len(sequence) - 1

        if valid_length > max_length:
            raise ValueError(
                f"{document.sample_id} requires {valid_length} text positions, capacity={max_length}"
            )

        first_supervised = int(visual) + len(document.prefix)

        padding = max_length - valid_length

        inputs.append(sequence[:-1] + [ByteTokenizer.PAD] * padding)
        targets.append(sequence[1:] + [ByteTokenizer.PAD] * padding)

        masks.append(
            [i >= first_supervised for i in range(valid_length)] + [False] * padding
        )

        if visual:
            array = np.asarray(document.image.convert("RGB"), dtype=np.float32)  # type:ignore

            if array.shape != (64, 64, 3):
                raise ValueError("Expected 64x64 image")

            array = array / 255.0

        else:
            array = np.zeros((64, 64, 3), dtype=np.float32)

        pixels.append(array)
        visual_flags.append(visual)

    return TrainingBatch(
        input_ids=jnp.asarray(inputs, dtype=jnp.int32),
        target_ids=jnp.asarray(targets, dtype=jnp.int32),
        loss_mask=jnp.asarray(masks, dtype=jnp.bool_),
        images=jnp.asarray(np.stack(pixels), dtype=jnp.float32),
        has_image=jnp.asarray(visual_flags, dtype=jnp.bool_),
    )
