from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from synthesize_vl.data.schema import PretrainingDocument, TrainingBatch
from synthesize_vl.pretraining.visual import VisualDocument
from synthesize_vl.tokenizer import ByteTokenizer

MixedDocument = PretrainingDocument | VisualDocument


def pack_mixed_documents(
    documents: Sequence[MixedDocument], *, max_length: int = 512
) -> TrainingBatch:
    """Pack full-loss VM and image-conditioned visual sequences"""
    if not documents:
        raise ValueError("No documents supplied")

    if max_length < 2:
        raise ValueError("max_length must be >= 2")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    images: list[np.ndarray] = []
    has_image: list[bool] = []

    for document in documents:
        if isinstance(document, VisualDocument):
            if not document.completion:
                raise ValueError("Empty visual completion")

            sequence = (
                [ByteTokenizer.BOS, ByteTokenizer.IMAGE]
                + list(document.prefix)
                + list(document.completion)
                + [ByteTokenizer.EOS]
            )

            # The target array is sequence[1:]
            supervision_start = 1 + len(document.prefix)

            image_array = np.asarray(document.image.convert("RGB"), dtype=np.float32)

            if image_array.shape != (64, 64, 3):
                raise ValueError("Expected a 64x64 RGB image")

            image_array = image_array / 255.0
            visual = True

        else:
            if not document.payload:
                raise ValueError("Empty VM document")

            sequence = (
                [ByteTokenizer.BOS] + list(document.payload) + [ByteTokenizer.EOS]
            )

            # All VM bytes and EOS are supervised
            supervision_start = 0
            image_array = np.zeros((64, 64, 3), dtype=np.float32)
            visual = False

        valid_length = len(sequence) - 1

        if valid_length > max_length:
            raise ValueError(
                f"{document.sample_id} needs {valid_length} positions, but max_length={max_length}"
            )

        padding = max_length - valid_length

        input_rows.append(sequence[:-1] + [ByteTokenizer.PAD] * padding)
        target_rows.append(sequence[1:] + [ByteTokenizer.PAD] * padding)
        mask_rows.append(
            [index >= supervision_start for index in range(valid_length)]
            + [False] * padding
        )

        images.append(image_array)
        has_image.append(visual)

    return TrainingBatch(
        input_ids=jnp.asarray(input_rows, dtype=jnp.int32),
        target_ids=jnp.asarray(target_rows, dtype=jnp.int32),
        loss_mask=jnp.asarray(mask_rows, dtype=jnp.bool_),
        images=jnp.asarray(np.stack(images), dtype=jnp.float32),
        has_image=jnp.asarray(has_image, dtype=jnp.bool_),
    )
