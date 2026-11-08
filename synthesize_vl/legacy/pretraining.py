from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax

from synthesize_vl.data.schema import PretrainingDocument, TrainingBatch
from synthesize_vl.tokenizer import ByteTokenizer


def pack_pretraining_documents(
    documents: Sequence[PretrainingDocument], *, max_length: int = 128
) -> TrainingBatch:
    """Create answer-independent full-sequence LM supervision.

    One document occupies one batch row.

    Input:  [BOS, byte0, ..., byteN-1]
    Target: [byte0, ..., byteN-1, EOS]

    Padding targets are ignored in the loss
    """
    if not documents:
        raise ValueError("At least one document is required")

    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []

    for document in documents:
        if not document.payload:
            raise ValueError(f"Empty document: {document.sample_id}")

        # Each byte and the final EOS need a target slot.
        required_length = len(document.payload) + 1

        if required_length > max_length:
            raise ValueError(
                f"Document {document.sample_id} needs {required_length} slots: capacity is {max_length}"
            )

        sequence = [ByteTokenizer.BOS] + list(document.payload) + [ByteTokenizer.EOS]

        inputs = sequence[:-1]
        targets = sequence[1:]

        padding = max_length - required_length

        input_rows.append(inputs + [ByteTokenizer.PAD] * padding)
        target_rows.append(targets + [ByteTokenizer.PAD] * padding)
        mask_rows.append([True] * required_length + [False] * padding)

    batch_size = len(documents)

    return TrainingBatch(
        input_ids=jnp.asarray(input_rows, dtype=jnp.int32),
        target_ids=jnp.asarray(target_rows, dtype=jnp.int32),
        loss_mask=jnp.asarray(mask_rows, dtype=jnp.bool_),
        images=jnp.zeros((batch_size, 64, 64, 3), dtype=jnp.float32),
        has_image=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def bits_per_byte(logits: jax.Array, batch: TrainingBatch) -> jax.Array:
    """
    Average predictive cross-entropy in bits per byte

    Excludes EOS and padding, unlike full LM training loss.
    """
    if logits.ndim != 3 or (logits.shape[:2] != batch.target_ids.shape):
        raise ValueError("Logits and targets have incompatible shapes")

    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.astype(jnp.float32), labels=batch.target_ids
    )

    is_byte = batch.target_ids < 256

    weights = (batch.loss_mask & is_byte).astype(jnp.float32)

    total_nats = jnp.sum(token_losses * weights)
    byte_count = jnp.maximum(jnp.sum(weights), 1.0)

    return total_nats / byte_count / jnp.log(2.0)
