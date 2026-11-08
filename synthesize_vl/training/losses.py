import jax
import jax.numpy as jnp
import optax

from synthesize_vl.data.schema import TrainingBatch


def masked_lm_loss(logits: jax.Array, batch: TrainingBatch) -> jax.Array:
    if logits.ndim != 3 or logits.shape[:2] != batch.target_ids.shape:
        raise ValueError("Expected logits [B, T, V] matching target_ids [B, T]")

    per_token = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.astype(jnp.float32), labels=batch.target_ids
    )

    weights = batch.loss_mask.astype(jnp.float32)

    return jnp.sum(per_token * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def masked_token_accuracy(
    logits: jax.Array,
    batch: TrainingBatch,
) -> jax.Array:
    if logits.ndim != 3 or logits.shape[:2] != batch.target_ids.shape:
        raise ValueError("Expected logits [B, T, V] matching target_ids [B, T]")

    predictions = jnp.argmax(logits, axis=-1)

    correct = (predictions == batch.target_ids) & batch.loss_mask

    denominator = jnp.maximum(jnp.sum(batch.loss_mask), 1)

    return jnp.sum(correct).astype(jnp.float32) / denominator
