from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import TrainingBatch

AccumulatedStep = Callable[[TrainState, TrainingBatch], tuple[TrainState, jax.Array]]


def split_microbatches(batch: TrainingBatch, microbatch_size: int) -> TrainingBatch:
    """Convert [B, ...] arrays into [A, M, ...] arrays."""
    batch_size = batch.input_ids.shape[0]

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")

    if batch_size % microbatch_size:
        raise ValueError("Batch size must be divisible by microbatch_size")

    count = batch_size // microbatch_size

    return jax.tree_util.tree_map(
        lambda x: x.reshape((count, microbatch_size) + x.shape[1:]), batch
    )


def make_accumulated_step(*, multimodal: bool) -> AccumulatedStep:
    """Create one update function for the selected model interface"""

    def step(
        state: TrainState, microbatches: TrainingBatch
    ) -> tuple[TrainState, jax.Array]:
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)

        initial = (
            zero_grads,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )

        def accumulate(carry, microbatch):
            grad_sum, loss_sum, token_sum = carry

            def numerator(params):
                if multimodal:
                    inputs = (
                        microbatch.input_ids,
                        microbatch.images,
                        microbatch.has_image,
                    )
                else:
                    inputs = (microbatch.input_ids,)

                logits = state.apply_fn({"params": params}, *inputs)
                token_losses = optax.softmax_cross_entropy_with_integer_labels(
                    logits.astype(jnp.float32), microbatch.target_ids
                )
                weights = microbatch.loss_mask.astype(jnp.float32)

                return jnp.sum(token_losses * weights)

            value, grads = jax.value_and_grad(numerator)(state.params)

            grad_sum = jax.tree_util.tree_map(
                lambda old, new: old + new, grad_sum, grads
            )
            tokens = jnp.sum(microbatch.loss_mask).astype(jnp.float32)

            return (grad_sum, loss_sum + value, token_sum + tokens), None

        (grad_sum, loss_sum, token_sum), _ = jax.lax.scan(
            accumulate, initial, microbatches
        )
        denominator = jnp.maximum(token_sum, 1.0)

        gradients = jax.tree_util.tree_map(lambda g: g / denominator, grad_sum)
        updated = state.apply_gradients(grads=gradients)

        return updated, loss_sum / denominator

    return step
