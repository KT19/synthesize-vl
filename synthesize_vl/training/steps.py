from typing import Any

import jax
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.training.losses import masked_lm_loss, masked_token_accuracy


@jax.jit
def train_step(
    state: TrainState,
    batch: TrainingBatch,
    model_inputs: tuple[jax.Array, ...] | None = None,
) -> tuple[TrainState, jax.Array]:
    inputs = (batch.input_ids,) if model_inputs is None else model_inputs

    def loss_fn(params: Any) -> jax.Array:
        logits = state.apply_fn({"params": params}, *inputs)

        return masked_lm_loss(logits, batch)

    loss, gradients = jax.value_and_grad(loss_fn)(state.params)

    new_state = state.apply_gradients(grads=gradients)

    return new_state, loss


@jax.jit
def eval_step(
    state: TrainState,
    batch: TrainingBatch,
    model_inputs: tuple[jax.Array, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return loss and teacher-forced token accuracy"""
    inputs = (batch.input_ids,) if model_inputs is None else model_inputs

    logits = state.apply_fn({"params": state.params}, *inputs)

    loss = masked_lm_loss(logits, batch)
    accuracy = masked_token_accuracy(logits, batch)

    return loss, accuracy
