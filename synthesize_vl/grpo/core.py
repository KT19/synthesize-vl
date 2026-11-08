from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from synthesize_vl.opd.core import allowed_logits
from synthesize_vl.opd.data import OPDRollouts


# Shapes: N = prompts, G = group size, B = N * G responses, T = padded context,
# R = rollout length, V = vocabulary size.
def rollout_log_probs(model: nn.Module, params: Any, rollouts: OPDRollouts) -> jax.Array:
    """Log probability of each sampled token

    Returns (B, R), with padding positions set to zero.
    The logits at prefix_length - 1 + t predict
    the sampled token at continuation position t
    """
    logits = model.apply(
        {"params": params},
        rollouts.input_ids,  # (B, T)
        rollouts.images,  # (B, H, W, C)
        rollouts.has_image,  # (B,)
    )  # (B, T, V)

    batch_size, length = rollouts.sampled_tokens.shape  # B, R

    rows = jnp.arange(batch_size)[:, None]  # (B, 1)
    positions = rollouts.prefix_lengths[:, None] - 1 + jnp.arange(length)[None, :]  # (B, 1) + (1, R) -> (B, R)

    selected_logits = allowed_logits(logits[rows, positions, :].astype(jnp.float32))  # type: ignore  # (B, R, V)
    log_probs = jax.nn.log_softmax(selected_logits, axis=-1)  # (B, R, V)

    sampled = jnp.take_along_axis(
        log_probs,
        rollouts.sampled_tokens[..., None],  # (B, R, 1)
        axis=-1,
    )[..., 0]  # (B, R, 1) -> (B, R)

    return jnp.where(rollouts.valid_mask, sampled, 0.0)  # (B, R)


def grpo_clipped_loss(
    new_log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    valid_mask: jax.Array,
    *,
    clip_epsilon: float,
) -> tuple[jax.Array, jax.Array]:
    """Per-response clipped GRPO surrogate

    Inputs: new/old_log_probs and valid_mask (B, R), advantages (B,).
    Returns: response_losses (B,), clip_fraction.
    """
    log_ratio = jnp.clip(
        new_log_probs.astype(jnp.float32) - jax.lax.stop_gradient(old_log_probs.astype(jnp.float32)), -20.0, 20.0
    )  # (B, R)
    ratio = jnp.exp(log_ratio)  # (B, R)

    weighted = ratio * advantages[:, None]  # (B, R)
    clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[:, None]  # (B, R)

    surrogate = jnp.minimum(weighted, clipped)  # (B, R)

    masked = jnp.where(valid_mask, surrogate, 0.0)  # (B, R)
    max_length = new_log_probs.shape[1]  # R

    per_response_loss = -jnp.sum(masked, axis=-1) / max_length  # (B,)

    clip_fraction = jnp.sum(
        jnp.where(valid_mask, (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32), 0.0)
    ) / jnp.maximum(jnp.sum(valid_mask), 1)

    return per_response_loss, clip_fraction


class GRPOMetrics(NamedTuple):
    # Every metric below is a scalar ().
    loss: jax.Array
    text_loss: jax.Array
    visual_loss: jax.Array
    clip_fraction: jax.Array
    active_responses: jax.Array


def make_grpo_update(
    model: nn.Module, *, group_size: int, clip_epsilon: float = 0.2, visual_loss_share: float = 0.5
) -> Callable:
    def update(
        state: TrainState, rollouts: OPDRollouts, old_log_probs: jax.Array, advantages: jax.Array
    ) -> tuple[TrainState, GRPOMetrics]:
        # old_log_probs: (B, R); advantages: (B,); valid_mask: (B, R)
        batch_size = rollouts.sampled_tokens.shape[0]  # B

        if batch_size % group_size:
            raise ValueError("Incomplete rollout group")

        num_prompts = batch_size // group_size  #  B // G -> N

        def objective(params):
            new_log_probs = rollout_log_probs(model, params, rollouts)  # (B, R)
            response_losses, clip_fraction = grpo_clipped_loss(
                new_log_probs, old_log_probs, advantages, rollouts.valid_mask, clip_epsilon=clip_epsilon
            )  # response_losses: (B,); clip_fraction: ()

            # Average sampled responses within each prompt group.
            problem_losses = jnp.mean(
                response_losses.reshape(num_prompts, group_size), axis=1
            )  # (B,) -> (N, G) -> (N,)
            visual_problems = rollouts.has_image.reshape(num_prompts, group_size)[:, 0]  # (B,) -> (N, G) -> (N,)

            text_problems = ~visual_problems  # (N,)

            text_count = jnp.sum(text_problems.astype(jnp.float32))
            visual_count = jnp.sum(visual_problems.astype(jnp.float32))

            text_mean = jnp.sum(problem_losses * text_problems.astype(jnp.float32)) / jnp.maximum(text_count, 1.0)
            visual_mean = jnp.sum(problem_losses * visual_problems.astype(jnp.float32)) / jnp.maximum(visual_count, 1.0)

            text_mass = jnp.where(text_count > 0, 1.0 - visual_loss_share, 0.0)
            visual_mass = jnp.where(visual_count > 0, visual_loss_share, 0.0)
            total_mass = jnp.maximum(text_mass + visual_mass, 1e-8)

            loss = text_mass / total_mass * text_mean + visual_mass / total_mass * visual_mean

            active = jnp.sum(jnp.abs(advantages) > 1e-7)  # ()

            return loss, (text_mean, visual_mean, clip_fraction, active)

        (loss, auxiliary), grads = jax.value_and_grad(objective, has_aux=True)(state.params)

        updated = state.apply_gradients(grads=grads)

        return updated, GRPOMetrics(
            loss=loss,
            text_loss=auxiliary[0],
            visual_loss=auxiliary[1],
            clip_fraction=auxiliary[2],
            active_responses=auxiliary[3],
        )

    return update
