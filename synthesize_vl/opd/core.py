from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from synthesize_vl.opd.data import OPDPrompts, OPDRollouts
from synthesize_vl.tokenizer import ByteTokenizer

FORBIDDEN = jnp.asarray([ByteTokenizer.BOS, ByteTokenizer.PAD, ByteTokenizer.IMAGE], dtype=jnp.int32)


def allowed_logits(logits: jax.Array) -> jax.Array:
    return logits.at[..., FORBIDDEN].set(-1e9)


# Shapes: B = batch size, T = padded text context length, R = rollout length, V = vocabulary size
def reverse_kl(student_logits: jax.Array, teacher_logits: jax.Array, *, temperature: float = 1.0) -> jax.Array:
    """
    Exact KL(student || teacher) over the allowed vocabulary.

    Inputs have shape (..., V); the result has shape (...).
    During an OPD update, these are (B, R, V) and (B, R), respectively.
    """
    student = allowed_logits(student_logits.astype(jnp.float32)) / temperature  # (..., V)
    teacher = allowed_logits(teacher_logits.astype(jnp.float32)) / temperature  # (..., V)

    log_student = jax.nn.log_softmax(student, axis=-1)
    log_teacher = jax.lax.stop_gradient(jax.nn.log_softmax(teacher, axis=-1))

    probabilities = jnp.exp(log_student)

    kl = jnp.sum(probabilities * (log_student - log_teacher), axis=-1)

    return kl * (temperature**2)


def make_student_sampler(model: nn.Module, *, new_tokens: int, temperature: float) -> Callable:
    if new_tokens < 1 or temperature <= 0:
        raise ValueError("Invalid sampling settings")

    def sample(params: Any, batch: OPDPrompts, key: jax.Array) -> OPDRollouts:
        batch_size = batch.input_ids.shape[0]  # Python int B; input_ids: (B, T)
        rows = jnp.arange(batch_size)  # (B,)

        def sample_step(carry, step):
            ids, active, rng_key = carry  # (B, T), (B,), ()

            rng_key, subkey = jax.random.split(rng_key)

            # full-prefix recurrent model inference
            logits = model.apply(
                {"params": params},
                ids,  # (B, T)
                batch.images,  # (B, H, W, C)
                batch.has_image,  # (B,)
            )  # (B, T, V)

            positions = batch.prefix_lengths - 1 + step  # (B,)
            current = allowed_logits(logits[rows, positions, :])  # type: ignore  # (B, V)

            sampled = jax.random.categorical(subkey, current / temperature, axis=-1).astype(jnp.int32)  # (B,)

            # After EOS, mark all later slots invalid
            tokens = jnp.where(active, sampled, ByteTokenizer.PAD)  # (B,)

            # A sampled token is only written into the NEXT input position
            write_positions = batch.prefix_lengths + step  # (B,)
            ids = ids.at[rows, write_positions].set(tokens)  # (B, T)

            next_active = active & (tokens != ByteTokenizer.EOS)  # (B,)

            return (ids, next_active, rng_key), (tokens, active)  # Carry: (B, T), (B,), (); outputs: (B,), (B,)

        def skip_step(carry, step):
            active = carry[1]  # (B,)
            tokens = jnp.full((batch_size,), ByteTokenizer.PAD, dtype=jnp.int32)  # (B,)

            return carry, (tokens, active)

        def body(carry, step):
            return jax.lax.cond(jnp.any(carry[1]), sample_step, skip_step, carry, step)

        initial = (batch.input_ids, jnp.ones((batch_size,), dtype=jnp.bool_), key)  # (B, T), (B,), ()
        (final_ids, _, _), (tokens, mask) = jax.lax.scan(
            body,
            initial,
            jnp.arange(new_tokens),  # Scan steps: (R,)
        )  # final_ids: (B, T); tokens and mask: (R, B)

        return OPDRollouts(
            input_ids=final_ids,  # (B, T)
            prefix_lengths=batch.prefix_lengths,  # (B,)
            images=batch.images,
            has_image=batch.has_image,
            sampled_tokens=jnp.swapaxes(tokens, 0, 1),  # (R, B) -> (B, R)
            valid_mask=jnp.swapaxes(mask, 0, 1),  # (R, B) -> (B, R)
        )

    return sample


class OPDMetrics(NamedTuple):
    loss: jax.Array
    text_kl: jax.Array
    visual_kl: jax.Array
    text_tokens: jax.Array
    visual_tokens: jax.Array
    eos_fraction: jax.Array


def make_opd_update(
    student_model: nn.Module,
    teacher_model: nn.Module,
    *,
    distillation_temperature: float = 1.0,
    visual_loss_share: float = 0.5,
) -> Callable:
    """Update student using teacher logits at student prefixes."""

    def update(state: TrainState, teacher_params: Any, rollouts: OPDRollouts) -> tuple[TrainState, OPDMetrics]:
        batch_size, count = rollouts.sampled_tokens.shape  # Python ints B, R

        rows = jnp.arange(batch_size)[:, None]  # (B, 1)
        positions = rollouts.prefix_lengths[:, None] - 1 + jnp.arange(count)[None, :]  # (B, 1) + (1, R) -> (B, R)

        # Teacher is frozen: its parameters are not part of the differentiated argument
        teacher_full = teacher_model.apply(
            {"params": teacher_params},
            rollouts.input_ids,  # (B, T)
            rollouts.images,  # (B, H, W, C)
            rollouts.has_image,  # (B,)
        )

        teacher_logits = jax.lax.stop_gradient(teacher_full[rows, positions, :])  # type: ignore  # (B, R, V)
        valid = rollouts.valid_mask  # (B, R)

        text_mask = valid & ~rollouts.has_image[:, None]  # (B, R); modality flag broadcasts from (B, 1)
        visual_mask = valid & rollouts.has_image[:, None]  # (B, R); modality flag broadcasts from (B, 1)

        text_count = jnp.sum(text_mask.astype(jnp.float32))
        visual_count = jnp.sum(visual_mask.astype(jnp.float32))

        text_mass = jnp.where(text_count > 0, 1.0 - visual_loss_share, 0.0)
        visual_mass = jnp.where(visual_count > 0, visual_loss_share, 0.0)

        total_mass = jnp.maximum(text_mass + visual_mass, 1e-8)

        def objective(student_params):
            student_full = student_model.apply(
                {"params": student_params},
                rollouts.input_ids,
                rollouts.images,
                rollouts.has_image,
            )  # (B, T, V)

            student_logits = student_full[rows, positions, :]  # type: ignore  # (B, R, V)
            token_kl = reverse_kl(
                student_logits=student_logits,  # (B, R, V)
                teacher_logits=teacher_logits,  # (B, R, V)
                temperature=distillation_temperature,
            )  # (B, R), one KL value per rollout position

            text_sum = jnp.sum(token_kl * text_mask.astype(jnp.float32))
            visual_sum = jnp.sum(token_kl * visual_mask.astype(jnp.float32))

            text_mean = text_sum / jnp.maximum(text_count, 1.0)
            visual_mean = visual_sum / jnp.maximum(visual_count, 1.0)

            loss = (text_mass / total_mass) * text_mean + (visual_mass / total_mass) * visual_mean

            return loss, (text_mean, visual_mean)

        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)

        updated = state.apply_gradients(grads=gradients)

        text_kl, visual_kl = auxiliary

        emitted_eos = (rollouts.sampled_tokens == ByteTokenizer.EOS) & rollouts.valid_mask  # (B, R)

        eos_fraction = jnp.mean(jnp.any(emitted_eos, axis=-1).astype(jnp.float32))

        return updated, OPDMetrics(
            loss=loss,
            text_kl=text_kl,
            visual_kl=visual_kl,
            text_tokens=text_count,
            visual_tokens=visual_count,
            eos_fraction=eos_fraction,
        )

    return update
