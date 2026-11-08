from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import TrainingBatch


class JointMetrics(NamedTuple):
    loss: jax.Array
    vm_nats: jax.Array
    visual_nats: jax.Array
    vm_targets: jax.Array
    visual_targets: jax.Array


JointStep = Callable[[TrainState, TrainingBatch], tuple[TrainState, JointMetrics]]


def make_joint_step(*, visual_loss_share: float = 0.5) -> JointStep:
    """One optimizer update over accumulated microbatches

    microbatches has leading dimensions:
        [acumulation_steps, microbatch_size, ...]
    """

    if not 0.0 < visual_loss_share < 1.0:
        raise ValueError("visual_loss_share must be strictly between 0 and 1")

    def step(
        state: TrainState, microbatches: TrainingBatch
    ) -> tuple[TrainState, JointMetrics]:
        masks = microbatches.loss_mask

        visual_rows = microbatches.has_image[..., None]

        vm_masks = masks & ~visual_rows
        visual_masks = masks & visual_rows

        vm_count = jnp.sum(vm_masks.astype(jnp.float32))
        visual_count = jnp.sum(visual_masks.astype(jnp.float32))

        # If only one modality occurs, use its full loss
        vm_mass = jnp.where(vm_count > 0, 1.0 - visual_loss_share, 0.0)
        visual_mass = jnp.where(visual_count > 0, visual_loss_share, 0.0)
        total_mass = jnp.maximum(vm_mass + visual_mass, 1e-8)

        vm_coefficient = vm_mass / total_mass / jnp.maximum(vm_count, 1.0)
        visual_coefficient = visual_mass / total_mass / jnp.maximum(visual_count, 1.0)

        initial_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)

        initial = (
            initial_grads,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

        def accumulate(carry, microbatch: TrainingBatch):
            grad_sum, weighted_sum, vm_sum, visual_sum = carry

            def objective(params):
                logits = state.apply_fn(
                    {"params": params},
                    microbatch.input_ids,
                    microbatch.images,
                    microbatch.has_image,
                )

                losses = optax.softmax_cross_entropy_with_integer_labels(
                    logits.astype(jnp.float32), microbatch.target_ids
                )

                vm_mask = microbatch.loss_mask & ~microbatch.has_image[:, None]
                visual_mask = microbatch.loss_mask & microbatch.has_image[:, None]

                vm_numerator = jnp.sum(losses * vm_mask.astype(jnp.float32))
                visual_numerator = jnp.sum(losses * visual_mask.astype(jnp.float32))

                weighted = (
                    vm_coefficient * vm_numerator
                    + visual_coefficient * visual_numerator
                )

                return weighted, (vm_numerator, visual_numerator)

            (weighted, auxiliary), grads = jax.value_and_grad(objective, has_aux=True)(
                state.params
            )

            vm_numerator, visual_numerator = auxiliary

            grad_sum = jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, grads)

            return (
                grad_sum,
                weighted_sum + weighted,
                vm_sum + vm_numerator,
                visual_sum + visual_numerator,
            ), None

        (grads, joint_loss, vm_sum, visual_sum), _ = jax.lax.scan(
            accumulate, initial, microbatches
        )

        updated = state.apply_gradients(grads=grads)

        metrics = JointMetrics(
            loss=joint_loss,
            vm_nats=(vm_sum / jnp.maximum(vm_count, 1.0)),
            visual_nats=(visual_sum / jnp.maximum(visual_count, 1.0)),
            vm_targets=vm_count,
            visual_targets=visual_count,
        )

        return updated, metrics

    return step
