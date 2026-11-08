from dataclasses import dataclass

import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")

        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def create_train_state(
    model: nn.Module,
    example_inputs: tuple[jax.Array, ...],
    *,
    seed: int,
    config: OptimizerConfig,
) -> TrainState:
    """Initialize a Linen module and an AdamW optimizer."""
    variables = model.init(jax.random.key(seed), *example_inputs)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=config.learning_rate, weight_decay=config.weight_decay
        ),
    )

    return TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer
    )
