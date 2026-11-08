from dataclasses import dataclass

import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


@dataclass(frozen=True, slots=True)
class AdvancedOptimizerConfig:
    muon_lr: float = 3e-3
    adamw_lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    total_steps: int = 100
    warmup_steps: int = 5

    def __post_init__(self) -> None:
        if self.muon_lr <= 0 or self.adamw_lr <= 0:
            raise ValueError("Learning rates must bepositive")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("Invalid warmup or total steps")


def muon_parameter_specs(params: object) -> object:
    """Select only known 2D hidden-layer matricies for Muon."""

    def select(path: tuple, param: jax.Array) -> object:
        names = tuple(
            str(getattr(entry, "key", getattr(entry, "name", ""))) for entry in path
        )

        is_feedforward = "feedforward" in names and names[-2:] in (
            ("gate_proj", "kernel"),
            ("up_proj", "kernel"),
            ("down_proj", "kernel"),
        )
        is_attention_output = "attention" in names and names[-2:] == (
            "out_proj",
            "kernel",
        )

        if param.ndim == 2 and (is_feedforward or is_attention_output):
            return optax.contrib.MuonDimensionNumbers()

        return None

    return jax.tree_util.tree_map_with_path(select, params)


def learning_rate_schedule(
    peak: float, config: AdvancedOptimizerConfig
) -> optax.Schedule:
    if config.warmup_steps == 0:
        return optax.cosine_decay_schedule(
            init_value=peak, decay_steps=config.total_steps, alpha=0.1
        )

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=peak * 0.1,
    )


def make_optimizer(
    params: object, config: AdvancedOptimizerConfig
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.contrib.muon(
            learning_rate=learning_rate_schedule(config.muon_lr, config),
            adam_learning_rate=learning_rate_schedule(config.adamw_lr, config),
            adam_weight_decay=config.weight_decay,
            muon_weight_dimension_numbers=muon_parameter_specs(params=params),  # type: ignore
        ),
    )


def create_advanced_state(
    model: nn.Module,
    example_inputs: tuple[jax.Array, ...],
    *,
    seed: int = 0,
    config: AdvancedOptimizerConfig,
) -> TrainState:
    variables = model.init(jax.random.key(seed), *example_inputs)

    params = variables["params"]

    return TrainState.create(
        apply_fn=model.apply, params=params, tx=make_optimizer(params, config)
    )
