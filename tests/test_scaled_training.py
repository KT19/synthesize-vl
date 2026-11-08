import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.training.accumulation import (
    make_accumulated_step,
    split_microbatches,
)
from synthesize_vl.training.optimizers import (
    AdvancedOptimizerConfig,
    create_advanced_state,
    muon_parameter_specs,
)
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import train_step


def make_setup():
    batch = pack_examples(
        [
            TrainingSample("math", "Q: 1+1=", "2"),
            TrainingSample("math", "Q: 2+2=", "4"),
            TrainingSample("math", "Q: 3+3=", "6"),
            TrainingSample("math", "Q: 10+1=", "11"),
        ],
        max_text_length=32,
    )

    model = CausalLM(
        hidden_size=16, num_heads=2, num_layers=1, intermediate_size=32, max_length=32
    )

    return model, batch


def test_muon_parameter_selection() -> None:
    params = {
        "block_0": {
            "feedforward": {"gate_proj": {"kernel": jnp.ones((16, 32))}},
            "attention": {
                "out_proj": {"kernel": jnp.ones((16, 16))},
                "q_proj": {"kernel": jnp.ones((16, 2, 8))},
            },
        },
        "text_embedding": {"token_embedding": {"embedding": jnp.ones((260, 16))}},
    }

    specs = muon_parameter_specs(params)

    assert specs["block_0"]["feedforward"]["gate_proj"]["kernel"] is not None  # type: ignore
    assert specs["block_0"]["attention"]["out_proj"]["kernel"] is not None  # type: ignore
    assert specs["block_0"]["attention"]["q_proj"]["kernel"] is None  # type: ignore
    assert specs["text_embedding"]["token_embedding"]["embedding"] is None  # type: ignore


def test_microbatch_split() -> None:
    _, batch = make_setup()

    micro = split_microbatches(batch, 2)

    assert micro.input_ids.shape == (2, 2, 32)
    assert micro.images.shape == (2, 2, 64, 64, 3)

    np.testing.assert_array_equal(micro.input_ids.reshape(4, 32), batch.input_ids)

    with pytest.raises(ValueError):
        split_microbatches(batch, 3)


def test_accumulation_matches_full_batch() -> None:
    model, batch = make_setup()

    state = create_train_state(
        model, (batch.input_ids,), seed=0, config=OptimizerConfig()
    )
    full_state, full_loss = train_step(state, batch)

    micro = split_microbatches(batch, 2)

    accumulated_step = jax.jit(make_accumulated_step(multimodal=False))
    accumulated_state, accumulated_loss = accumulated_step(state, micro)

    np.testing.assert_allclose(full_loss, accumulated_loss, rtol=1e-5, atol=1e-5)

    full_leaves = jax.tree_util.tree_leaves(full_state.params)
    accumulated_leaves = jax.tree_util.tree_leaves(accumulated_state.params)

    for expected, actual in zip(full_leaves, accumulated_leaves):
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-4)


def test_muon_adamw_update() -> None:
    model, batch = make_setup()

    state = create_advanced_state(
        model,
        (batch.input_ids,),
        config=AdvancedOptimizerConfig(total_steps=10, warmup_steps=0),
    )

    step_fn = jax.jit(make_accumulated_step(multimodal=False))

    updated, loss = step_fn(state, split_microbatches(batch, 2))

    assert int(updated.step) == 1
    assert bool(jnp.isfinite(loss))

    assert any(
        not bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(updated.params),
        )
    )
