import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


def make_test_batch() -> TrainingBatch:
    red = render_scene([ObjectSpec("digit", "red", "3")], jax.random.key(10))
    blue = render_scene([ObjectSpec("digit", "blue", "7")], jax.random.key(11))
    samples = [
        TrainingSample("math", "Compute: 2+3\nAnswer: ", "5"),
        TrainingSample("digit", "Which digit?\nAnswer: ", "3", red.image),
        TrainingSample("digit", "Which digit?\nAnswer: ", "?", blue.image),
    ]

    return pack_examples(samples, max_text_length=48)


def small_model(
    steps: int = 2,
    position_mode: str = "rope",
    *,
    core_layers: int = 1,
    residual_type: str = "standard",
) -> RecurrentMultimodalCausalLM:
    return RecurrentMultimodalCausalLM(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        max_text_length=48,
        recurrent_steps=steps,
        core_layers=core_layers,
        residual_type=residual_type,
        position_mode=position_mode,
        use_attention_gate=True,
    )


def inputs(batch: TrainingBatch) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (batch.input_ids, batch.images, batch.has_image)


def count_params(params: object) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize("mode", ["rope", "nope", "learned"])
def test_output_shapes(mode: str) -> None:
    batch = make_test_batch()
    model = small_model(position_mode=mode)

    variables = model.init(jax.random.key(0), *inputs(batch))
    logits = model.apply(variables, *inputs(batch))

    assert logits.shape == (3, 48, 260)  # type: ignore
    assert bool(jnp.all(jnp.isfinite(logits)))  # type: ignore


@pytest.mark.parametrize("residual_type", ["standard", "mhc"])
@pytest.mark.parametrize("core_layers", [1, 2])
def test_core_parameters_are_shared(core_layers: int, residual_type: str) -> None:
    batch = make_test_batch()

    one_loop = small_model(steps=1, core_layers=core_layers, residual_type=residual_type)
    four_loops = small_model(steps=4, core_layers=core_layers, residual_type=residual_type)

    variables_1 = one_loop.init(jax.random.key(0), *inputs(batch))
    variables_4 = four_loops.init(jax.random.key(0), *inputs(batch))

    params_1 = variables_1["params"]
    params_4 = variables_4["params"]

    # Extra loop must not create extra parameters
    assert count_params(params_1) == count_params(params_4)

    # The stack has distinct layers, reused across every iteration.
    prefix = "shared_mhc" if residual_type == "mhc" else "shared_block"
    assert {name for name in params_4["core"] if name.startswith(prefix)} == {
        f"{prefix}_{layer}" for layer in range(core_layers)
    }
    assert "input_bridge" in params_4["core"]

    # Additional loops should change the computation.
    logits_1 = one_loop.apply(variables_1, *inputs(batch))
    logits_4 = four_loops.apply(variables_4, *inputs(batch))

    assert not np.allclose(logits_1, logits_4)  # type: ignore


@pytest.mark.parametrize("residual_type", ["standard", "mhc"])
def test_recurrence_preserves_causality(residual_type: str) -> None:
    batch = make_test_batch()
    model = small_model(steps=3, core_layers=2, residual_type=residual_type)

    variables = model.init(jax.random.key(1), *inputs(batch))
    before = model.apply(variables, *inputs(batch))

    # First digit-answer target is supervised here
    first_answer = int(jnp.argmax(batch.loss_mask[1]))

    # Change the answer bytes in the model's input, which is one position after its first target.
    change_at = first_answer + 1

    changed_ids = batch.input_ids.at[1, change_at].set(ord("9"))

    after = model.apply(variables, changed_ids, batch.images, batch.has_image)

    # All earlier outputs must remain unchanged
    np.testing.assert_allclose(before[1, :change_at], after[1, :change_at], rtol=1e-5, atol=1e-5)  # type: ignore
