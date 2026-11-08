import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.attention_masks import make_hybrid_local_global_mask
from synthesize_vl.models.multimodal import make_prefix_causal_mask
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


def example_tokens() -> tuple[jax.Array, jax.Array]:
    tokens = jnp.array(
        [
            [256, 65, 66, 67, 68, 69, 70, 258],
            [256, 65, 66, 67, 68, 69, 70, 258],
        ],
        dtype=jnp.int32,
    )

    has_image = jnp.array([True, False])

    return tokens, has_image


def make_test_batch() -> TrainingBatch:
    scene_3 = render_scene([ObjectSpec("digit", "red", "3")], jax.random.key(10))
    scene_7 = render_scene([ObjectSpec("digit", "blue", "7")], jax.random.key(11))

    return pack_examples(
        [
            TrainingSample("math", "Q: 1+1=", "2"),
            TrainingSample("vision", "Which digit?\nAnswer: ", "3", scene_3.image),
            TrainingSample("vision", "Which digit?\nAnswer: ", "7", scene_7.image),
        ],
        max_text_length=32,
    )


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (batch.input_ids, batch.images, batch.has_image)


def small_model(
    core_attention: str = "hybrid", residual_type: str = "mhc", recurrent_steps: int = 2
) -> RecurrentMultimodalCausalLM:
    return RecurrentMultimodalCausalLM(
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        max_text_length=32,
        recurrent_steps=recurrent_steps,
        core_layers=2,
        position_mode="rope",
        use_attention_gate=True,
        residual_type=residual_type,
        num_streams=2,
        sinkhorn_iterations=10,
        core_attention=core_attention,
        local_window=4,
        global_text_tokens=1,
    )


def test_hybrid_visibility() -> None:
    tokens, has_image = example_tokens()

    mask = make_hybrid_local_global_mask(tokens, has_image, n_visual_tokens=2, local_window=2, global_text_tokens=1)

    assert mask.shape == (2, 1, 10, 10)

    # All valid image patches communicate
    assert bool(mask[0, 0, 0, 1])
    assert bool(mask[0, 0, 1, 0])

    # Image queries cannot see text
    assert not bool(mask[0, 0, 0, 2])

    # Text queries can see both image patches
    assert bool(mask[0, 0, 8, 0])
    assert bool(mask[0, 0, 8, 1])

    # The first text position is a global key
    assert bool(mask[0, 0, 8, 2])

    # Window of 2 means current and previous tokens
    assert bool(mask[0, 0, 8, 8])
    assert bool(mask[0, 0, 8, 7])

    # A distant, non-global text token is blocked
    assert not bool(mask[0, 0, 8, 4])
    assert not bool(mask[0, 0, 7, 8])


def test_padding_and_text_only_behavior() -> None:
    tokens, has_image = example_tokens()

    mask = make_hybrid_local_global_mask(tokens, has_image, n_visual_tokens=2, local_window=2)

    # Image slots are inactive for the text-only sample
    assert not bool(mask[1, 0, 8, 0])
    assert not bool(mask[1, 0, 8, 1])

    # Image queries do not receive text information
    assert not bool(mask[0, 0, 0, 8])

    # Padding must not be visible as a key to valid text
    assert not bool(mask[0, 0, 8, 9])

    # Every inactive query retains its safef self-entry
    assert bool(jnp.all(jnp.any(mask, axis=-1)))


def test_hybrid_is_subset_of_full() -> None:
    tokens, has_image = example_tokens()

    full = make_prefix_causal_mask(tokens, has_image, n_visual_tokens=2)

    hybrid = make_hybrid_local_global_mask(tokens, has_image, n_visual_tokens=2, local_window=2, global_text_tokens=1)

    assert bool(jnp.all(~hybrid | full))

    # With a sufficiently larg local window, the hybrid mask is identical to full attention
    unrestricted = make_hybrid_local_global_mask(
        tokens, has_image, n_visual_tokens=2, local_window=64, global_text_tokens=0
    )

    np.testing.assert_array_equal(unrestricted, full)


@pytest.mark.parametrize("residual_type", ["standard", "mhc"])
def test_both_recurrent_cores_support_hybrid(residual_type: str) -> None:
    batch = make_test_batch()

    model = small_model(residual_type=residual_type)
    variables = model.init(jax.random.key(0), *model_inputs(batch))

    logits = model.apply(variables, *model_inputs(batch))

    assert logits.shape == (3, 32, 260)  # type: ignore
    assert bool(jnp.all(jnp.isfinite(logits)))  # type: ignore


def test_full_and_hybrid_share_parameter_structure() -> None:
    batch = make_test_batch()

    full = small_model(core_attention="full")
    hybrid = small_model(core_attention="hybrid")

    full_vars = full.init(jax.random.key(1), *model_inputs(batch))
    hybrid_vars = hybrid.init(jax.random.key(1), *model_inputs(batch))

    same = jax.tree_util.tree_map(jnp.array_equal, full_vars["params"], hybrid_vars["params"])

    assert all(bool(x) for x in jax.tree_util.tree_leaves(same))
