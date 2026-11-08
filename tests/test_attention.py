import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.models.attention import (
    MultiHeadAttention,
    make_rope_positions,
    scaled_dot_product_attention,
)
from synthesize_vl.models.multimodal import make_prefix_causal_mask


def test_rope_positions():
    positions = make_rope_positions(text_length=3, n_visual_tokens=2)

    assert positions.tolist() == [0, 0, 0, 1, 2]


def test_attention_probability_masking():
    q = jnp.ones((1, 5, 1, 4))
    token_ids = jnp.array([[256, 65, 66]], dtype=jnp.int32)

    mask = make_prefix_causal_mask(token_ids, jnp.array([True]), n_visual_tokens=2)
    context, weights = scaled_dot_product_attention(q, q, q, mask)

    assert context.shape == (1, 5, 1, 4)
    assert weights.shape == (1, 1, 5, 5)

    # Image queries may access other image patches
    assert float(weights[0, 0, 0, 1]) > 0

    # Image queries canno access text
    assert float(weights[0, 0, 0, 2]) == 0

    # Text queries can access image patches
    assert float(weights[0, 0, 2, 0]) > 0

    # Text queries cannot access future text
    assert float(weights[0, 0, 2, 3]) == 0

    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-6)


def test_fully_masked_rows_are_safe():
    x = jnp.ones((1, 2, 1, 4))
    mask = jnp.zeros((1, 1, 2, 2), dtype=bool)

    context, weights = scaled_dot_product_attention(x, x, x, mask)

    assert bool(jnp.all(jnp.isfinite(context)))
    assert bool(jnp.all(context == 0))
    assert bool(jnp.all(weights == 0))


def test_module_shapes_and_parameter_count():
    x = jax.random.normal(jax.random.key(0), (2, 5, 32))

    mask = jnp.broadcast_to(jnp.tril(jnp.ones((5, 5), dtype=bool)), (2, 1, 5, 5))

    model = MultiHeadAttention(num_heads=4, position_mode="rope")
    positions = jnp.arange(5)

    variables = model.init(jax.random.key(1), x, mask, positions)
    y = model.apply(variables, x, mask, positions)

    assert y.shape == x.shape  # type: ignore

    count = sum(a.size for a in jax.tree_util.tree_leaves(variables["params"]))

    assert count == 4 * 32 * 32


def test_future_input_cannot_change_past_output():
    x = jax.random.normal(jax.random.key(2), (1, 5, 16))

    token_ids = jnp.array([[256, 65, 66]], dtype=jnp.int32)

    mask = make_prefix_causal_mask(token_ids, jnp.array([True]), n_visual_tokens=2)
    positions = make_rope_positions(text_length=3, n_visual_tokens=2)
    model = MultiHeadAttention(num_heads=2, position_mode="rope")
    variables = model.init(jax.random.key(3), x, mask, positions)
    y1 = model.apply(variables, x, mask, positions)

    # Change only the final input token
    x_changed = x.at[:, 4, :].set(1000.0)

    y2 = model.apply(variables, x_changed, mask, positions)

    np.testing.assert_allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-6)  # type: ignore


def test_rope_and_nope_differ():
    x = jax.random.normal(jax.random.key(4), (1, 5, 16))
    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    rope = MultiHeadAttention(num_heads=2, position_mode="rope")
    nope = MultiHeadAttention(num_heads=2, position_mode="nope")

    variables = rope.init(jax.random.key(5), x, mask, jnp.arange(5))
    # Both modes use the exact same parameters
    y_rope = rope.apply(variables, x, mask, jnp.arange(5))
    y_nope = nope.apply(variables, x, mask)

    assert not np.allclose(y_rope, y_nope)  # type: ignore


def test_jit_and_gradients():
    x = jax.random.normal(jax.random.key(6), (1, 5, 16))

    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    positions = jnp.arange(5)
    model = MultiHeadAttention(num_heads=2)

    variables = model.init(jax.random.key(7), x, mask, positions)

    @jax.jit
    def loss_fn(params):
        output = model.apply({"params": params}, x, mask, positions)

        return jnp.mean(output**2)  # type: ignore

    _, grads = jax.value_and_grad(loss_fn)(variables["params"])

    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))
    assert float(jnp.linalg.norm(grads["q_proj"]["kernel"])) > 0
