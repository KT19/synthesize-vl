import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze

from synthesize_vl.models.attention import MultiHeadAttention
from synthesize_vl.models.gating import HeadwiseAttentionGate


def test_gate_initialization() -> None:
    x = jnp.ones((2, 5, 16))
    context = jnp.ones((2, 5, 4, 4))

    model = HeadwiseAttentionGate(num_heads=4)
    variables = model.init(jax.random.key(0), x, context)

    output = model.apply(variables, x, context)

    np.testing.assert_allclose(output, 0.5)  # type: ignore
    assert output.shape == context.shape  # type: ignore

    kernel = variables["params"]["gate_proj"]["kernel"]
    assert kernel.shape == (16, 4)  # type: ignore


def test_gate_gradients() -> None:
    x = jax.random.normal(jax.random.key(0), (2, 5, 16))
    context = jax.random.normal(jax.random.key(1), (2, 5, 4, 4))

    model = HeadwiseAttentionGate(num_heads=4)
    variables = model.init(jax.random.key(2), x, context)

    def loss_fn(params):
        y = model.apply({"params": params}, x, context)

        return jnp.mean(y**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])
    gate_grad = grads["gate_proj"]["kernel"]

    assert bool(jnp.all(jnp.isfinite(gate_grad)))
    assert float(jnp.linalg.norm(gate_grad)) > 0


def test_gated_attention_initial_scaling() -> None:
    x = jax.random.normal(jax.random.key(3), (1, 5, 16))

    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    positions = jnp.arange(5)

    gated = MultiHeadAttention(num_heads=4, use_gate=True)
    plain = MultiHeadAttention(num_heads=4, use_gate=False)

    variables = gated.init(jax.random.key(4), x, mask, positions)
    gated_output = gated.apply(variables, x, mask, positions)

    # Use exactly the same Q/K/V/output weights
    plain_params = unfreeze(variables["params"])
    plain_params.pop("head_gate")

    plain_output = plain.apply({"params": plain_params}, x, mask, positions)

    np.testing.assert_allclose(gated_output, 0.5 * plain_output, rtol=1e-5, atol=1e-5)  # type: ignore


def test_gated_attention_is_causal() -> None:
    x = jax.random.normal(jax.random.key(5), (1, 5, 16))

    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    positions = jnp.arange(5)

    model = MultiHeadAttention(num_heads=4, use_gate=True)

    variables = model.init(jax.random.key(6), x, mask, positions)
    before = model.apply(variables, x, mask, positions)

    # Modify the future token
    changed_x = x.at[:, 4, :].set(100.0)
    after = model.apply(variables, changed_x, mask, positions)

    np.testing.assert_allclose(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-5)  # type: ignore
