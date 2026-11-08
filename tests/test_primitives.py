import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.models.primitives import RMSNorm, SwiGLU, apply_rope, rotate_qk


def test_rmsnorm_shape_and_magnitude():
    x = jnp.array([[[3.0, 4.0], [-5.0, 12.0]]])

    model = RMSNorm()
    variables = model.init(jax.random.key(0), x)
    y = model.apply(variables, x)

    assert y.shape == x.shape  # type: ignore

    rms = jnp.sqrt(jnp.mean(y**2, axis=-1))  # type: ignore

    np.testing.assert_allclose(rms, jnp.ones_like(rms), atol=1e-5)


def test_rmsnorm_zero_input():
    model = RMSNorm()
    x = jnp.zeros((2, 4, 16))

    variables = model.init(jax.random.key(0), x)
    y = model.apply(variables, x)

    np.testing.assert_array_equal(y, x)

    assert bool(jnp.all(jnp.isfinite(y)))  # type: ignore


def test_rmsnorm_gradients():
    model = RMSNorm()
    x = jax.random.normal(jax.random.key(1), (2, 4, 16))

    variables = model.init(jax.random.key(2), x)

    def loss_fn(params):
        y = model.apply({"params": params}, x)

        return jnp.mean(y**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    assert bool(jnp.all(jnp.isfinite(grads["scale"])))
    assert float(jnp.linalg.norm(grads["scale"])) > 0


def test_swiglu_shape_and_gradients():
    model = SwiGLU(intermediate_size=32)

    x = jax.random.normal(jax.random.key(0), (2, 5, 16))

    variables = model.init(jax.random.key(1), x)
    y = model.apply(variables, x)

    assert y.shape == x.shape  # type: ignore

    def loss_fn(params):
        output = model.apply({"params": params}, x)
        return jnp.mean(output**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    for grad in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.all(jnp.isfinite(grad)))


def test_rope_preserves_norm():
    x = jax.random.normal(jax.random.key(0), (2, 8, 4, 16))
    y = apply_rope(x)

    assert y.shape == x.shape

    np.testing.assert_allclose(
        jnp.linalg.norm(x, axis=-1), jnp.linalg.norm(y, axis=-1), atol=1e-5, rtol=1e-5
    )


def test_rope_zero_position_is_identity():
    x = jax.random.normal(jax.random.key(0), (1, 8, 2, 16))

    y = apply_rope(x, jnp.zeros(8))

    np.testing.assert_allclose(x, y, atol=1e-6)


def test_rope_relative_positions():
    x = jax.random.normal(jax.random.key(0), (1, 4, 2, 8))

    positions = jnp.arange(4)

    # should not change when both positions receive the same shift
    a = apply_rope(x, positions)
    b = apply_rope(x, positions + 17)

    dot_a = jnp.sum(a[:, 0, 0, :] * a[:, 2, 0, :], axis=-1)
    dot_b = jnp.sum(b[:, 0, 0, :] * b[:, 2, 0, :], axis=-1)

    np.testing.assert_allclose(dot_a, dot_b, atol=1e-5, rtol=1e-5)


def test_rope_jit_and_gradients():
    x = jax.random.normal(jax.random.key(0), (2, 5, 4, 8))

    y = jax.jit(apply_rope)(x)

    assert y.shape == x.shape

    grads = jax.grad(lambda z: jnp.mean(apply_rope(z) ** 2))(x)

    assert bool(jnp.all(jnp.isfinite(grads)))


def test_rope_nope_switch():
    x = jnp.ones((1, 4, 2, 8))

    q, k = rotate_qk(x, x, mode="nope")

    np.testing.assert_array_equal(q, x)
    np.testing.assert_array_equal(k, x)

    with pytest.raises(ValueError):
        rotate_qk(x, x, mode="unknown")
