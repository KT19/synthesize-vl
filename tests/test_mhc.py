import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.models.mhc import MHCBlock, sinkhorn


def test_sinkhorn_is_doubly_stochastic() -> None:
    logits = jax.random.normal(jax.random.key(0), (2, 3, 4, 4))

    matrix = sinkhorn(logits, iterations=20)

    assert matrix.shape == logits.shape
    assert bool(jnp.all(matrix >= 0))

    np.testing.assert_allclose(matrix.sum(axis=-1), 1.0, atol=1e-4)
    np.testing.assert_allclose(matrix.sum(axis=-2), 1.0, atol=1e-4)


def test_sinkhorn_gradients() -> None:
    logits = jax.random.normal(jax.random.key(1), (2, 4, 4))

    def loss_fn(x: jax.Array) -> jax.Array:
        return jnp.sum(sinkhorn(x) ** 2)

    grads = jax.jit(jax.grad(loss_fn))(logits)

    assert bool(jnp.all(jnp.isfinite(grads)))
    assert float(jnp.linalg.norm(grads)) > 0


def test_mhc_block_shapes_and_gradients() -> None:
    streams = jax.random.normal(jax.random.key(2), (1, 5, 2, 16))

    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    positions = jnp.arange(5)

    model = MHCBlock(num_streams=2, num_heads=4, intermediate_size=32)

    variables = model.init(jax.random.key(3), streams, mask, positions)
    output = model.apply(variables, streams, mask, positions)

    assert output.shape == streams.shape  # type: ignore

    def loss_fn(params):
        result = model.apply({"params": params}, streams, mask, positions)

        return jnp.mean(result**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    for grad in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.all(jnp.isfinite(grad)))

    assert float(jnp.linalg.norm(grads["pre_router"]["kernel"])) > 0


def test_mhc_block_preserves_causality() -> None:
    streams = jax.random.normal(jax.random.key(4), (1, 5, 2, 16))
    mask = jnp.tril(jnp.ones((5, 5), dtype=bool))[None, None, :, :]

    positions = jnp.arange(5)

    model = MHCBlock(num_streams=2, num_heads=4, intermediate_size=32)
    variables = model.init(jax.random.key(5), streams, mask, positions)
    before = model.apply(variables, streams, mask, positions)
    # Change only the final sequence position
    changed = streams.at[:, 4, :, :].set(100.0)

    after = model.apply(variables, changed, mask, positions)

    np.testing.assert_allclose(before[:, :4], after[:, :4], atol=1e-5, rtol=1e-5)  # type: ignore
