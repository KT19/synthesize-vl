import jax
import jax.numpy as jnp

from synthesize_vl.models.mhc import MHCBlock


def main() -> None:
    key = jax.random.key(0)
    streams = jax.random.normal(key, (2, 6, 2, 16))

    mask = jnp.tril(
        jnp.ones(
            (
                6,
                6,
            ),
            dtype=jnp.bool_,
        )
    )[None, None, :, :]
    mask = jnp.broadcast_to(mask, (2, 1, 6, 6))

    model = MHCBlock(num_streams=2, num_heads=4, intermediate_size=32)

    positions = jnp.arange(6)
    variables = model.init(jax.random.key(1), streams, mask, positions)

    output = model.apply(variables, streams, mask, positions)

    def loss_fn(params: dict) -> jax.Array:
        result = model.apply({"params": params}, streams, mask, positions)

        return jnp.mean(result**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    assert output.shape == streams.shape  # type: ignore
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))

    print("Input streams:", streams.shape)
    print("Output streams:", output.shape)  # type: ignore
    print("mHC block: PASS")


if __name__ == "__main__":
    main()
