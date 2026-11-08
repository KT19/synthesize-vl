import jax
import jax.numpy as jnp

from synthesize_vl.models.gating import HeadwiseAttentionGate


def main() -> None:
    x = jax.random.normal(jax.random.key(0), (2, 5, 16))

    context = jax.random.normal(jax.random.key(1), (2, 5, 4, 4))
    gate = HeadwiseAttentionGate(num_heads=4)

    variables = gate.init(jax.random.key(2), x, context)

    output = gate.apply(variables, x, context)

    # At initialization, sigmoid(0) == 0.5
    assert bool(jnp.allclose(output, 0.5 * context))  # type: ignore

    def loss_fn(params: dict) -> jax.Array:
        y = gate.apply({"params": params}, x, context)

        return jnp.mean(y**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])
    kernel_grad = grads["gate_proj"]["kernel"]

    assert bool(jnp.all(jnp.isfinite(kernel_grad)))
    assert float(jnp.linalg.norm(kernel_grad)) > 0

    print("Input:", x.shape)
    print("Context:", context.shape)
    print("Gated context:", output.shape)  # type: ignore
    print("Gate kernel:", kernel_grad.shape)
    print("Gradient norm:", float(jnp.linalg.norm(kernel_grad)))
    print("Headwise attention gate: PASS")


if __name__ == "__main__":
    main()
