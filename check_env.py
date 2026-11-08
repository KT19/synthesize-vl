import jax
import jax.numpy as jnp

print("JAV version:", jax.__version__)

devices = jax.devices()
print("Devices:", devices)


def loss_fn(x):
    return jnp.sum(x**2)


x = jnp.array([1.0, 2.0, 3.0])

grad_fn = jax.jit(jax.grad(loss_fn))
gradients = grad_fn(x)
print("Gradients: ", gradients)

gpus = [d for d in devices if d.platform == "gpu"]
print("GPU count: ", len(gpus))
