import jax
import jax.numpy as jnp
from flax import linen as nn


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (x.shape[-1],),
        )

        # Compute normalization in float for stability
        x32 = x.astype(jnp.float32)

        rms_inverse = jax.lax.rsqrt(
            jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps
        )

        return (x32 * rms_inverse * scale).astype(x.dtype)


class SwiGLU(nn.Module):
    intermediate_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
        update = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)

        hidden = jax.nn.silu(gate) * update

        return nn.Dense(x.shape[-1], use_bias=False, name="down_proj")(hidden)


def apply_rope(
    x: jax.Array, positions: jax.Array | None = None, base: float = 10000.0
) -> jax.Array:
    """Rotate a tensor with shape [B, T, H, head_dim]"""

    if x.ndim != 4:
        raise ValueError("RoPE expects [B, T, heads, head_dim]")

    batch, length, heads, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    if base <= 0:
        raise ValueError("base must be positive")

    if positions is None:
        positions = jnp.arange(length)

    positions = jnp.asarray(positions, dtype=jnp.float32)

    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )

    # Support a shared [T] position vector or [B, T]
    if positions.shape == (length,):
        angles = (positions[:, None] * inv_freq)[None, :, None, :]

    elif positions.shape == (batch, length):
        angles = (positions[..., None] * inv_freq)[:, :, None, :]

    else:
        raise ValueError("positions must have shape [T] or [B, T]")

    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    # Pair adjacent feature dimensions
    pairs = x.astype(jnp.float32).reshape(batch, length, heads, head_dim // 2, 2)

    even = pairs[..., 0]
    odd = pairs[..., 1]

    # Apply the 2D rotation to each pair
    rotated = jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)

    return rotated.reshape(x.shape).astype(x.dtype)


def rotate_qk(
    q: jax.Array, k: jax.Array, positions: jax.Array | None = None, mode: str = "rope"
) -> tuple[jax.Array, jax.Array]:
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")

    if mode == "nope":
        return q, k

    if mode == "rope":
        return (apply_rope(q, positions=positions), apply_rope(k, positions=positions))

    raise ValueError(f"Unknown position mode: {mode}")
