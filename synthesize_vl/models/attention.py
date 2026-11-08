import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from synthesize_vl.models.gating import HeadwiseAttentionGate
from synthesize_vl.models.primitives import rotate_qk


def make_rope_positions(text_length: int, n_visual_tokens: int = 64) -> jax.Array:
    """Image patches get zero rotary position."""
    if text_length < 0 or n_visual_tokens < 0:
        raise ValueError("Sequence lengths must be nonnegative")

    return jnp.concatenate(
        [
            jnp.zeros(n_visual_tokens, dtype=jnp.int32),
            jnp.arange(text_length, dtype=jnp.int32),
        ]
    )


def scaled_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return context vectors and attention probabilities.
    q, k, v: [B, L, H, Dh]
    mask: [B, 1, L, L], True means allowed

    context: [B, L, H, Dh]
    probabilities: [B, H, L, L]
    """

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, V must share [B, L, H, Dh] shape")

    batch, length, _, head_dim = q.shape

    if head_dim < 1 or mask.shape != (batch, 1, length, length):
        raise ValueError("Invalid head dimension or attention mask shape")

    # [B, L, H, Dh] x [B, L, H, Dh] -> [B, H, L, L]
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)
    ) / math.sqrt(head_dim)

    # Apply the attention mask
    allowed = mask.astype(jnp.bool_)

    scores = jnp.where(allowed, scores, jnp.finfo(jnp.float32).min)

    probabilities = jax.nn.softmax(scores, axis=-1)

    # For disallowed entries to exactly zero.
    probabilities = jnp.where(allowed, probabilities, 0.0)
    probabilities = probabilities / jnp.maximum(
        probabilities.sum(axis=-1, keepdims=True), 1e-9
    )

    # Weighted sum of value vectors.
    # [B, H, L, L] x [B, L, H, Dh] -> [B, L, H, Dh]
    context = jnp.einsum("bhqk,bkhd->bqhd", probabilities, v.astype(jnp.float32))

    return context.astype(v.dtype), probabilities


class MultiHeadAttention(nn.Module):
    num_heads: int
    position_mode: str = "rope"
    use_gate: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None,
        positions: jax.Array | None = None,
        *,
        decode: bool = False,
        max_decode_length: int | None = None,
    ) -> jax.Array:
        if x.ndim != 3:
            raise ValueError("Expected hidden states of shape [B, L, D]")

        batch, length, hidden_size = x.shape

        if self.num_heads <= 0 or hidden_size % self.num_heads:
            raise ValueError("num_heads must evenly divide hidden_size")

        if self.position_mode not in ("rope", "nope"):
            raise ValueError("Unknown position mode")

        head_dim = hidden_size // self.num_heads

        if decode:
            if length != 1:
                raise ValueError("Decode requires one token at a time")
            if max_decode_length is None or max_decode_length < 1:
                raise ValueError("Invalid cache capacity")
        else:
            if attention_mask is None or attention_mask.shape != (
                batch,
                1,
                length,
                length,
            ):
                raise ValueError("Invalid attention mask")
            if self.position_mode == "rope" and positions is None:
                raise ValueError("RoPE requires positions")

        # Project hidden states into multiple heads.
        def project(name: str) -> jax.Array:
            return nn.DenseGeneral(
                features=(self.num_heads, head_dim), use_bias=False, name=name
            )(x)

        q = project("q_proj")
        k = project("k_proj")
        v = project("v_proj")

        if decode:
            capacity = max_decode_length
            initialized = self.has_variable("cache", "cached_key")

            cached_key = self.variable(
                "cache",
                "cached_key",
                jnp.zeros,
                (batch, capacity, self.num_heads, head_dim),
                k.dtype,
            )
            cached_value = self.variable(
                "cache",
                "cached_value",
                jnp.zeros,
                (batch, capacity, self.num_heads, head_dim),
                v.dtype,
            )

            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )

            if initialized:
                index = cache_index.value
                # Absolute position of the current token
                step_positions = index + jnp.arange(1, dtype=jnp.int32)
                q, k = rotate_qk(q, k, step_positions, mode=self.position_mode)

                keys = jax.lax.dynamic_update_slice(
                    cached_key.value, k, (0, index, 0, 0)
                )
                values = jax.lax.dynamic_update_slice(
                    cached_value.value, v, (0, index, 0, 0)
                )

                cached_key.value = keys
                cached_value.value = values
                cache_index.value = index + 1

                scores = jnp.einsum(
                    "bqhd,bkhd->bhqk", q.astype(jnp.float32), keys.astype(jnp.float32)
                ) / jnp.sqrt(float(head_dim))

                valid_keys = (jnp.arange(capacity) <= index)[None, None, None, :]
                scores = jnp.where(valid_keys, scores, jnp.finfo(jnp.float32).min)

                weights = jax.nn.softmax(scores, axis=-1)
                weights = jnp.where(valid_keys, weights, 0.0)

                context = jnp.einsum(
                    "bhqk,bkhd->bqhd", weights, values.astype(jnp.float32)
                ).astype(v.dtype)

            else:
                # Cache initialization only
                # Parameters and cache arrays are created, but no token is consumed
                context = jnp.zeros_like(q)

        else:
            # Apply RoPE or leave Q/K unchanged for NoPE.
            q, k = rotate_qk(q, k, positions, mode=self.position_mode)

            context, _ = scaled_dot_product_attention(q, k, v, attention_mask)  # type: ignore

        if self.use_gate:
            context = HeadwiseAttentionGate(num_heads=self.num_heads, name="head_gate")(
                x, context
            )

        # Merge attention heads back together
        context = context.reshape(batch, length, hidden_size)

        return nn.Dense(hidden_size, use_bias=False, name="out_proj")(context)
