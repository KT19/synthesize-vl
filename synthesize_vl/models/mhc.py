import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from synthesize_vl.models.primitives import RMSNorm
from synthesize_vl.models.transformer import TransformerBlock


def sinkhorn(logits: jax.Array, iterations: int = 20) -> jax.Array:
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Expected [..., S, S] square matricies")

    if iterations < 1:
        raise ValueError("iterations must be positive")

    log_matrix = logits.astype(jnp.float32)

    for _ in range(iterations):
        # Normalize rows.
        log_matrix = log_matrix - jax.scipy.special.logsumexp(
            log_matrix, axis=-1, keepdims=True
        )
        # Normalize columns
        log_matrix = log_matrix - jax.scipy.special.logsumexp(
            log_matrix, axis=-2, keepdims=True
        )

    return jnp.exp(log_matrix)


class MHCBlock(nn.Module):
    """One mHC-routed Transformer residual update."""

    num_streams: int
    num_heads: int
    intermediate_size: int
    position_mode: str = "rope"
    use_attention_gate: bool = True
    sinkhorn_iterations: int = 20

    @nn.compact
    def __call__(
        self,
        streams: jax.Array,
        attention_mask: jax.Array,
        positions: jax.Array | None = None,
    ) -> jax.Array:
        if streams.ndim != 4:
            raise ValueError("Expected streams [B, L, S, D]")

        batch, length, stream_count, hidden_size = streams.shape

        if self.num_streams < 2 or stream_count != self.num_streams:
            raise ValueError("Expected at least two matching streams")
        if attention_mask.shape != (batch, 1, length, length):
            raise ValueError("Invalid attention mask shape")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")

        # Routing depends on the current token's streams.
        flattened = streams.reshape(batch, length, stream_count * hidden_size)
        routing_input = RMSNorm(name="router_norm")(flattened.astype(jnp.float32))

        def identity_biased_init(
            key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
        ) -> jax.Array:
            del key
            identity = jnp.eye(stream_count, dtype=dtype)
            values = jnp.where(identity > 0, 2.0, -2.0)

            return values.reshape(shape)

        # Initially, each stream contributions about 1 / S
        pre_logits = nn.Dense(
            stream_count,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.constant(-math.log(stream_count - 1)),
            name="pre_router",
        )(routing_input)

        # Initially, each stream receives about one copy of the Transformer residual update
        post_logits = nn.Dense(
            stream_count,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.zeros,
            name="post_router",
        )(routing_input)

        # Initialize residual mixing close to identity
        res_logits = nn.Dense(
            stream_count * stream_count,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=identity_biased_init,
            name="residual_router",
        )(routing_input)

        h_pre = jax.nn.sigmoid(pre_logits)
        h_post = 2.0 * jax.nn.sigmoid(post_logits)

        h_res = sinkhorn(
            res_logits.reshape(batch, length, stream_count, stream_count),
            self.sinkhorn_iterations,
        )

        fp32_streams = streams.astype(jnp.float32)

        # Read S streams into one [B, L, D] representation
        read = jnp.einsum("bts,btsd->btd", h_pre, fp32_streams)
        # Residual stream mixing
        mixed = jnp.einsum("btij,btjd->btid", h_res, fp32_streams)

        # Extract the residual update from our existing pre-normalized Transformer Block
        read = read.astype(streams.dtype)

        transformed = TransformerBlock(
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            position_mode=self.position_mode,
            use_attention_gate=self.use_attention_gate,
            name="residual_function",
        )(read, attention_mask, positions=positions)

        delta = (transformed - read).astype(jnp.float32)

        # Write the Transformer update into every stream
        updated = mixed + h_post[..., None] * delta[:, :, None, :]

        return updated.astype(streams.dtype)
