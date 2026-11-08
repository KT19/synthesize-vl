import jax
import jax.numpy as jnp
from flax import linen as nn


class HeadwiseAttentionGate(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        Gate SDPA outputs using the query hidden states.

        x: [B, L, D]
        context: [B. L, H, Dh]
        returns: [B, L, H, Dh]
        """
        if x.ndim != 3 or context.ndim != 4:
            raise ValueError("Invalid input rank")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if x.shape[:2] != context.shape[:2]:
            raise ValueError("Batch/sequence dimensions differ")

        if context.shape[2] != self.num_heads:
            raise ValueError("Unexpected number of heads")

        # One learned gate per token and attention head
        logits = nn.Dense(
            features=self.num_heads,
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="gate_proj",
        )(x)

        gates = jax.nn.sigmoid(logits.astype(jnp.float32)).astype(context.dtype)

        # [B, L, H] -> [B, L, H, 1]
        return context * gates[..., None]
