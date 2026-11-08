import jax
import jax.numpy as jnp
from flax import linen as nn

from synthesize_vl.models.attention import MultiHeadAttention
from synthesize_vl.models.embedding import TextEmbedding
from synthesize_vl.models.multimodal import make_prefix_causal_mask
from synthesize_vl.models.primitives import RMSNorm, SwiGLU
from synthesize_vl.tokenizer import ByteTokenizer


class TransformerBlock(nn.Module):
    num_heads: int
    intermediate_size: int
    position_mode: str = "rope"
    use_attention_gate: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None,
        positions: jax.Array | None = None,
        *,
        decode: bool = False,
        max_decode_length: int | None = None,
    ) -> jax.Array:
        # First sublayer: pre-normalized attention
        h = RMSNorm(name="attention_norm")(x)

        h = MultiHeadAttention(
            num_heads=self.num_heads,
            position_mode=self.position_mode,
            use_gate=self.use_attention_gate,
            name="attention",
        )(h, mask, positions, decode=decode, max_decode_length=max_decode_length)

        x = x + h

        # Second sublayer: pre-normalized SwiGLU
        h = RMSNorm(name="feedforward_norm")(x)

        h = SwiGLU(intermediate_size=self.intermediate_size, name="feedforward")(h)

        return x + h


class CausalLM(nn.Module):
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 128
    max_length: int = 128
    position_mode: str = "rope"
    use_attention_gate: bool = False
    vocab_size: int = ByteTokenizer.VOCAB_SIZE

    @nn.compact
    def __call__(self, input_ids: jax.Array, *, decode: bool = False) -> jax.Array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        batch_size, sequence_length = input_ids.shape

        if not (1 <= sequence_length <= self.max_length):
            raise ValueError("Invalid input sequence length")

        if self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("num_heads must divide hidden_size")

        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("Layers and intermediate_size must be positive")

        if self.position_mode not in ("rope", "nope"):
            raise ValueError("position_mode must be `rope` and `nope`")

        head_dim = self.hidden_size // self.num_heads

        if self.position_mode == "rope" and head_dim % 2:
            raise ValueError("RoPE head dimension must be even")

        # Token embeddings
        x = TextEmbedding(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            name="text_embedding",
        )(input_ids)

        if decode:
            mask = None
            positions = None
        else:
            # Causal attention mask
            mask = make_prefix_causal_mask(
                input_ids,
                has_image=jnp.zeros((batch_size,), dtype=jnp.bool_),
                n_visual_tokens=0,
            )

            # Positional encoding
            positions = (
                jnp.arange(sequence_length, dtype=jnp.int32)
                if self.position_mode == "rope"
                else None
            )

        # Transformer Stack
        for layer in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                position_mode=self.position_mode,
                use_attention_gate=self.use_attention_gate,
                name=f"block_{layer}",
            )(x, mask, positions, decode=decode, max_decode_length=self.max_length)

        # Language modeling head
        x = RMSNorm(name="final_norm")(x)

        logits = nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(x)

        return logits
