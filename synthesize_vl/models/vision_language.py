import jax
from flax import linen as nn

from synthesize_vl.models.attention import make_rope_positions
from synthesize_vl.models.multimodal import MultimodalEmbedding, make_prefix_causal_mask
from synthesize_vl.models.primitives import RMSNorm
from synthesize_vl.models.transformer import TransformerBlock
from synthesize_vl.tokenizer import ByteTokenizer


class MultimodalCausalLM(nn.Module):
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 128

    max_text_length: int = 128
    image_size: int = 64
    patch_size: int = 8

    position_mode: str = "rope"
    use_attention_gate: bool = False
    vocab_size: int = ByteTokenizer.VOCAB_SIZE

    @nn.compact
    def __call__(
        self, input_ids: jax.Array, images: jax.Array, has_image: jax.Array
    ) -> jax.Array:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids [B, T]")

        batch_size, text_length = input_ids.shape

        if not 1 <= text_length <= self.max_text_length:
            raise ValueError("Invalid text sequence length")

        if images.shape != (batch_size, self.image_size, self.image_size, 3):
            raise ValueError("Unexpected image shape")

        if has_image.shape != (batch_size,):
            raise ValueError("Expected has_image [B]")

        if self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("num_heads must divide hidden_size")

        if self.patch_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("Invalid vision patch size")

        if self.position_mode not in ("rope", "nope", "learned"):
            raise ValueError("Unknown positional encoding")

        head_dim = self.hidden_size // self.num_heads

        if self.position_mode == "rope" and head_dim % 2:
            raise ValueError("RoPE head dimension must be even")

        n_visual = (self.image_size // self.patch_size) ** 2

        # Learned positions are handled in the embedding layer
        # RoPE is handled inside attention
        text_position_mode = "learned" if self.position_mode == "learned" else "nope"
        attention_position_mode = "rope" if self.position_mode == "rope" else "nope"

        # 1. Joint image/text embedding
        x = MultimodalEmbedding(
            hidden_size=self.hidden_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_length=self.max_text_length,
            text_position_mode=text_position_mode,
            name="multimodal_embedding",
        )(input_ids, images, has_image)

        # 2. Prefix-causal attention mask
        mask = make_prefix_causal_mask(input_ids, has_image, n_visual_tokens=n_visual)

        # 3. Rotary positions, if enables
        positions = (
            make_rope_positions(text_length=text_length, n_visual_tokens=n_visual)
            if attention_position_mode == "rope"
            else None
        )

        # 4. Shared multimodal Transformer
        for layer in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                position_mode=attention_position_mode,
                use_attention_gate=self.use_attention_gate,
                name=f"block_{layer}",
            )(x, mask, positions)

        # 5. Final normalization
        x = RMSNorm(name="final_norm")(x)

        # 6. Retain only text positions
        text_hidden = x[:, n_visual:, :]

        # 7. Predict next byte tokens
        logits = nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(text_hidden)

        return logits
