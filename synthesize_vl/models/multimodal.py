import jax
import jax.numpy as jnp
from flax import linen as nn

from synthesize_vl.models.embedding import TextEmbedding, VisionPatchEmbedding
from synthesize_vl.tokenizer import ByteTokenizer


class MultimodalEmbedding(nn.Module):
    hidden_size: int
    image_size: int = 64
    patch_size: int = 8
    max_text_length: int = 128
    text_position_mode: str = "learned"

    @nn.compact
    def __call__(
        self, token_ids: jax.Array, images: jax.Array, has_image: jax.Array
    ) -> jax.Array:
        batch, text_length = token_ids.shape

        if images.shape != (batch, self.image_size, self.image_size, 3):
            raise ValueError("Unexpected image shape")

        if has_image.shape != (batch,):
            raise ValueError("Unexpected has_image shape")

        if text_length > self.max_text_length:
            raise ValueError("Text exceeds maximum length")

        if self.patch_size <= 0:
            raise ValueError("Invalid patch size")

        if self.image_size % self.patch_size:
            raise ValueError("Incompatible patch size")

        # Text embedding
        text = TextEmbedding(hidden_size=self.hidden_size, name="text_embedding")(
            token_ids
        )

        # Visual patch embeddings
        vision = VisionPatchEmbedding(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            name="vision_embedding",
        )(images)

        # Text positions
        if self.text_position_mode == "learned":
            positions = jnp.arange(text_length)

            pos = nn.Embed(
                num_embeddings=self.max_text_length,
                features=self.hidden_size,
                name="text_positions",
            )(positions)

            text = text + pos[None, :, :]

        elif self.text_position_mode == "nope":
            # No explicit text positional embedding
            pass
        else:
            raise ValueError("Unknown text position mode")

        # 2D visual positions
        grid = self.image_size // self.patch_size

        row_ids = jnp.repeat(jnp.arange(grid), grid)
        col_ids = jnp.tile(jnp.arange(grid), grid)

        row = nn.Embed(
            num_embeddings=grid, features=self.hidden_size, name="image_row_positions"
        )(row_ids)
        col = nn.Embed(
            num_embeddings=grid, features=self.hidden_size, name="image_col_positions"
        )(col_ids)

        vision = vision + (row + col)[None, :, :]

        # Modality embeddings
        modality = self.param(
            "modality_embedding",
            nn.initializers.normal(stddev=0.02),
            (2, self.hidden_size),
        )

        text = text + modality[0]
        vision = vision + modality[1]

        # Disable unused visual inputs
        vision = jnp.where(has_image[:, None, None], vision, jnp.zeros_like(vision))

        # Disable text padding representations
        text = jnp.where(
            (token_ids != ByteTokenizer.PAD)[:, :, None], text, jnp.zeros_like(text)
        )

        # (B, N_visual + T, D)
        return jnp.concatenate([vision, text], axis=1)


def make_prefix_causal_mask(
    token_ids: jax.Array, has_image: jax.Array, n_visual_tokens: int = 64
) -> jax.Array:
    """Create a bidirectional-image, causal-text mask."""
    batch, text_length = token_ids.shape

    if has_image.shape != (batch,):
        raise ValueError("Unexpected has_image shape")

    if n_visual_tokens < 0:
        raise ValueError("Invalid visual token count")

    visual_valid = jnp.broadcast_to(has_image[:, None], (batch, n_visual_tokens))

    text_valid = token_ids != ByteTokenizer.PAD

    valid = jnp.concatenate([visual_valid, text_valid], axis=1)

    total_length = n_visual_tokens + text_length
    positions = jnp.arange(total_length)

    query_is_visual = positions[:, None] < n_visual_tokens
    key_is_visual = positions[None, :] < n_visual_tokens

    causal = positions[None, :] <= positions[:, None]

    allowed = jnp.where(query_is_visual, key_is_visual, key_is_visual | causal)
    # Exclude invalid queries and kyes.
    allowed = allowed[None, :, :] & valid[:, :, None] & valid[:, None, :]

    # Give inactive queries a self-entry to avoid completely masked softmax rows
    eye = jnp.eye(total_length, dtype=jnp.bool_)

    allowed = allowed | (~valid)[:, :, None] & eye[None, :, :]

    return allowed[:, None, :, :]
