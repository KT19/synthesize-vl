import jax
from flax import linen as nn

from synthesize_vl.tokenizer import ByteTokenizer


class TextEmbedding(nn.Module):
    hidden_size: int
    vocab_size: int = ByteTokenizer.VOCAB_SIZE

    @nn.compact
    def __call__(self, token_ids: jax.Array) -> jax.Array:
        if token_ids.ndim != 2:
            raise ValueError("Expected token IDs of shape [B, T]")

        return nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="token_embedding",
        )(token_ids)


class VisionPatchEmbedding(nn.Module):
    hidden_size: int
    patch_size: int = 8
    channels: int = 3

    @nn.compact
    def __call__(self, images: jax.Array) -> jax.Array:
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, H, W, C]")

        batch, height, width, channels = images.shape

        if channels != self.channels:
            raise ValueError("Unexpected number of image channels")

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image dimensions must be divisible by patch_size")

        patches = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="patch_projection",
        )(images)

        # [B, H/P, W/P, D] -> [B, N_patches, D]
        batch, grid_h, grid_w, dim = patches.shape

        return patches.reshape((batch, grid_h * grid_w, dim))
