import jax
import jax.numpy as jnp
from flax import linen as nn

from synthesize_vl.models.attention import make_rope_positions
from synthesize_vl.models.attention_masks import make_hybrid_local_global_mask
from synthesize_vl.models.mhc import MHCBlock
from synthesize_vl.models.multimodal import MultimodalEmbedding, make_prefix_causal_mask
from synthesize_vl.models.primitives import RMSNorm
from synthesize_vl.models.transformer import TransformerBlock
from synthesize_vl.tokenizer import ByteTokenizer


class RecurrentCore(nn.Module):
    num_heads: int
    intermediate_size: int
    num_layers: int = 1
    position_mode: str = "rope"
    use_attention_gate: bool = True

    residual_type: str = "standard"
    num_streams: int = 2
    sinkhorn_iterations: int = 20

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        encoded: jax.Array,
        mask: jax.Array,
        positions: jax.Array | None,
        steps: int,
    ) -> jax.Array:
        """Apply a shared stack of distinct layers repeatedly.

        state: [B, L, D] evolving latent state
        encoded: [B, L, D] fixed Prelude output
        """

        if state.shape != encoded.shape:
            raise ValueError("State and encoded shapes differ")

        if steps < 1:
            raise ValueError("steps must be positive")

        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

        if self.residual_type == "mhc" and self.num_streams < 2:
            raise ValueError("mHC requires at least two streams")

        hidden_size = state.shape[-1]

        # A learned bridge reinjects the Prelude representation
        injection = nn.Dense(
            features=hidden_size,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="input_bridge",
        )

        if self.residual_type == "standard":
            # Each layer has its own parameters; reuse the stack across iterations.
            shared_blocks = [
                TransformerBlock(
                    num_heads=self.num_heads,
                    intermediate_size=self.intermediate_size,
                    position_mode=self.position_mode,
                    use_attention_gate=self.use_attention_gate,
                    name=f"shared_block_{layer}",
                )
                for layer in range(self.num_layers)
            ]

            for _ in range(steps):
                combined = jnp.concatenate([state, encoded], axis=-1)

                state = state + injection(combined)

                for block in shared_blocks:
                    state = block(state, mask, positions)

            return state

        # mHC recurrent core
        # Lift the single hidden representation into slightly different learned residual streams.
        offsets = self.param(
            "stream_offsets",
            nn.initializers.normal(stddev=0.01),
            (self.num_streams, hidden_size),
        )

        streams = state[:, :, None, :] + offsets[None, None, :, :]

        # Keep the residual streams through every layer and iteration.
        shared_mhc_blocks = [
            MHCBlock(
                num_streams=self.num_streams,
                sinkhorn_iterations=self.sinkhorn_iterations,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                position_mode=self.position_mode,
                use_attention_gate=self.use_attention_gate,
                name=f"shared_mhc_{layer}",
            )
            for layer in range(self.num_layers)
        ]

        for _ in range(steps):
            summary = jnp.mean(streams, axis=2)

            combined = jnp.concatenate([summary, encoded], axis=-1)

            streams = streams + injection(combined)[:, :, None, :]
            for block in shared_mhc_blocks:
                streams = block(streams, mask, positions=positions)

        # Merge streams for the single-stream Coda
        return jnp.mean(streams, axis=2)


class RecurrentMultimodalCausalLM(nn.Module):
    hidden_size: int = 64
    num_heads: int = 4
    intermediate_size: int = 128

    recurrent_steps: int = 2
    core_layers: int = 1

    max_text_length: int = 128
    image_size: int = 64
    patch_size: int = 8

    residual_type: str = "standard"
    num_streams: int = 2
    sinkhorn_iterations: int = 20

    core_attention: str = "full"
    local_window: int = 32
    global_text_tokens: int = 1

    position_mode: str = "rope"
    use_attention_gate: bool = True
    vocab_size: int = ByteTokenizer.VOCAB_SIZE

    @nn.compact
    def __call__(
        self, input_ids: jax.Array, images: jax.Array, has_image: jax.Array
    ) -> jax.Array:
        if input_ids.ndim != 2:
            raise ValueError("Expected token IDs [B, T]")

        batch, text_length = input_ids.shape

        if not 1 <= text_length <= self.max_text_length:
            raise ValueError("Invalid text length")

        if images.shape != (batch, self.image_size, self.image_size, 3):
            raise ValueError("Unexpected image shape")

        if has_image.shape != (batch,):
            raise ValueError("Expected has_image [B]")

        if self.recurrent_steps < 1:
            raise ValueError("Invalid recurrence count")

        if self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("Invalid head configuration")

        if self.intermediate_size < 1:
            raise ValueError("Invalid intermediate size")

        if self.patch_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("Invalid patch size")

        if self.position_mode not in ("rope", "nope", "learned"):
            raise ValueError("Unknown position mode")

        head_dim = self.hidden_size // self.num_heads

        if self.position_mode == "rope" and head_dim % 2:
            raise ValueError("RoPE head dimension must be even")

        if self.core_attention not in ("full", "hybrid"):
            raise ValueError("core_attention must be 'full' or 'hybrid'")

        if self.local_window < 1:
            raise ValueError("local_window must be positive")

        if self.global_text_tokens < 0:
            raise ValueError("global_text_tokens must be nonnegative")

        n_visual = (self.image_size // self.patch_size) ** 2

        text_position_mode = "learned" if self.position_mode == "learned" else "nope"
        attention_position_mode = "rope" if self.position_mode == "rope" else "nope"

        # 1. Native multimodal embeddings
        x = MultimodalEmbedding(
            hidden_size=self.hidden_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_length=self.max_text_length,
            text_position_mode=text_position_mode,
            name="multimodal_embedding",
        )(input_ids, images, has_image)

        # Prelude and Coda
        full_mask = make_prefix_causal_mask(
            input_ids, has_image, n_visual_tokens=n_visual
        )

        if self.core_attention == "full":
            core_mask = full_mask
        else:
            core_mask = make_hybrid_local_global_mask(
                input_ids,
                has_image,
                n_visual_tokens=n_visual,
                local_window=self.local_window,
                global_text_tokens=self.global_text_tokens,
            )

        positions = (
            make_rope_positions(text_length, n_visual_tokens=n_visual)
            if attention_position_mode == "rope"
            else None
        )

        # 2. Prelude
        encoded = TransformerBlock(
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            position_mode=attention_position_mode,
            use_attention_gate=self.use_attention_gate,
            name="prelude",
        )(x, full_mask, positions)

        # 3. Shared recurrent reasoning
        state = RecurrentCore(
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_layers=self.core_layers,
            position_mode=attention_position_mode,
            use_attention_gate=self.use_attention_gate,
            residual_type=self.residual_type,
            num_streams=self.num_streams,
            sinkhorn_iterations=self.sinkhorn_iterations,
            name="core",
        )(
            state=encoded,
            encoded=encoded,
            mask=core_mask,
            positions=positions,
            steps=self.recurrent_steps,
        )

        # 4. Coda
        state = TransformerBlock(
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            position_mode=attention_position_mode,
            use_attention_gate=self.use_attention_gate,
        )(x=state, mask=full_mask, positions=positions)

        # 5. Final normalization
        state = RMSNorm(name="final_norm")(state)

        # Only text tokens contribute to the LM head
        text_hidden = state[:, n_visual:, :]

        return nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(text_hidden)
