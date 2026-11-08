import jax
import jax.numpy as jnp

from synthesize_vl.models.multimodal import make_prefix_causal_mask


def make_hybrid_local_global_mask(
    token_ids: jax.Array,
    has_image: jax.Array,
    *,
    n_visual_tokens: int = 64,
    local_window: int = 32,
    global_text_tokens: int = 1,
) -> jax.Array:
    """Hybrid prefix-causal mask with shape [B, 1, L, L]

    Image queries:
        Attend to all valid image patches

    Text queries:
        Attend to valid image patches, the first global_text_tokens text positions, and the local causal text window.
    """
    if local_window < 1:
        raise ValueError("local_wwindow must be positive")

    if global_text_tokens < 0:
        raise ValueError("global_text_tokens must be nonnegative")

    if n_visual_tokens < 0:
        raise ValueError("n_visual_tokens must be nonnegative")

    full_mask = make_prefix_causal_mask(
        token_ids, has_image, n_visual_tokens=n_visual_tokens
    )
    length = n_visual_tokens + token_ids.shape[1]

    query = jnp.arange(length)[:, None]
    key = jnp.arange(length)[None, :]

    query_is_image = query < n_visual_tokens
    key_is_image = key < n_visual_tokens

    # Includes the query token itself
    # local_window=3 permits, i, i-1, and i-2
    local_text = (
        (query >= n_visual_tokens)
        & (key >= n_visual_tokens)
        & (query >= key)
        & ((query - key) < local_window)
    )

    # The first few text positions remain global keys.
    # Causality still restrics when they can be read.
    global_text = (key >= n_visual_tokens) & (
        key < n_visual_tokens + global_text_tokens
    )

    keep = query_is_image | key_is_image | local_text | global_text

    return full_mask & keep[None, None, :, :]
