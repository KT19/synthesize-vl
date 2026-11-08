from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from PIL import Image

from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.models.vision_language import MultimodalCausalLM
from synthesize_vl.tokenizer import ByteTokenizer


@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool


def _prefix_tokens(prompt: str, has_image: bool) -> list[int]:
    tok = ByteTokenizer()

    return [tok.BOS] + ([tok.IMAGE] if has_image else []) + list(prompt.encode("utf-8"))


def _choose_token(logits: jax.Array) -> int:
    """Greedy decoding, excluding input-only special tokens."""
    scores = logits

    for token_id in (ByteTokenizer.BOS, ByteTokenizer.PAD, ByteTokenizer.IMAGE):
        scores = scores.at[token_id].set(-jnp.inf)

    return int(jnp.argmax(scores))


def _result(generated: list[int], stopped: bool) -> GenerationResult:
    # Incomplete UTF-8 sequences are possible mid-generation
    # Preserve raw token IDs and decode text safely
    text = bytes(t for t in generated if t < 256).decode("utf-8", errors="replace")

    return GenerationResult(
        token_ids=tuple(generated), text=text, stopped_on_eos=stopped
    )


def _validate_length(prefix_length: int, max_new_tokens: int, capacity: int) -> None:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")

    # The last output token need not be fed back when no further generation step is requested.
    consumed = prefix_length + max(0, max_new_tokens - 1)

    if consumed > capacity:
        raise ValueError("Generation exceeds context capacity")


def generate_greedy(
    model: nn.Module,
    params: Any,
    prompt: str,
    *,
    image: Image.Image | None = None,
    max_new_tokens: int = 32,
) -> GenerationResult:
    """Correctness-first generation without KV caching"""

    multimodal = isinstance(model, (MultimodalCausalLM, RecurrentMultimodalCausalLM))

    if image is not None and not multimodal:
        raise ValueError("This model does not accept images")

    visual = image is not None
    tokens = _prefix_tokens(prompt, visual)

    capacity = model.max_text_length if multimodal else model.max_length

    _validate_length(len(tokens), max_new_tokens, capacity)

    if multimodal:
        image_size = model.image_size

        if visual:
            if image.size != (image_size, image_size):
                raise ValueError("Unexpected image size")

            image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

        else:
            image_array = np.zeros((image_size, image_size, 3), dtype=np.float32)

        images = jnp.asarray(image_array[None, ...])
        has_image = jnp.asarray([visual])

    if max_new_tokens == 0:
        return _result([], False)

    prefix_length = len(tokens)

    padded_length = capacity

    # pre-defined
    input_ids = jnp.full((1, padded_length), ByteTokenizer.PAD, dtype=jnp.int32)
    input_ids = input_ids.at[0, :prefix_length].set(
        jnp.asarray(tokens, dtype=jnp.int32)
    )

    apply_model = jax.jit(type(model).apply, static_argnums=(0,))

    generated: list[int] = []
    stopped = False

    for step in range(max_new_tokens):
        if multimodal:
            logits = apply_model(
                model, {"params": params}, input_ids, images, has_image
            )
        else:
            logits = apply_model(model, {"params": params}, input_ids)

        position = jnp.asarray(prefix_length + step - 1, dtype=jnp.int32)
        last_logits = jax.lax.dynamic_index_in_dim(
            logits[0], position, axis=0, keepdims=False
        )
        next_id = _choose_token(last_logits)

        if next_id == ByteTokenizer.EOS:
            stopped = True
            break

        generated.append(next_id)

        if step + 1 < max_new_tokens:
            input_ids = input_ids.at[0, prefix_length + step].set(next_id)

    return _result(generated=generated, stopped=stopped)


CachedStep = Callable[[Any, Any, jax.Array], tuple[jax.Array, Any]]


def initialize_text_cache(model: CausalLM) -> Any:
    """Allocate an empty cache for every Transformer layer."""
    dummy_token = jnp.asarray([[ByteTokenizer.BOS]], dtype=jnp.int32)

    # Only the cache collection is used
    # Model parameters are provided separately at apply time
    variables = model.init(jax.random.key(0), dummy_token, decode=True)

    return variables["cache"]


def make_cached_step(model: CausalLM) -> CachedStep:
    """Compile one token's forward pass and cache updates."""

    @jax.jit
    def step(params: Any, cache: Any, token_ids: jax.Array) -> tuple[jax.Array, Any]:
        logits, updates = model.apply(
            {
                "params": params,
                "cache": cache,
            },
            token_ids,
            decode=True,
            mutable=["cache"],
        )

        return logits, updates["cache"]

    return step


def generate_cached(
    model: CausalLM, params: Any, prompt: str, *, max_new_tokens: int = 32
) -> GenerationResult:
    """Greedy decoding with one KV cache per Transformer Layer."""
    tokens = _prefix_tokens(prompt, has_image=False)

    _validate_length(
        len(tokens), max_new_tokens=max_new_tokens, capacity=model.max_length
    )

    if max_new_tokens == 0:
        return _result([], False)

    cache = initialize_text_cache(model)
    step = make_cached_step(model)

    # Prefill the cache sequentially with the entire prompt
    for token_id in tokens:
        one_token = jnp.asarray([[token_id]], dtype=jnp.int32)

        logits, cache = step(params, cache, one_token)

    generated: list[int] = []
    stopped = False

    for index in range(max_new_tokens):
        next_id = _choose_token(logits[0, -1])

        if next_id == ByteTokenizer.EOS:
            stopped = True
            break

        generated.append(next_id)

        # No need to compute another forward pass if this was the final requested output token
        if index + 1 < max_new_tokens:
            one_token = jnp.asarray([[next_id]], dtype=jnp.int32)

            logits, cache = step(params, cache, one_token)

    return _result(generated, stopped)
