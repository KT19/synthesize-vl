import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from synthesize_vl.inference.generation import (
    generate_cached,
    generate_greedy,
    initialize_text_cache,
    make_cached_step,
)
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.tokenizer import ByteTokenizer


def small_model(position_mode: str = "rope", use_gate: bool = True) -> CausalLM:
    return CausalLM(
        hidden_size=16,
        num_heads=2,
        num_layers=2,
        intermediate_size=32,
        max_length=32,
        position_mode=position_mode,
        use_attention_gate=use_gate,
    )


@pytest.mark.parametrize("position_mode,use_gate", [("rope", True), ("nope", False)])
def test_cached_logits_match_full_forward(position_mode: str, use_gate: bool) -> None:
    with jax.default_matmul_precision("highest"):
        model = small_model(position_mode, use_gate)

        tokens = jnp.asarray([[256, ord("H"), ord("i"), ord("!")]], dtype=jnp.int32)

        params = model.init(jax.random.key(0), tokens)["params"]

        cache = initialize_text_cache(model)
        cached_step = make_cached_step(model)

        for length in range(1, tokens.shape[1] + 1):
            newest = tokens[:, length - 1 : length]

            cached_logits, cache = cached_step(params, cache, newest)

            full_logits = model.apply({"params": params}, tokens[:, :length])

            np.testing.assert_allclose(cached_logits[0, -1], full_logits[0, -1], atol=1e-4, rtol=1e-4)  # type: ignore

        # Every transformer layer has consumed four tokens
        for layer in range(model.num_layers):
            index = cache[f"block_{layer}"]["attention"]["cache_index"]

            assert int(index) == 4


def test_cached_greedy_matches_reference() -> None:
    model = small_model()

    tokens = jnp.asarray([[256, 65, 66]])
    params = model.init(jax.random.key(1), tokens)["params"]

    reference = generate_greedy(model, params, "AB", max_new_tokens=6)
    cached = generate_cached(model, params, "AB", max_new_tokens=6)

    assert reference.token_ids == cached.token_ids
    assert reference.stopped_on_eos == cached.stopped_on_eos


class AlwaysEOS(nn.Module):
    max_length: int = 16

    @nn.compact
    def __call__(self, input_ids: jax.Array) -> jax.Array:
        logits = jnp.full((input_ids.shape[0], input_ids.shape[1], ByteTokenizer.VOCAB_SIZE), -100.0)

        return logits.at[..., ByteTokenizer.EOS].set(100.0)


def test_generation_stops_on_eos() -> None:
    result = generate_greedy(AlwaysEOS(), {}, "Hi", max_new_tokens=5)

    assert result.token_ids == ()
    assert result.text == ""
    assert result.stopped_on_eos


def test_reference_recurrent_multimodal_generation() -> None:
    scene = render_scene([ObjectSpec("digit", "blue", "3")], jax.random.key(3))

    model = RecurrentMultimodalCausalLM(
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        max_text_length=48,
        recurrent_steps=1,
        residual_type="mhc",
        num_streams=2,
        core_attention="hybrid",
        local_window=4,
    )

    prompt = "Digit?\nAnswer: "

    input_ids = jnp.asarray([[ByteTokenizer.BOS, ByteTokenizer.IMAGE, *prompt.encode("utf-8")]])

    pixels = np.asarray(scene.image, dtype=np.float32)[None, ...] / 255.0

    params = model.init(jax.random.key(4), input_ids, jnp.asarray(pixels), jnp.asarray([True]))["params"]

    result = generate_greedy(model, params, prompt, image=scene.image, max_new_tokens=2)

    assert len(result.token_ids) <= 2
    assert all(0 <= t < 256 for t in result.token_ids)
    assert isinstance(result.text, str)
