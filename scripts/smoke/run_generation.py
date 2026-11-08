import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.inference.generation import (
    _prefix_tokens,
    generate_cached,
    generate_greedy,
)
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.models.transformer import CausalLM


def main() -> None:
    text_model = CausalLM(
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=64,
        max_length=64,
        position_mode="rope",
        use_attention_gate=True,
    )

    prompt = "Q: 2+3="

    variables = text_model.init(
        jax.random.key(0), jnp.asarray([_prefix_tokens(prompt, False)])
    )

    params = variables["params"]

    reference = generate_greedy(text_model, params, prompt, max_new_tokens=8)

    cached = generate_cached(text_model, params, prompt, max_new_tokens=8)

    print("Full-prefix IDs:", reference.token_ids)
    print("Cached IDs:  ", cached.token_ids)

    assert reference.token_ids == cached.token_ids
    assert reference.stopped_on_eos == cached.stopped_on_eos

    print("PASS")

    # Recurrent multimodal model
    scene = render_scene([ObjectSpec("digit", "red", "7")], jax.random.key(10))

    vision_model = RecurrentMultimodalCausalLM(
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        max_text_length=64,
        recurrent_steps=1,
        residual_type="mhc",
        num_streams=2,
        core_attention="hybrid",
        local_window=8,
    )

    vision_prompt = "Which digit?\nAnswer: "

    visual_ids = jnp.asarray([_prefix_tokens(vision_prompt, True)])
    pixels = np.asarray(scene.image, dtype=np.float32)[None, ...] / 255.0

    vision_vars = vision_model.init(
        jax.random.key(11), visual_ids, jnp.asarray(pixels), jnp.asarray([True])
    )

    vision_result = generate_greedy(
        vision_model,
        vision_vars["params"],
        vision_prompt,
        image=scene.image,
        max_new_tokens=3,
    )

    print("Visual output IDs:", vision_result.token_ids)
    print("Visual output text:", repr(vision_result.text))
    print("PASS")


if __name__ == "__main__":
    main()
