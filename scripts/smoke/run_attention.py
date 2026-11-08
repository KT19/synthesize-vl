import jax
import jax.numpy as jnp

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.legacy.synthetic import make_dataset
from synthesize_vl.legacy.vision import ObjectSpec, make_question, render_scene
from synthesize_vl.models.attention import MultiHeadAttention, make_rope_positions
from synthesize_vl.models.multimodal import MultimodalEmbedding, make_prefix_causal_mask


def main() -> None:
    # General synthetic examples
    text_examples = make_dataset(seed=42, count=2)
    scene = render_scene([ObjectSpec("digit", "red", "7")], jax.random.key(7))
    qa = make_question(scene, "identify")

    samples = [
        TrainingSample(ex.kind, ex.prompt + " ", ex.answer) for ex in text_examples
    ]

    samples.append(
        TrainingSample(qa.task, qa.question + "\nAnswer: ", qa.answer, scene.image)
    )

    batch = pack_examples(samples, max_text_length=128)

    # Multmodal embeddings
    embedder = MultimodalEmbedding(
        hidden_size=32, max_text_length=128, text_position_mode="nope"
    )
    embed_vars = embedder.init(
        jax.random.key(0), batch.input_ids, batch.images, batch.has_image
    )
    x = embedder.apply(embed_vars, batch.input_ids, batch.images, batch.has_image)

    # Prefix-causal mask
    mask = make_prefix_causal_mask(batch.input_ids, batch.has_image)

    positions = make_rope_positions(text_length=batch.input_ids.shape[1])

    # Multi-head attention
    model = MultiHeadAttention(num_heads=4, position_mode="rope")
    variables = model.init(jax.random.key(1), x, mask, positions)
    y = model.apply(variables, x, mask, positions)

    # Gradient test
    def loss_fn(params):
        output = model.apply({"params": params}, x, mask, positions)
        return jnp.mean(output**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))

    print("Embedding shape:", x.shape)  # type: ignore
    print("Attention mask:", mask.shape)
    print("Output shape:", y.shape)  # type: ignore
    print("Attention parameters:", param_count)

    assert x.shape == y.shape == (3, 192, 32)  # type: ignore
    assert mask.shape == (3, 1, 192, 192)
    assert param_count == 4 * 32 * 32

    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))

    print("Pass")


if __name__ == "__main__":
    main()
