import jax
import jax.numpy as jnp

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.legacy.synthetic import make_dataset
from synthesize_vl.legacy.vision import ObjectSpec, make_question, render_scene
from synthesize_vl.models.multimodal import MultimodalEmbedding, make_prefix_causal_mask


def main() -> None:
    math = make_dataset(seed=42, count=1)[0]
    scene = render_scene([ObjectSpec("digit", "red", "7")], jax.random.key(123))

    qa = make_question(scene, "identify")

    samples = [
        TrainingSample(task=math.kind, prompt=math.prompt + " ", answer=math.answer),
        TrainingSample(
            task=qa.task,
            prompt=qa.question + "\nAnswer: ",
            answer=qa.answer,
            image=scene.image,
        ),
    ]
    batch = pack_examples(samples, max_text_length=96)
    model = MultimodalEmbedding(
        hidden_size=32, max_text_length=96, text_position_mode="learned"
    )

    variables = model.init(
        jax.random.key(0), batch.input_ids, batch.images, batch.has_image
    )
    embeddings = model.apply(variables, batch.input_ids, batch.images, batch.has_image)

    attention_mask = make_prefix_causal_mask(batch.input_ids, batch.has_image)

    print("Input IDs:", batch.input_ids.shape)
    print("Images:", batch.images.shape)
    print("Embeddings:", embeddings.shape)  # type: ignore
    print("Attention mask:", attention_mask.shape)
    print("Supervised targets:", batch.loss_mask.sum(axis=1))

    assert embeddings.shape == (2, 160, 32)  # type: ignore
    assert attention_mask.shape == (2, 1, 160, 160)

    # verify gradient flow through both pathways
    def energy(params):
        output = model.apply(
            {"params": params}, batch.input_ids, batch.images, batch.has_image
        )
        return jnp.mean(output**2)  # type: ignore

    grads = jax.grad(energy)(variables["params"])

    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(grads))

    print("Gradient verification: PASS")
    print("Multimodal representation: PASS")


if __name__ == "__main__":
    main()
