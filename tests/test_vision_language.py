import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.legacy.synthetic import make_dataset
from synthesize_vl.legacy.vision import ObjectSpec, make_question, render_scene
from synthesize_vl.models.vision_language import MultimodalCausalLM
from synthesize_vl.training.losses import masked_lm_loss


def small_model(position_mode: str = "rope") -> MultimodalCausalLM:
    return MultimodalCausalLM(
        hidden_size=32,
        num_heads=4,
        num_layers=1,
        intermediate_size=64,
        max_text_length=128,
        position_mode=position_mode,
    )


def make_tiny_multimodal_batch(seed: int = 42) -> TrainingBatch:
    """Create reproducible text, math, code, and vision examples."""
    text_examples = make_dataset(seed=seed, count=2)
    samples = [
        TrainingSample(
            task=example.kind, prompt=example.prompt + " ", answer=example.answer
        )
        for example in text_examples
    ]

    # Cross digit identity and color so color alone cannot distinguish 3 from 7.
    visual_specs = [
        ObjectSpec("digit", "red", "3"),
        ObjectSpec("digit", "red", "7"),
        ObjectSpec("digit", "blue", "3"),
        ObjectSpec("digit", "blue", "7"),
        ObjectSpec("circle", "green"),
        ObjectSpec("square", "purple"),
    ]

    root_key = jax.random.key(seed)

    for index, spec in enumerate(visual_specs):
        key = jax.random.fold_in(root_key, index + 100)
        scene = render_scene([spec], key)
        qa = make_question(scene, "identify")
        samples.append(
            TrainingSample(
                task=qa.task,
                prompt=qa.question + "\nAnswer: ",
                answer=qa.answer,
                image=scene.image,
            )
        )

    return pack_examples(samples, max_text_length=128)


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the inputs accepted by the multimodal language models."""
    return (batch.input_ids, batch.images, batch.has_image)


@pytest.mark.parametrize("position_mode", ["rope", "nope", "learned"])
def test_multimodal_output_shape(position_mode: str) -> None:
    batch = make_tiny_multimodal_batch()
    model = small_model(position_mode)

    variables = model.init(jax.random.key(0), *model_inputs(batch))
    logits = model.apply(variables, *model_inputs(batch))

    assert logits.shape == (8, 128, 260)  # type: ignore
    assert bool(jnp.all(jnp.isfinite(logits)))  # type: ignore


def test_visual_input_affects_predictions() -> None:
    batch = make_tiny_multimodal_batch()
    model = small_model()

    variables = model.init(jax.random.key(1), *model_inputs(batch))

    logits = model.apply(variables, *model_inputs(batch))

    # Example 2 and 3 have the same digit question
    first = int(jnp.argmax(batch.loss_mask[2]))
    assert first == int(jnp.argmax(batch.loss_mask[3]))

    np.testing.assert_array_equal(
        batch.input_ids[2, : first + 1], batch.input_ids[3, : first + 1]
    )

    # Their visual information can affect logits
    difference = jnp.linalg.norm(logits[2, first] - logits[3, first])  # type: ignore

    assert float(difference) > 1e-8


def test_text_only_ignores_image_pixels() -> None:
    batch = make_tiny_multimodal_batch()
    model = small_model()

    variables = model.init(jax.random.key(2), *model_inputs(batch))
    original = model.apply(variables, *model_inputs(batch))

    # Example 0 is text-only, Replace its unused image tensor with nonzero pixels
    modified_images = batch.images.at[0].set(jnp.ones_like(batch.images[0]))

    changed = model.apply(variables, batch.input_ids, modified_images, batch.has_image)
    np.testing.assert_allclose(original[0], changed[0], atol=1e-6, rtol=1e-6)


def test_vision_encoder_receives_gradients() -> None:
    batch = make_tiny_multimodal_batch()
    model = small_model()

    variables = model.init(jax.random.key(3), *model_inputs(batch))

    def loss_fn(params):
        logits = model.apply({"params": params}, *model_inputs(batch))

        return masked_lm_loss(logits, batch)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])

    patch_grad = grads["multimodal_embedding"]["vision_embedding"]["patch_projection"][
        "kernel"
    ]

    assert bool(jnp.all(jnp.isfinite(patch_grad)))
    assert float(jnp.linalg.norm(patch_grad)) > 0.0
