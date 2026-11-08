import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.embedding import TextEmbedding, VisionPatchEmbedding


def test_text_embedding_shape():
    model = TextEmbedding(hidden_size=32)

    tokens = jnp.array([[256, 65, 66, 257], [256, 67, 68, 257]], dtype=jnp.int32)

    variables = model.init(jax.random.key(0), tokens)
    output = model.apply(variables, tokens)

    assert output.shape == (2, 4, 32)  # type: ignore
    assert bool(jnp.all(jnp.isfinite(output)))  # type: ignore


def test_vision_embedding_shape():
    model = VisionPatchEmbedding(hidden_size=32, patch_size=8)

    images = jnp.ones((2, 64, 64, 3))

    variables = model.init(jax.random.key(0), images)
    output = model.apply(variables, images)

    assert output.shape == (2, 64, 32)  # type: ignore

    kernel = variables["params"]["patch_projection"]["kernel"]
    assert kernel.shape == (8, 8, 3, 32)  # type: ignore


def test_reproducible_initialization():
    model = TextEmbedding(hidden_size=16)

    tokens = jnp.array([[256, 65, 257]])

    variables_a = model.init(jax.random.key(42), tokens)
    variables_b = model.init(jax.random.key(42), tokens)

    output_a = model.apply(variables_a, tokens)
    output_b = model.apply(variables_b, tokens)

    np.testing.assert_array_equal(output_a, output_b)


def test_text_gradients():
    model = TextEmbedding(hidden_size=16)
    tokens = jnp.array([[256, 65, 257]])

    variables = model.init(jax.random.key(0), tokens)

    def loss_fn(params):
        output = model.apply({"params": params}, tokens)

        return jnp.mean(output**2)  # type: ignore

    grads = jax.grad(loss_fn)(variables["params"])
    embedding_grad = grads["token_embedding"]["embedding"]

    assert bool(jnp.all(jnp.isfinite(embedding_grad)))
    assert float(jnp.linalg.norm(embedding_grad[65])) > 0.0


def test_color_changes_visual_embeddings():
    key = jax.random.key(10)

    red = render_scene([ObjectSpec("digit", "red", "7")], key)
    blue = render_scene([ObjectSpec("digit", "blue", "7")], key)

    images = jnp.asarray(
        np.stack([np.asarray(red.image), np.asarray(blue.image)]).astype(np.float32)
        / 255.0
    )

    model = VisionPatchEmbedding(hidden_size=32)
    variables = model.init(jax.random.key(0), images)

    output = model.apply(variables, images)

    assert output.shape == (2, 64, 32)  # type: ignore
    assert not np.allclose(output[0], output[1])  # type: ignore


def test_invalid_image_size():
    model = VisionPatchEmbedding(hidden_size=32, patch_size=8)

    images = jnp.ones((1, 62, 64, 3))
    with pytest.raises(ValueError):
        model.init(jax.random.key(0), images)
