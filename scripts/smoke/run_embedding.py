import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.embedding import TextEmbedding, VisionPatchEmbedding
from synthesize_vl.tokenizer import ByteTokenizer


def main() -> None:
    hidden_size = 64

    # Text input
    tokenizer = ByteTokenizer()

    token_ids = jnp.asarray([tokenizer.encode("Which digit is red?")], dtype=jnp.int32)

    # Vision input
    scene = render_scene([ObjectSpec("digit", "red", "7")], jax.random.key(42))

    # Normalize RGB
    image = np.asarray(scene.image, dtype=np.float32) / 255.0
    images = jnp.asarray(image[None, ...])

    # Initialize modules
    text_model = TextEmbedding(hidden_size=hidden_size)
    vision_model = VisionPatchEmbedding(hidden_size=hidden_size, patch_size=8)

    key_text, key_vision = jax.random.split(jax.random.key(0))

    text_vars = text_model.init(key_text, token_ids)
    vision_vars = vision_model.init(key_vision, images)

    # Forward passes
    text_vectors = text_model.apply(text_vars, token_ids)
    vision_vectors = vision_model.apply(vision_vars, images)

    print("Text input:", token_ids.shape)
    print("Image input:", images.shape)
    print("Text embeddings:", text_vectors.shape)  # type: ignore
    print("Vision embeddings:", vision_vectors.shape)  # type: ignore

    assert text_vectors.shape[-1] == hidden_size  # type: ignore
    assert vision_vectors.shape == (1, 64, hidden_size)  # type: ignore

    # Gradient verification
    def vision_loss(params):
        output = vision_model.apply({"params": params}, images)
        return jnp.mean(output**2)  # type: ignore

    grads = jax.grad(vision_loss)(vision_vars["params"])

    kernel_grad = grads["patch_projection"]["kernel"]

    print("Vision kernel:", kernel_grad.shape)
    print("Gradient norm:", float(jnp.linalg.norm(kernel_grad)))

    assert bool(jnp.all(jnp.isfinite(kernel_grad)))
    assert float(jnp.linalg.norm(kernel_grad)) > 0.0

    print("Embedding demonstration: PASS")


if __name__ == "__main__":
    main()
