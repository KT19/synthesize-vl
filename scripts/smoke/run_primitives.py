import jax
import jax.numpy as jnp

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.multimodal import MultimodalEmbedding
from synthesize_vl.models.primitives import RMSNorm, SwiGLU, rotate_qk


def main() -> None:
    scene = render_scene([ObjectSpec("digit", "red", "7")], jax.random.key(11))
    batch = pack_examples(
        [
            TrainingSample("math", "Compute: 2 + 3\nAnswer: ", "5"),
            TrainingSample("vision", "Which digit!?\nAnswer: ", "7", scene.image),
        ],
        max_text_length=64,
    )

    # Use NoPE text embeddings because we'll test RoPE
    embedding = MultimodalEmbedding(
        hidden_size=32, max_text_length=64, text_position_mode="nope"
    )
    embedding_vars = embedding.init(
        jax.random.key(0), batch.input_ids, batch.images, batch.has_image
    )

    x = embedding.apply(embedding_vars, batch.input_ids, batch.images, batch.has_image)

    # RMSNorm
    norm = RMSNorm()
    norm_vars = norm.init(jax.random.key(1), x)
    x_norm = norm.apply(norm_vars, x)

    # SwiGLU
    mlp = SwiGLU(intermediate_size=64)
    mlp_vars = mlp.init(jax.random.key(2), x_norm)
    y = mlp.apply(mlp_vars, x_norm)

    # RoPE
    q = x_norm.reshape(2, 128, 4, 8)  # type: ignore
    # Image pathes use zero rotary position
    positions = jnp.concatenate(
        [jnp.zeros(64, dtype=jnp.int32), jnp.arange(64, dtype=jnp.int32)]
    )

    q_rot, k_rot = rotate_qk(q, q, positions, mode="rope")

    # Verify gradients
    def loss_fn(params):
        out = mlp.apply({"params": params}, x_norm)

        return jnp.mean(out**2)  # type: ignore

    grads = jax.grad(loss_fn)(mlp_vars["params"])

    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    print("Multimodal embeddings:", x.shape)  # type: ignore
    print("RMSNorm output:", x_norm.shape)  # type: ignore
    print("SwiGLU output:", y.shape)  # type: ignore
    print("Rotated Q/K:", q_rot.shape, k_rot.shape)
    print("SwiGLU gradient norm:", float(grad_norm))

    assert x.shape == x_norm.shape == y.shape == (2, 128, 32)  # type: ignore
    assert q_rot.shape == k_rot.shape == (2, 128, 4, 8)  # type: ignore
    assert bool(jnp.isfinite(grad_norm) & (grad_norm > 0))

    print("PASS")


if __name__ == "__main__":
    main()
