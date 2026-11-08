import jax.numpy as jnp

from synthesize_vl.models.attention_masks import make_hybrid_local_global_mask


def main() -> None:
    tokens = jnp.asarray(
        [[256, 65, 66, 67, 68, 69, 70, 258], [256, 65, 66, 67, 68, 69, 70, 258]],
        dtype=jnp.int32,
    )

    has_image = jnp.asarray([True, False])

    mask = make_hybrid_local_global_mask(
        tokens, has_image, n_visual_tokens=2, local_window=2, global_text_tokens=1
    )

    assert mask.shape == (2, 1, 10, 10)

    # Last valid text query can access both image patches
    assert bool(mask[0, 0, 8, 0])
    assert bool(mask[0, 0, 8, 1])

    # Text-only samples cannot access inactive image slots
    assert not bool(mask[1, 0, 8, 0])

    # A distant ordinary text key is blocked
    assert not bool(mask[0, 0, 8, 4])

    # A recent text key and global anchor are allowed
    assert bool(mask[0, 0, 8, 7])
    assert bool(mask[0, 0, 8, 2])

    print("Hybrid mask shape:", mask.shape)
    print("PASS")


if __name__ == "__main__":
    main()
