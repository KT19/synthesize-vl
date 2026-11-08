import jax
import jax.numpy as jnp

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.legacy.vision import ObjectSpec, render_scene
from synthesize_vl.models.multimodal import MultimodalEmbedding, make_prefix_causal_mask
from synthesize_vl.tokenizer import ByteTokenizer


def make_test_batch():
    scene = render_scene([ObjectSpec("digit", "blue", "3")], jax.random.key(0))
    samples = [
        TrainingSample("math", "Compute: 2+3\nA: ", "5"),
        TrainingSample("vision", "Which digit?\nA: ", "3", scene.image),
    ]

    return pack_examples(samples, max_text_length=64)


def test_batch_structure():
    batch = make_test_batch()

    assert batch.input_ids.shape == (2, 64)
    assert batch.images.shape == (2, 64, 64, 3)
    assert batch.has_image.tolist() == [False, True]


def test_image_marker():
    batch = make_test_batch()

    assert int(batch.input_ids[1, 0]) == ByteTokenizer.BOS
    assert int(batch.input_ids[1, 1]) == ByteTokenizer.IMAGE


def test_supervised_answer_tokens():
    batch = pack_examples(
        [TrainingSample("math", "Q: 2+3? A: ", "5")], max_text_length=32
    )

    labels = batch.target_ids[0][batch.loss_mask[0]]

    assert labels.tolist() == [ord("5"), ByteTokenizer.EOS]


def test_text_cannot_see_future():
    batch = make_test_batch()
    mask = make_prefix_causal_mask(batch.input_ids, batch.has_image)

    # First text token
    assert not bool(mask[1, 0, 64, 65])
    assert bool(mask[1, 0, 65, 64])


def test_image_attention():
    batch = make_test_batch()

    mask = make_prefix_causal_mask(batch.input_ids, batch.has_image)

    # Image patches can see other image patches

    assert bool(mask[1, 0, 0, 63])
    assert bool(mask[1, 0, 63, 0])

    # Image patches cannot see text
    assert not bool(mask[1, 0, 0, 64])

    # Valid text can see the image
    assert bool(mask[1, 0, 64, 0])

    # Text-only samples cannot see inactive patches
    assert not bool(mask[0, 0, 64, 0])


def test_no_fully_masked_rows():
    batch = make_test_batch()
    mask = make_prefix_causal_mask(batch.input_ids, batch.has_image)

    assert bool(jnp.all(jnp.any(mask, axis=-1)))


def test_positional_modes():
    batch = make_test_batch()

    for mode in ("learned", "nope"):
        model = MultimodalEmbedding(
            hidden_size=16, max_text_length=64, text_position_mode=mode
        )
        variables = model.init(
            jax.random.key(42), batch.input_ids, batch.images, batch.has_image
        )

        output = model.apply(variables, batch.input_ids, batch.images, batch.has_image)

        assert output.shape == (2, 128, 16)  # type: ignore
        assert bool(jnp.all(jnp.isfinite(output)))  # type: ignore

        assert ("text_positions" in variables["params"]) == (mode == "learned")


def test_jit_compatible_batch():
    batch = make_test_batch()

    @jax.jit
    def count_targets(data):
        return jnp.sum(data.loss_mask)

    assert int(count_targets(batch)) > 0
