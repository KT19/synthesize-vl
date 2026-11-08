import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingSample
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.tokenizer import ByteTokenizer
from synthesize_vl.training.losses import masked_lm_loss


def small_model(position_mode="rope"):
    return CausalLM(
        hidden_size=32,
        num_heads=4,
        num_layers=1,
        intermediate_size=64,
        max_length=32,
        position_mode=position_mode,
    )


def test_model_output_shape():
    tokens = jnp.array([[256, 65, 66, 257], [256, 67, 68, 257]])

    model = small_model()
    variables = model.init(jax.random.key(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (2, 4, ByteTokenizer.VOCAB_SIZE)  # type: ignore
    assert bool(jnp.all(jnp.isfinite(logits)))  # type: ignore


def test_nope_mode():
    tokens = jnp.array([[256, 65, 66, 257]])
    model = small_model("nope")
    variables = model.init(jax.random.key(1), tokens)

    assert model.apply(variables, tokens).shape == (1, 4, 260)  # type: ignore


def test_future_tokens_do_not_change_past_logits():
    tokens = jnp.array([[256, 65, 66, 67, 68, 69]])

    model = small_model()
    variables = model.init(jax.random.key(2), tokens)

    logits_a = model.apply(variables, tokens)
    changed = tokens.at[0, 5].set(70)

    logits_b = model.apply(variables, changed)

    np.testing.assert_allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6, rtol=1e-6)  # type: ignore


def test_answer_targets_and_loss_mask():
    batch = pack_examples(
        [TrainingSample("math", "Q: 2+3:? A: ", "5")], max_text_length=32
    )

    labels = batch.target_ids[0][batch.loss_mask[0]]

    assert labels.tolist() == [ord("5"), ByteTokenizer.EOS]
    logits = jnp.zeros((1, 32, ByteTokenizer.VOCAB_SIZE))

    original_loss = masked_lm_loss(logits, batch)

    # Changing an unsupervised padding target
    modified = batch.replace(target_ids=batch.target_ids.at[0, -1].set(123))  # type: ignore
    changed_loss = masked_lm_loss(logits, modified)

    np.testing.assert_allclose(original_loss, changed_loss)
