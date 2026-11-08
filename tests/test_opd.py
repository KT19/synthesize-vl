import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.opd.core import allowed_logits, reverse_kl
from synthesize_vl.opd.data import pack_opd_prompts
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.tokenizer import ByteTokenizer


def test_reverse_kl_is_zero_for_matching_logits():
    logits = jnp.asarray([1.0, 2.0, -1.0] + [0.0] * 257, dtype=jnp.float32)

    loss = reverse_kl(logits, logits)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_reverse_kl_is_positive_for_different_logits():
    student = jnp.zeros((1, 260))
    teacher = student.at[0, 5].set(4.0)

    loss = reverse_kl(student, teacher)
    assert float(loss[0]) > 0

    gradient = jax.grad(lambda logits: jnp.sum(reverse_kl(logits, teacher)))(student)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0


def test_sampling_masks_structural_tokens():
    logits = jnp.zeros((260,), dtype=jnp.float32)

    logits = logits.at[ByteTokenizer.BOS].set(100.0)
    logits = logits.at[ByteTokenizer.PAD].set(100.0)
    logits = logits.at[ByteTokenizer.IMAGE].set(100.0)

    masked = allowed_logits(logits)

    for token in (ByteTokenizer.BOS, ByteTokenizer.PAD, ByteTokenizer.IMAGE):
        assert float(masked[token]) < -1e8

    assert float(masked[ByteTokenizer.EOS]) == 0


def make_tiny_opd_inputs():
    config = SFTDataConfig(seed=42)

    examples = [
        generate_sft_example(
            config,
            split="validation",
            index=0,
            task_override="text_arithmetic",
            difficulty_override="easy",
            cot_override=False,
        ),
        generate_sft_example(
            config,
            split="validation",
            index=1,
            task_override="visual_arithmetic",
            difficulty_override="easy",
            cot_override=False,
        ),
    ]

    batch = pack_opd_prompts(examples, context_length=512, rollout_tokens=4)

    return batch


def test_prompt_batch_contains_no_oracle_response():
    batch = make_tiny_opd_inputs()

    assert batch.input_ids.shape == (2, 512)
    assert batch.images.shape == (2, 64, 64, 3)

    np.testing.assert_array_equal(batch.has_image, [False, True])

    for row in range(2):
        prefix_length = int(batch.prefix_lengths[row])

        assert int(batch.input_ids[row, 0]) == ByteTokenizer.BOS
        assert bool(jnp.all(batch.input_ids[row, prefix_length:] == ByteTokenizer.PAD))

    assert int(batch.input_ids[1, 1]) == ByteTokenizer.IMAGE
