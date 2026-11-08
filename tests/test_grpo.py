import jax.numpy as jnp
import numpy as np

from synthesize_vl.grpo.data import decode_rollout, group_advantages, score_rollout_groups
from synthesize_vl.opd.data import OPDRollouts
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.tokenizer import ByteTokenizer


def test_group_advantages():
    rewards = np.asarray([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)

    advantages = group_advantages(rewards)

    np.testing.assert_allclose(advantages[0], [1, -1, 1, -1], atol=1e-6)
    np.testing.assert_array_equal(advantages[1], np.zeros(4))
    np.testing.assert_array_equal(advantages[2], np.zeros(4))

    np.testing.assert_allclose(advantages.mean(axis=1), 0.0, atol=1e-6)


def _response_tokens(response: str, *, max_tokens: int = 40) -> tuple[np.ndarray, np.ndarray]:
    values = list(response.encode("utf-8")) + [ByteTokenizer.EOS]

    if len(values) > max_tokens:
        raise ValueError("Unit response too long")

    row = np.full((max_tokens,), ByteTokenizer.PAD, dtype=np.int32)
    mask = np.zeros((max_tokens,), dtype=bool)

    row[: len(values)] = values
    mask[: len(values)] = True

    return row, mask


def test_decode_requires_eos():
    tokens, valid = _response_tokens("Answer: 12\n")

    assert decode_rollout(tokens, valid) == "Answer: 12\n"

    no_eos = valid.copy()
    no_eos[np.flatnonzero(valid)[-1]] = False

    assert decode_rollout(tokens, no_eos) is None


def test_exact_reward_groups():
    config = SFTDataConfig(seed=27)

    example = generate_sft_example(
        config,
        split="validation",
        index=1,
        task_override="text_arithmetic",
        difficulty_override="easy",
        cot_override=False,
    )

    correct = f"Answer: {example.oracle_answer}\n"
    wrong = f"Answer: {example.oracle_answer + 1}\n"

    responses = [correct, wrong, correct, wrong]
    packed = [_response_tokens(response) for response in responses]

    tokens = jnp.asarray(np.stack([item[0] for item in packed]))
    mask = jnp.asarray(np.stack([item[1] for item in packed]))

    rollouts = OPDRollouts(
        input_ids=jnp.zeros((4, 64), dtype=jnp.int32),
        prefix_lengths=jnp.ones((4,), dtype=jnp.int32),
        images=jnp.zeros((4, 64, 64, 3), dtype=jnp.float32),
        has_image=jnp.zeros((4,), dtype=jnp.bool_),
        sampled_tokens=tokens,
        valid_mask=mask,
    )

    report = score_rollout_groups([example], rollouts, group_size=4)

    np.testing.assert_array_equal(report.rewards, [[1, 0, 1, 0]])
    np.testing.assert_allclose(report.advantages, [[1, -1, 1, -1]])

    assert report.has_signal
    assert report.informative_groups == 1
    assert report.correct_rate == 0.5
