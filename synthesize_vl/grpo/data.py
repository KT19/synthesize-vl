from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from synthesize_vl.opd.data import OPDPrompts, OPDRollouts, pack_opd_prompts
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_examples
from synthesize_vl.sft.schema import SFTExample, exact_answer_reward
from synthesize_vl.tokenizer import ByteTokenizer


@dataclass(frozen=True, slots=True)
class RewardReport:
    rewards: np.ndarray  # [problems, group_size]
    advantages: np.ndarray  # [problems, group_size]
    correct_rate: float
    informative_groups: int

    @property
    def has_signal(self) -> bool:
        return self.informative_groups > 0


def make_grop_prompts(
    config: SFTDataConfig, *, start: int, group_size: int, context_length: int, new_tokens: int
) -> tuple[list[SFTExample], OPDPrompts]:
    """
    Rows:
        problem0/sample0
        problem0/sample1
        ...
        problem1/sample0
        ...
    """
    examples = generate_sft_examples(config, split="train", start=start, count=config.batch_size, cot_override=True)

    base = pack_opd_prompts(examples, context_length=context_length, rollout_tokens=new_tokens)
    grouped = jax.tree_util.tree_map(lambda array: jnp.repeat(array, repeats=group_size, axis=0), base)

    return examples, grouped


def group_advantages(rewards: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Center and normalize rewards within each problem"""
    rewards = np.asarray(rewards, dtype=np.float32)

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected rewards [num_problems, group_size]")

    if not np.all(np.isfinite(rewards)):
        raise ValueError("Non-finite rewards")

    means = rewards.mean(axis=1, keepdims=True)
    deviations = rewards.std(axis=1, keepdims=True)

    return np.where(deviations > epsilon, (rewards - means) / np.maximum(deviations, epsilon), 0.0).astype(np.float32)


def decode_rollout(tokens: np.ndarray, valid_mask: np.ndarray) -> str | None:
    """Decode only a completed EOS-terminated answer"""

    valid = np.asarray(tokens)[np.asarray(valid_mask, dtype=bool)]
    eos_positions = np.flatnonzero(valid == ByteTokenizer.EOS)

    if len(eos_positions) != 1:
        return None

    eos_index = int(eos_positions[0])
    # EOS must be the last valid sampled token
    if eos_index != len(valid) - 1:
        return None

    payload = valid[:eos_index]

    if np.any(payload < 0) or np.any(payload > 255):
        return None

    try:
        return bytes(int(token) for token in payload).decode("utf-8")
    except UnicodeDecodeError:
        return None


def score_rollout_groups(examples: Sequence[SFTExample], rollouts: OPDRollouts, *, group_size: int) -> RewardReport:
    """Give binary oracle rewards to sampled, completed responses"""
    sampled = np.asarray(jax.device_get(rollouts.sampled_tokens))
    valid = np.asarray(jax.device_get(rollouts.valid_mask))

    if sampled.shape[0] != len(examples) * group_size:
        raise ValueError("Incorrect number of sampled responses")

    scores = np.zeros((len(examples), group_size), dtype=np.float32)

    for problem_index, example in enumerate(examples):
        for sample_index in range(group_size):
            row = problem_index * group_size + sample_index

            response = decode_rollout(sampled[row], valid[row])

            if response is None:
                continue

            scores[problem_index, sample_index] = exact_answer_reward(response, example.oracle_answer)

    advantages = group_advantages(scores)

    informative = int(np.sum(np.max(scores, axis=1) > np.min(scores, axis=1)))

    return RewardReport(
        rewards=scores, advantages=advantages, correct_rate=float(scores.mean()), informative_groups=informative
    )
