from collections import defaultdict
from collections.abc import Sequence

from synthesize_vl.sft.schema import SFTExample, exact_answer_reward


def evaluate_answers(
    examples: Sequence[SFTExample], predictions: Sequence[str | bytes]
) -> dict[str, float]:
    if len(examples) != len(predictions):
        raise ValueError("Number of predictions does not match examples")

    if not examples:
        raise ValueError("Cannot evaluate an empty collection")

    correct: dict[str, float] = defaultdict(float)
    total: dict[str, int] = defaultdict(int)

    for example, prediction in zip(examples, predictions):
        reward = exact_answer_reward(prediction, example.oracle_answer)

        groups = ("overall", example.task, f"{example.task}/{example.difficulty}")

        for group in groups:
            correct[group] += reward
            total[group] += 1

    return {group: correct[group] / total[group] for group in total}
