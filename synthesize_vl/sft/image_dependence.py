from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image

from synthesize_vl.sft.schema import SFTExample, exact_answer_reward, parse_final_answer

Predict = Callable[[str, Image.Image], str | bytes]

VisualPair = tuple[SFTExample, SFTExample]


@dataclass(frozen=True, slots=True)
class ImageDependentMetrics:
    pairs: int
    image_accuracy: float
    paired_accuracy: float
    blank_image_accuracy: float
    answer_change_rate: float


def evaluate_image_dependence(
    pairs: Sequence[VisualPair], predict: Predict
) -> ImageDependentMetrics:
    if not pairs:
        raise ValueError("At least one counterfactual pair is required")

    blank = Image.new("RGB", (64, 64), (248, 248, 248))

    correct = 0
    both_correct = 0
    blank_correct = 0
    changed = 0

    for low, high in pairs:
        if (
            low.task != "visual_arithmetic"
            or high.task != "visual_arithmetic"
            or low.image is None
            or high.image is None
            or low.prompt != high.prompt
            or low.oracle_answer == high.oracle_answer
        ):
            raise ValueError("Invalid counterfactual pair")

        low_prediction = predict(low.prompt, low.image)  # type: ignore
        high_prediction = predict(high.prompt, high.image)  # type: ignore

        low_correct = exact_answer_reward(low_prediction, low.oracle_answer)
        high_correct = exact_answer_reward(high_prediction, high.oracle_answer)

        correct += int(low_correct)
        correct += int(high_correct)

        both_correct += int(low_correct == 1.0 and high_correct == 1.0)

        # Diagnostic:
        changed += int(
            parse_final_answer(low_prediction) != parse_final_answer(high_prediction)
        )

        # Blank-image baseline
        blank_correct += int(
            exact_answer_reward(predict(low.prompt, blank), low.oracle_answer)
        )
        blank_correct += int(
            exact_answer_reward(predict(high.prompt, blank), high.oracle_answer)
        )

    count = len(pairs)

    return ImageDependentMetrics(
        pairs=count,
        image_accuracy=correct / (2 * count),
        paired_accuracy=both_correct / count,
        blank_image_accuracy=blank_correct / (2 * count),
        answer_change_rate=changed / count,
    )
