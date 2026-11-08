from typing import Any

from flax import linen as nn
from PIL import Image

from synthesize_vl.inference.generation import generate_greedy
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.sft.evaluation import evaluate_answers
from synthesize_vl.sft.stream import LEVELS, TASKS
from synthesize_vl.sft.train_config import SFTEvaluationConfig


def generate_sft_response(
    model: nn.Module,
    params: Any,
    prompt: str,
    image: Image.Image | None,
    *,
    max_new_tokens: int,
) -> str:
    """Return a complete generated response, requiring EOS."""
    result = generate_greedy(model, params=params, prompt=prompt, image=image, max_new_tokens=max_new_tokens)

    if not result.stopped_on_eos:
        return ""

    token_ids = tuple(int(token) for token in result.token_ids)

    # Our tokenizer uses 0..255 for bytes
    if any(not 0 <= token < 256 for token in token_ids):
        return ""

    return bytes(token_ids).decode("utf-8", errors="replace")


def evaluate_sft_generation(model: nn.Module, params: Any, data: SFTDataConfig, settings: SFTEvaluationConfig) -> None:
    """Evaluate direct-answer accuracy on test problems."""
    examples = []

    for task_index, task in enumerate(TASKS):
        for level_index, difficulty in enumerate(LEVELS):
            for i in range(settings.exact_examples_per_level):
                examples.append(
                    generate_sft_example(
                        data,
                        split="test",
                        index=700000 + task_index * 100000 + level_index * 10000 + i,
                        task_override=task,
                        difficulty_override=difficulty,
                        cot_override=True,
                    )
                )

    predictions = [
        generate_sft_response(
            model,
            params,
            example.prompt,
            example.image,
            max_new_tokens=settings.max_new_tokens,
        )
        for example in examples
    ]

    scores = evaluate_answers(examples, predictions)
    print("\nFree-running exact-answer accuracy")

    for name, score in sorted(scores.items()):
        print(f"{name:<32} {score:6.1%}")
