import numpy as np
import pytest

from synthesize_vl.sft.arithmetic import (
    LEVELS,
    Number,
    render_expression,
    sample_expression,
    verify_expression,
)
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.sft.evaluation import evaluate_answers
from synthesize_vl.sft.schema import (
    exact_answer_reward,
    format_response,
    parse_final_answer,
)
from synthesize_vl.sft.visual_arithmetic import (
    VISUAL_LEVELS,
    calculate_visual_score,
    count_matching,
    sample_visual_arithmetic,
)


def test_answer_parser() -> None:
    assert parse_final_answer("Work:\n2 + 3 = 5\nAnswer: 5\n") == 5
    assert parse_final_answer("Answer: -13") == -13
    assert parse_final_answer(b"Answer: 42\n") == 42
    assert parse_final_answer("Work:\n2 + 3 = 5\n") is None
    assert exact_answer_reward("Answer: 5\n", 5) == 1.0
    assert exact_answer_reward("Answer: 6\n", 5) == 0.0


def test_response_format() -> None:
    result = format_response(
        answer=14, steps=("3 + 4 = 7", "7 * 2 = 14"), show_work=True
    )

    assert result == ("Work:\n3 + 4 = 7\n7 * 2 = 14\nAnswer: 14\n")

    assert format_response(answer=14, steps=(), show_work=False) == "Answer: 14\n"


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard", "ood"))
def test_expression_oracle(difficulty) -> None:
    minimum, maximum, max_operand, max_intermediate = LEVELS[difficulty]
    operation_counts = set()

    for seed in range(100):
        expression = sample_expression(np.random.default_rng(seed), difficulty)

        answer, steps = verify_expression(expression)

        assert answer == expression.value
        assert minimum <= len(steps) <= maximum
        operation_counts.add(len(steps))

        assert render_expression(expression)
        assert all(" = " in step for step in steps)

        nodes = [expression]
        while nodes:
            node = nodes.pop()
            if isinstance(node, Number):
                assert abs(node.value) <= max_operand
            else:
                assert abs(node.value) <= max_intermediate
                nodes.extend((node.left, node.right))

    assert operation_counts == set(range(minimum, maximum + 1))


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard", "ood"))
def test_visual_arithmetic_oracle(difficulty) -> None:
    (min_objects, max_objects, min_terms, max_terms, min_weight, max_weight) = VISUAL_LEVELS[difficulty]
    distractor_counts = set()

    for seed in range(30):
        problem = sample_visual_arithmetic(np.random.default_rng(seed), difficulty)

        assert problem.image.size == (64, 64)

        assert min_objects <= len(problem.objects) <= max_objects
        assert min_terms <= len(problem.terms) <= max_terms
        assert all(min_weight <= abs(term.weight) <= max_weight for term in problem.terms)
        assert problem.answer == calculate_visual_score(problem.objects, problem.terms)

        assert all(count_matching(problem.objects, term) >= 1 for term in problem.terms)
        scored_pairs = {(term.color, term.shape) for term in problem.terms}

        distractor_counts.add(
            sum((obj.color, obj.shape) not in scored_pairs for obj in problem.objects)
        )
        assert len(problem.steps) == (2 * len(problem.terms) + len(problem.terms) - 1)

    assert 0 in distractor_counts
    assert any(count > 0 for count in distractor_counts)


def test_ood_difficulty_exceeds_training_ranges() -> None:
    training_levels = ("easy", "medium", "hard")
    assert LEVELS["ood"][0] > max(LEVELS[level][1] for level in training_levels)
    assert VISUAL_LEVELS["ood"][4] > max(VISUAL_LEVELS[level][5] for level in training_levels)


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard", "ood"))
def test_visual_arithmetic_mitigates_sum_of_weights_shortcut(difficulty) -> None:
    shortcut_correct = 0
    samples = 200

    for seed in range(samples):
        problem = sample_visual_arithmetic(np.random.default_rng(seed), difficulty)
        shortcut_correct += problem.answer == sum(term.weight for term in problem.terms)

    assert shortcut_correct / samples < 0.65


@pytest.mark.parametrize("difficulty", ("medium", "hard", "ood"))
def test_multiple_scored_categories_can_repeat(difficulty) -> None:
    observed_counts = set()

    for seed in range(100):
        problem = sample_visual_arithmetic(np.random.default_rng(seed), difficulty)
        counts = tuple(
            sorted(count_matching(problem.objects, term) for term in problem.terms)
        )
        observed_counts.add(counts)

    if difficulty == "medium":
        assert (2, 3) in observed_counts
    else:
        assert any(len(counts) == 3 and min(counts) > 1 for counts in observed_counts)


def test_evaluation_uses_exact_final_answers() -> None:
    config = SFTDataConfig(seed=5)

    examples = [
        generate_sft_example(
            config,
            split="test",
            index=i,
            task_override="text_arithmetic" if i == 0 else "visual_arithmetic",
        )
        for i in range(2)
    ]

    predictions = [
        f"Answer: {examples[0].oracle_answer}\n",
        f"Answer: {examples[1].oracle_answer + 1}\n",
    ]
    scores = evaluate_answers(examples, predictions)

    assert scores["overall"] == 0.5
    assert scores["text_arithmetic"] == 1.0
    assert scores["visual_arithmetic"] == 0.0
