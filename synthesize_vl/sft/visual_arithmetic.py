from dataclasses import dataclass

import numpy as np
from PIL import Image

from synthesize_vl.pretraining.world import (
    COLORS,
    SHAPES,
    Color,
    Shape,
    WorldObject,
    render_world,
)
from synthesize_vl.sft.schema import Difficulty


@dataclass(frozen=True, slots=True)
class ScoringTerm:
    color: Color
    shape: Shape
    weight: int


@dataclass(frozen=True, slots=True)
class VisualArithmeticProblem:
    image: Image.Image
    objects: tuple[WorldObject, ...]
    terms: tuple[ScoringTerm, ...]
    prompt_rules: str
    steps: tuple[str, ...]
    answer: int


# Min objects, max objects, min weighted categories,
# max weighted categories, min |weight|, max |weight|
VISUAL_LEVELS = {
    "easy": (2, 3, 1, 1, 0, 10),
    "medium": (4, 6, 2, 2, 0, 30),
    "hard": (6, 8, 2, 3, 0, 50),
    "ood": (8, 9, 3, 3, 51, 100),
}


def count_matching(objects: tuple[WorldObject, ...], term: ScoringTerm) -> int:
    return sum(obj.color == term.color and obj.shape == term.shape for obj in objects)


def calculate_visual_score(objects: tuple[WorldObject, ...], terms: tuple[ScoringTerm, ...]) -> int:
    return sum(count_matching(objects, term) * term.weight for term in terms)


def _plural(shape: Shape) -> str:
    return {"circle": "circles", "square": "squares", "triangle": "triangles"}[shape]


def _make_steps(objects: tuple[WorldObject, ...], terms: tuple[ScoringTerm, ...]) -> tuple[str, ...]:
    """Generate verified perception and arithmetic steps."""
    steps: list[str] = []
    contributions: list[int] = []

    for term in terms:
        count = count_matching(objects, term)

        if count < 1:
            raise ValueError("Scoring category is absent from the image")

        steps.append(f"{term.color} {_plural(term.shape)}: {count}")

    for term in terms:
        count = count_matching(objects, term)
        contribution = count * term.weight

        steps.append(f"{count} * {term.weight} = {contribution}")

        contributions.append(contribution)

    running = contributions[0]

    for contribution in contributions[1:]:
        total = running + contribution

        steps.append(f"{running} + {contribution} = {total}")

        running = total

    if running != calculate_visual_score(objects, terms):
        raise ValueError("Visual arithmetic trace failed verification")

    return tuple(steps)


def sample_visual_arithmetic(rng: np.random.Generator, difficulty: Difficulty) -> VisualArithmeticProblem:
    """Generate a legible scene, rules and a correct solution."""
    if difficulty not in VISUAL_LEVELS:
        raise ValueError("Unknown visual difficulty")

    min_objects, max_objects, min_terms, max_terms, min_weight, max_weight = VISUAL_LEVELS[difficulty]

    term_count = int(rng.integers(min_terms, max_terms + 1))

    # Vary each scored category's count within the scene's object limit.
    counts = rng.integers(1, 4, size=term_count)
    while int(counts.sum()) > max_objects:
        counts = rng.integers(1, 4, size=term_count)

    categories = [(color, shape) for color in COLORS for shape in SHAPES]
    selected = rng.choice(len(categories), size=term_count, replace=False)
    scored_pairs = [categories[int(i)] for i in selected]

    pairs = [pair for pair, repeats in zip(scored_pairs, counts, strict=True) for _ in range(int(repeats))]
    distractor_count = int(rng.integers(0, max_objects - len(pairs) + 1))
    distractors = [pair for pair in categories if pair not in scored_pairs]
    pairs.extend(distractors[int(rng.integers(len(distractors)))] for _ in range(distractor_count))

    # Fill small scenes with scored objects so zero distractors remains possible.
    pairs.extend(scored_pairs[int(rng.integers(term_count))] for _ in range(max(0, min_objects - len(pairs))))
    count = len(pairs)

    rng.shuffle(pairs)

    slots = rng.choice(9, size=count, replace=False)
    objects = tuple(
        WorldObject(color=pair[0], shape=pair[1], row=int(slot) // 3, col=int(slot) % 3)  # type: ignore
        for pair, slot in zip(pairs, slots, strict=True)
    )

    terms: list[ScoringTerm] = []

    for color, shape in scored_pairs:
        magnitude = int(rng.integers(min_weight, max_weight + 1))

        if rng.random() < 0.35:
            magnitude = -magnitude

        terms.append(ScoringTerm(color=color, shape=shape, weight=magnitude))  # type: ignore

    orcle_terms = tuple(terms)
    rules = "; ".join(f"each {term.color} {term.shape} is worth {term.weight} points" for term in orcle_terms)

    rules += "; all other objects are worth 0 points"

    answer = calculate_visual_score(objects, orcle_terms)
    steps = _make_steps(objects, orcle_terms)

    return VisualArithmeticProblem(
        image=render_world(objects, rng),
        objects=objects,
        terms=orcle_terms,
        prompt_rules=rules,
        steps=steps,
        answer=answer,
    )
