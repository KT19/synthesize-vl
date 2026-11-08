from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from synthesize_vl.pretraining.world import (
    COLORS,
    Color,
    Shape,
    WorldObject,
    sample_world,
)

VisualFamily = Literal["inventory", "color_counts", "spatial", "weighted_sum"]
VisualProfile = Literal[
    "mixed",
    "foundation",
    "intermediate",
    "advanced",
    "heldout_objects",
    "heldout_weights",
]

FAMILIES: tuple[VisualFamily, ...] = (
    "inventory",
    "color_counts",
    "spatial",
    "weighted_sum",
)
TRAIN_PROFILES = ("foundation", "intermediate", "advanced")

# Minimum objects, maximum objects, minimum weight, maximum weight
DIFFICULTIES: dict[str, tuple[int, int, int, int]] = {
    "foundation": (2, 3, 1, 4),
    "intermediate": (3, 5, 1, 4),
    "advanced": (5, 6, 1, 4),
    "heldout_objects": (7, 8, 1, 4),
    "heldout_weights": (3, 6, 5, 9),
}

_SPLIT_IDS = {"train": 401, "validation": 503, "test": 607}


@dataclass(frozen=True, slots=True)
class ScoreTerm:
    color: Color
    shape: Shape
    weight: int


@dataclass(frozen=True, slots=True)
class VisualDocument:
    sample_id: str
    split: Literal["train", "validation", "test"]
    family: str
    image: Image.Image
    objects: tuple[WorldObject, ...]
    prefix: bytes
    completion: bytes
    score_terms: tuple[ScoreTerm, ...] = ()


def compute_score(
    objects: tuple[WorldObject, ...], terms: tuple[ScoreTerm, ...]
) -> int:
    return sum(
        term.weight
        for obj in objects
        for term in terms
        if obj.color == term.color and obj.shape == term.shape
    )


def _make_completion(
    family: VisualFamily,
    objects: tuple[WorldObject, ...],
    rng: np.random.Generator,
    *,
    min_weight: int,
    max_weight: int,
) -> tuple[bytes, bytes, tuple[ScoreTerm, ...]]:
    if family == "inventory":
        ordered = sorted(objects, key=lambda obj: (obj.row, obj.col))
        target = "".join(
            f"r{obj.row}c{obj.col} {obj.color} {obj.shape}\n" for obj in ordered
        )

        return b"inventory:\n", target.encode("ascii"), ()

    if family == "color_counts":
        target = " ".join(
            f"{color}={sum(obj.color == color for obj in objects)}" for color in COLORS
        )

        return (b"color_counts:\n", (target + "\n").encode("ascii"), ())

    if family == "spatial":
        left = min(objects, key=lambda obj: (obj.col, obj.row))
        right = min(objects, key=lambda obj: (-obj.col, obj.row))

        target = f"left={left.color} {left.shape} r{left.row}c{left.col}; right={right.color} {right.shape} r{right.row}c{right.col}\n"

        return b"extremes:\n", target.encode("ascii"), ()

    if family == "weighted_sum":
        # Select two different object types actually present
        unique_pairs = list(dict.fromkeys((obj.color, obj.shape) for obj in objects))

        if len(unique_pairs) < 2:
            raise ValueError("Weighted sum requires two object types")

        selected = rng.choice(len(unique_pairs), size=2, replace=False)

        terms = tuple(
            ScoreTerm(
                color=unique_pairs[int(i)][0],  # type: ignore
                shape=unique_pairs[int(i)][1],  # type: ignore
                weight=int(rng.integers(min_weight, max_weight + 1)),
            )
            for i in selected
        )

        first, second = terms

        prefix = f"score: {first.color} {first.shape}={first.weight}, {second.color} {second.shape}={second.weight}; value="
        target = f"{compute_score(objects, terms)}\n"

        return (prefix.encode("ascii"), target.encode("ascii"), terms)

    raise ValueError(f"Unknown family: {family}")


def generate_visual_document(
    *,
    seed: int,
    split: Literal["train", "validation", "test"],
    index: int,
    profile: VisualProfile = "mixed",
    family: VisualFamily | None = None,
) -> VisualDocument:
    """Generate one image and its orcle-verified completion."""
    if seed < 0 or index < 0:
        raise ValueError("seed/index must be nonnegative")

    if split not in _SPLIT_IDS:
        raise ValueError("Unknown split")

    if profile != "mixed" and profile not in DIFFICULTIES:
        raise ValueError("Unknown visual profile")

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, _SPLIT_IDS[split], index, 1818])
    )

    if profile == "mixed":
        chosen_profile = str(rng.choice(TRAIN_PROFILES, p=(0.40, 0.35, 0.25)))
    else:
        chosen_profile = profile

    selected_family = FAMILIES[index % len(FAMILIES)] if family is None else family

    min_objects, max_objects, min_weight, max_weight = DIFFICULTIES[chosen_profile]

    scene = sample_world(rng, min_objects=min_objects, max_objects=max_objects)

    prefix, completion, terms = _make_completion(
        selected_family,
        scene.objects,
        rng,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    return VisualDocument(
        sample_id=(f"visual:{split}:{index:012d}:{chosen_profile}:{selected_family}"),
        split=split,
        family=f"visual_{selected_family}",
        image=scene.image,
        objects=scene.objects,
        prefix=prefix,
        completion=completion,
        score_terms=terms,
    )
