import numpy as np

from synthesize_vl.pretraining.visual import compute_score, generate_visual_document
from synthesize_vl.pretraining.world import COLORS, sample_world


def test_scene_objects_are_unique_and_bounded() -> None:
    rng = np.random.default_rng(5)

    for _ in range(30):
        scene = sample_world(rng, min_objects=3, max_objects=8)

        assert scene.image.size == (64, 64)
        assert 3 <= len(scene.objects) <= 8

        positions = {(obj.row, obj.col) for obj in scene.objects}

        assert len(positions) == len(scene.objects)
        assert len({(obj.color, obj.shape) for obj in scene.objects}) >= 2


def test_visual_document_is_deterministic() -> None:
    first = generate_visual_document(
        seed=42, split="train", index=18, profile="advanced", family="inventory"
    )
    second = generate_visual_document(
        seed=42, split="train", index=18, profile="advanced", family="inventory"
    )

    assert first.sample_id == second.sample_id
    assert first.objects == second.objects
    assert first.prefix == second.prefix
    assert first.completion == second.completion

    np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))


def test_visual_color_count_orcle() -> None:
    for index in range(25):
        document = generate_visual_document(
            seed=7, split="validation", index=index, family="color_counts"
        )

        expected = (
            " ".join(
                f"{color}={sum(obj.color == color for obj in document.objects)}"
                for color in COLORS
            )
            + "\n"
        )

        assert document.completion == expected.encode("ascii")


def test_visual_weighted_arithmetic_oracle() -> None:
    for index in range(30):
        document = generate_visual_document(
            seed=8, split="train", index=index, family="weighted_sum"
        )

        assert len(document.score_terms) == 2

        expected = compute_score(document.objects, document.score_terms)

        assert int(document.completion.strip()) == expected

        for term in document.score_terms:
            assert any(
                obj.color == term.color and obj.shape == term.shape
                for obj in document.objects
            )


def test_visual_spatial_oracle() -> None:
    document = generate_visual_document(seed=9, split="test", index=3, family="spatial")

    left = min(document.objects, key=lambda obj: (obj.col, obj.row))
    right = min(document.objects, key=lambda obj: (-obj.col, obj.row))

    text = document.completion.decode("ascii")

    assert (f"left={left.color} {left.shape} r{left.row}c{left.col}") in text
    assert (f"right={right.color} {right.shape} r{right.row}c{right.col}") in text
