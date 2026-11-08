import numpy as np
import pytest
from jax import random
from PIL import ImageChops

from synthesize_vl.legacy.vision import (
    COLORS,
    IMAGE_SIZE,
    QA,
    ObjectSpec,
    make_question,
    make_random_scene,
    render_scene,
)


def test_scene_dimensions_and_metadata():
    specs = [ObjectSpec("digit", "red", "7"), ObjectSpec("square", "blue")]

    scene = render_scene(specs, random.key(0))

    assert scene.image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(o.spec for o in scene.objects) == tuple(specs)

    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox

        assert 0 <= x0 < x1 <= IMAGE_SIZE
        assert 0 <= y0 < y1 <= IMAGE_SIZE


def test_image_reproducibility():
    specs = [ObjectSpec("letter", "purple", "8")]

    a = render_scene(specs, random.key(2))
    b = render_scene(specs, random.key(2))

    difference = ImageChops.difference(a.image, b.image)

    assert difference.getbbox() is None
    assert a.objects == b.objects


def test_color_and_counting():
    specs = [
        ObjectSpec("digit", "red", "7"),
        ObjectSpec("circle", "red"),
        ObjectSpec("letter", "blue", "A"),
    ]

    scene = render_scene(specs, random.key(4))

    assert make_question(scene, "count_digits").answer == "1"
    assert make_question(scene, "count_color", "red").answer == "2"

    pixels = np.asarray(scene.image)

    assert np.any(np.all(pixels == COLORS["red"], axis=-1))


def test_single_object_questions():
    scene = render_scene([ObjectSpec("triangle", "green")], random.key(5))

    assert make_question(scene, "identify") == QA(
        task="identify", question="Which shape is shown?", answer="triangle"
    )

    assert make_question(scene, "color").answer == "green"


def test_random_generation_reproducibility():
    a = make_random_scene(random.key(10))
    b = make_random_scene(random.key(10))

    assert a.objects == b.objects
    assert ImageChops.difference(a.image, b.image).getbbox() is None


def test_invalid_inputs():
    with pytest.raises(ValueError):
        render_scene([ObjectSpec("digit", "red", "Q")], random.key(0))

    with pytest.raises(ValueError):
        render_scene([], random.key(0))

    scene = render_scene([ObjectSpec("digit", "blue", "2")], random.key(0))

    with pytest.raises(ValueError):
        make_question(scene, "nonsense")
