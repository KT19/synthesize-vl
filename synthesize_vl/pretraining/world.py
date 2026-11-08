from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

Color = Literal["red", "blue", "green", "yellow"]
Shape = Literal["circle", "square", "triangle"]

COLORS: tuple[Color, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle")

RGB = {
    "red": (220, 50, 65),
    "blue": (45, 100, 220),
    "green": (35, 160, 95),
    "yellow": (225, 175, 35),
}


@dataclass(frozen=True, slots=True)
class WorldObject:
    color: Color
    shape: Shape
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class WorldScene:
    objects: tuple[WorldObject, ...]
    image: Image.Image


def render_world(
    objects: tuple[WorldObject, ...], rng: np.random.Generator, *, size: int = 64
) -> Image.Image:
    """Draw objects on a 3x3 grid without overlap"""
    if size != 64:
        raise ValueError("Lesson 18 uses 64x64 images")

    cells = [(obj.row, obj.col) for obj in objects]

    if len(set(cells)) != len(cells):
        raise ValueError("Objects occupy duplicate cells")

    for obj in objects:
        if not (0 <= obj.row < 3 and 0 <= obj.col < 3):
            raise ValueError("Grid coordinatse out of bounds")

    # Render at higher resolution, then downsample
    scale = 3
    background = int(rng.integers(243, 253))

    image = Image.new("RGB", (size * scale, size * scale), (background,) * 3)

    draw = ImageDraw.Draw(image)

    for obj in objects:
        cx = (11 + 21 * obj.col) * scale
        cy = (11 + 21 * obj.row) * scale

        cx += int(rng.integers(-1, 2)) * scale
        cy += int(rng.integers(-1, 2)) * scale

        radius = 7 * scale
        color = RGB[obj.color]

        if obj.shape == "circle":
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
            )
        elif obj.shape == "square":
            draw.rectangle(
                (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
            )
        elif obj.shape == "triangle":
            draw.polygon(
                (
                    (cx, cy - radius),
                    (cx - radius, cy + radius),
                    (cx + radius, cy + radius),
                ),
                fill=color,
            )

    return image.resize((size, size), Image.Resampling.LANCZOS)


def sample_world(
    rng: np.random.Generator, *, min_objects: int, max_objects: int
) -> WorldScene:
    """Sample a scene with at least two distinct object types"""
    if not (2 <= min_objects <= max_objects <= 9):
        raise ValueError("Invalid number of objects")

    count = int(rng.integers(min_objects, max_objects + 1))
    positions = rng.choice(9, size=count, replace=False)

    objects: list[WorldObject] = []

    for cell in positions:
        row, col = divmod(int(cell), 3)

        color = COLORS[int(rng.integers(0, len(COLORS)))]
        shape = SHAPES[int(rng.integers(0, len(SHAPES)))]

        objects.append(WorldObject(color, shape, row, col))

    # Guarantee at least two distinct color-shape pairs
    if len({(obj.color, obj.shape) for obj in objects}) == 1:
        first = objects[0]
        objects[0] = WorldObject(
            color=COLORS[(COLORS.index(first.color) + 1) % len(COLORS)],
            shape=first.shape,
            row=first.row,
            col=first.col,
        )

    records = tuple(objects)

    return WorldScene(objects=records, image=render_world(records, rng))
