from dataclasses import dataclass

import jax
from jax import random
from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 64

KINDS = ("circle", "square", "triangle", "digit", "letter")

COLORS = {
    "red": (220, 45, 55),
    "blue": (35, 100, 215),
    "green": (20, 140, 85),
    "purple": (130, 65, 180),
}

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ObjectSpec:
    # What we want to generate
    kind: str
    color: str
    symbol: str = ""


@dataclass(frozen=True)
class VisualObject:
    # After rendering
    spec: ObjectSpec
    bbox: tuple[int, int, int, int]


@dataclass
class Scene:
    image: Image.Image
    objects: tuple[VisualObject, ...]


@dataclass(frozen=True, slots=True)
class QA:
    task: str
    question: str
    answer: str


def _validate(spec: ObjectSpec) -> None:
    if spec.kind not in KINDS or spec.color not in COLORS:
        raise ValueError(f"Invalid object: {spec}")

    if spec.kind == "digit" and spec.symbol not in DIGITS:
        raise ValueError("Expected one ASCII digit")

    if spec.kind == "leter" and spec.symbol not in LETTERS:
        raise ValueError("Expected one uppercase ASCII letter")

    if spec.kind in ("circle", "square", "triangle") and spec.symbol:
        raise ValueError("Shapes must not contain a symbol")


def render_scene(specs: list[ObjectSpec], key: jax.Array) -> Scene:
    """Draw 1-4 objects on a 64x64 canvas."""

    if not 1 <= len(specs) <= 4:
        raise ValueError("Expected between 1 and 4 objects")

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    font = ImageFont.load_default(size=20)

    objects = []

    for index, spec in enumerate(specs):
        _validate(spec)

        # Give each object reproducible position variation.
        position_key = random.fold_in(key, index)

        dx = int(random.randint(random.fold_in(position_key, 0), (), -3, 4))
        dy = int(random.randint(random.fold_in(position_key, 1), (), -3, 4))

        # Four non-overlapping cells
        cx = (index % 2) * 32 + 16 + dx
        cy = (index // 2) * 32 + 16 + dy

        # Draw each object on its own mask
        mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(mask)

        if spec.kind == "circle":
            draw.ellipse((cx - 10, cy - 10, cx + 9, cy + 9), fill=255)

        elif spec.kind == "square":
            draw.rectangle((cx - 10, cy - 10, cx + 9, cy + 9), fill=255)

        elif spec.kind == "triangle":
            draw.polygon(
                [(cx, cy - 11), (cx - 11, cy + 10), (cx + 11, cy + 10)], fill=255
            )

        else:
            # Digits and letters use Pillow's built-in font
            left, top, right, bottom = draw.textbbox((0, 0), spec.symbol, font=font)

            x = cx - (left + right) // 2
            y = cy - (top + bottom) // 2

            draw.text((x, y), spec.symbol, fill=255, font=font)

        # The true extend of the drawn pixels.
        bbox = mask.getbbox()

        if bbox is None:
            raise RuntimeError("Object was not rendered")

        # Apply color to the object mask
        image.paste(COLORS[spec.color], (0, 0), mask)

        objects.append(VisualObject(spec, bbox))

    return Scene(image, tuple(objects))


def make_random_scene(key: jax.Array) -> Scene:
    """Sample object descriptions and render the scene"""

    count = int(random.randint(random.fold_in(key, 0), (), 1, 5))

    specs = []
    color_names = tuple(COLORS)

    for index in range(count):
        obj_key = random.fold_in(key, index + 1)

        kind = KINDS[int(random.randint(random.fold_in(obj_key, 0), (), 0, len(KINDS)))]
        color = color_names[
            int(random.randint(random.fold_in(obj_key, 1), (), 0, len(COLORS)))
        ]

        symbol = ""

        if kind in ("digit", "letter"):
            alphabet = DIGITS if kind == "digit" else LETTERS

            symbol = alphabet[
                int(random.randint(random.fold_in(obj_key, 2), (), 0, len(alphabet)))
            ]

        specs.append(ObjectSpec(kind, color, symbol))

    return render_scene(specs, random.fold_in(key, 999))


def make_question(scene: Scene, task: str, color: str = "red") -> QA:
    """Generate a question and answer from exact metadata."""

    if task == "count_digits":
        count = sum(obj.spec.kind == "digit" for obj in scene.objects)

        return QA(task=task, question="How many digits are shown?", answer=str(count))

    if task == "count_color":
        if color not in COLORS:
            raise ValueError(f"Unknown color: {color}")

        count = sum(obj.spec.color == color for obj in scene.objects)
        return QA(
            task=task,
            question=f"How many {color} objects are shown?",
            answer=str(count),
        )

    if task in ("identify", "color"):
        if len(scene.objects) != 1:
            raise ValueError("Expected a single object")

        spec = scene.objects[0].spec

        if task == "color":
            question = "What color is the object?"
            answer = spec.color

        elif spec.kind == "digit":
            question = "Which digit is shown?"
            answer = spec.symbol

        elif spec.kind == "letter":
            question = "Which letter is shown?"
            answer = spec.symbol

        else:
            question = "Which shape is shown?"
            answer = spec.kind

        return QA(task=task, question=question, answer=answer)

    raise ValueError(f"Unknown task: {task}")
