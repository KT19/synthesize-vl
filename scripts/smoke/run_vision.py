import json
from dataclasses import asdict
from pathlib import Path

from jax import random

from synthesize_vl.legacy.vision import (
    ObjectSpec,
    make_question,
    make_random_scene,
    render_scene,
)


def main() -> None:
    output_dir = Path("outputs/vision")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ObjectSpec("digit", "red", "7"),
        ObjectSpec("triangle", "blue"),
        ObjectSpec("letter", "purple", "A"),
        ObjectSpec("circle", "green"),
    ]

    scene = render_scene(specs, random.key(42))
    scene.image.save(output_dir / "demo.png")

    examples = {}

    for task in ("count_digits", "count_color"):
        qa = make_question(scene, task)

        examples[task] = asdict(qa)

    metadata = {
        "objects": [
            {
                "kind": obj.spec.kind,
                "color": obj.spec.color,
                "symbol": obj.spec.symbol,
                "bbox": obj.bbox,
            }
            for obj in scene.objects
        ],
        "examples": examples,
    }

    (output_dir / "demo.json").write_text(json.dumps(metadata, indent=2))

    print("Saved:", output_dir / "demo.png")
    print("Saved:", output_dir / "demo.json")

    for task, qa in examples.items():
        print(task, "|", qa["question"], "->", qa["answer"])

    # Generate four additional randomized images
    root_key = random.key(123)

    for index in range(4):
        key = random.fold_in(root_key, index)
        random_scene = make_random_scene(key)

        random_scene.image.save(output_dir / f"random_{index}.png")

    print("Saved 4 additional random scenes")


if __name__ == "__main__":
    main()
