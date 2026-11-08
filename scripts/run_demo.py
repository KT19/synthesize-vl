import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml
from PIL import Image

from synthesize_vl.inference.generation import generate_greedy
from synthesize_vl.models.architecture import build_model, load_model_architecture
from synthesize_vl.sft.checkpoints import resolve_pretrained_checkpoint, restore_pretrained_parameters
from synthesize_vl.sft.config import load_sft_config
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.sft.schema import parse_final_answer
from synthesize_vl.tokenizer import ByteTokenizer

DIFFICULTIES = ("easy", "medium", "hard", "ood")


@dataclass(frozen=True, slots=True)
class DemoConfig:
    checkpoint_root: str
    architecture_config: str
    sft_data_yaml: str
    seed: int = 1234
    default_difficulty: str = "medium"
    show_work: bool = True
    max_new_tokens: int = 512
    save_dir: str = "runs/demo"

    open_image: bool = False


def load_demo_config(path: str | Path) -> DemoConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return DemoConfig(**raw)


def decode_generation(token_ids: Any, stopped_on_eos: bool) -> tuple[str, bool]:
    values = [int(token) for token in token_ids]

    if values and values[-1] == ByteTokenizer.EOS:
        values.pop()
    if any(token < 0 or token >= 256 for token in values):
        return "[non-byte special token in generated response]", False

    text = bytes(values).decode("utf-8", errors="replace")
    return text, bool(stopped_on_eos)


class DemoPolicy:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        checkpoint = resolve_pretrained_checkpoint(config.checkpoint_root)
        self.model = build_model(load_model_architecture(config.architecture_config))

        # Initialize every visual component in the same architecture
        example_ids = jnp.asarray(
            [[ByteTokenizer.BOS, ByteTokenizer.IMAGE] + list(b"Question: example")], dtype=jnp.int32
        )
        inputs = (example_ids, jnp.zeros((1, 64, 64, 3), dtype=jnp.float32), jnp.asarray([True]))
        template = self.model.init(jax.random.key(0), *inputs)["params"]
        self.params = restore_pretrained_parameters(checkpoint, template)
        print(f"Loaded weights: {checkpoint}")

    def answer(self, prompt: str, image: Image.Image | None) -> tuple[str, bool]:
        result = generate_greedy(
            self.model, self.params, prompt, image=image, max_new_tokens=self.config.max_new_tokens
        )

        return decode_generation(result.token_ids, result.stopped_on_eos)


def save_and_display_image(image: Image.Image, *, config: DemoConfig, filename: str) -> Path:
    output = Path(config.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    image.save(path)
    print(f"Image: {path.resolve()}")

    if config.open_image:
        try:
            image.show()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open system image viewer: {exc}")

    return path


def custom_text_prompt(expression: str, *, show_work: bool) -> str:
    expression = expression.strip()
    expression = expression.rstrip(".").strip()

    instruction = "Show your steps." if show_work else "Give only the answer."

    return f"Question: Compute {expression}.\n{instruction}\n"


def show_response(policy: DemoPolicy, prompt: str, image: Image.Image | None, *, oracle: int | None) -> None:
    print("\nQuestion:")
    print(prompt, end="" if prompt.endswith("\n") else "\n")
    print("\nModel response:")

    response, finished = policy.answer(prompt=prompt, image=image)
    print(response if response else "[empty response]", end="" if response.endswith("\n") else "\n")
    answer = parse_final_answer(response) if finished else None

    if not finished:
        print("[Did not terminate with EOS]")
    elif answer is None:
        print("[No valid final answer]")

    if oracle is not None:
        verdict = "CORRECT" if answer == oracle else "INCORRECT / INCOMPLETE"
        print(f"Oracle answer: {oracle} | {verdict}")
    elif answer is not None:
        print(f"Parsed answer: {answer} (no oracle for custom input)")


def run_demo(config: DemoConfig, *, once: bool = False) -> None:
    policy = DemoPolicy(config)

    data = replace(load_sft_config(config.sft_data_yaml).data, seed=config.seed)
    difficulty = config.default_difficulty
    show_work = config.show_work
    sample_index = 0

    print("\n Synthesize-VL demo")
    print("Commands: [t]ext, [v]isual, [c]ustom arithmetic, [d]ifficulty, [w]ork toggle, [q]uit")

    while True:
        print(f"\nDifficulty={difficulty} | show_work={show_work}")
        try:
            choice = "v" if once else input("Select [t/v/c/d/w/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if choice == "q":
            return

        if choice == "d":
            try:
                selected = input("Difficulty [easy/medium/hard/ood]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if selected in DIFFICULTIES:
                difficulty = selected
            else:
                print("Invalid difficulty")
            continue

        if choice == "w":
            show_work = not show_work
            continue

        if choice == "c":
            try:
                expr = input("Arithmetic expression: ")
                prompt = custom_text_prompt(expr, show_work=show_work)
                show_response(policy, prompt, None, oracle=None)
            except (ValueError, EOFError, KeyboardInterrupt) as exc:
                print(f"Input cancelled or invalid: {exc}")
            continue

        if choice in ("t", "v"):
            example = generate_sft_example(
                data,
                split="test",
                index=sample_index,
                task_override=("text_arithmetic" if choice == "t" else "visual_arithmetic"),
                difficulty_override=difficulty,  # type: ignore
                cot_override=show_work,
            )
            if example.image is not None:
                save_and_display_image(example.image, config=config, filename=f"visual_{sample_index:05d}.png")
            show_response(policy, example.prompt, example.image, oracle=example.oracle_answer)
            sample_index += 1

        else:
            print("Unknown command")
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize-VL inference demo")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--once", action="store_true")
    options = parser.parse_args()

    run_demo(load_demo_config(options.config), once=options.once)


if __name__ == "__main__":
    main()
