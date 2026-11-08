from pathlib import Path

from synthesize_vl.sft.config import load_sft_config
from synthesize_vl.sft.dataset import generate_sft_example, make_sft_batch
from synthesize_vl.sft.schema import exact_answer_reward, validate_sft_example


def main() -> None:
    config_path = Path("configs/sft_data.yaml")
    config = load_sft_config(config_path)
    output_dir = Path("runs/sft_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SFT configuration:", config_path)

    for task in ("text_arithmetic", "visual_arithmetic"):
        for difficulty_index, difficulty in enumerate(("easy", "medium", "hard")):
            example = generate_sft_example(
                config.data,
                split="validation",
                index=100 + difficulty_index,
                task_override=task,  # type: ignore
                difficulty_override=difficulty,
                cot_override=True,
            )
            validate_sft_example(example)

            assert exact_answer_reward(example.response, example.oracle_answer) == 1.0

            print("\n" + "=" * 50)
            print("ID:", example.sample_id)
            print(example.prompt)
            print(example.response)

            if example.image is not None:
                path = output_dir / f"visual_{difficulty}.png"

                example.image.save(path)
                print("Image:", path)

    batch = make_sft_batch(config.data, split="train", start=0)
    print("\nTraining batch")
    print("Input IDs:", batch.input_ids.shape)
    print("Target IDs:", batch.target_ids.shape)
    print("Images:", batch.images.shape)
    print("Visual rows:", int(batch.has_image.sum()))
    print("Supervised targets:", int(batch.loss_mask.sum()))
    print("PASS")


if __name__ == "__main__":
    main()
