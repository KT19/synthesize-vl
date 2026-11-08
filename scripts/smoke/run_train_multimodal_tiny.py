from flax.training.train_state import TrainState

from scripts.smoke.experiment_utils import (
    first_answer_accuracy,
    make_tiny_multimodal_batch,
    model_inputs,
)
from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.models.vision_language import MultimodalCausalLM
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import eval_step, train_step


def create_multimodal_state(
    batch: TrainingBatch, seed: int = 0, learning_rate: float = 2e-3
) -> TrainState:
    model = MultimodalCausalLM()

    return create_train_state(
        model=model,
        example_inputs=model_inputs(batch),
        seed=seed,
        config=OptimizerConfig(learning_rate=learning_rate),
    )


def main() -> None:
    batch = make_tiny_multimodal_batch()
    state = create_multimodal_state(batch)

    inputs = model_inputs(batch)

    initial_loss, _ = eval_step(state, batch, inputs)

    print(f"Initial loss: {float(initial_loss):.4f}")

    for step in range(1, 301):
        state, _ = train_step(state, batch, inputs)

        if step % 50 == 0:
            loss, accuracy = eval_step(state, batch, inputs)

            visual_accuracy = first_answer_accuracy(state, batch)

            print(
                f"Step {step:3d} | loss={float(loss):.4f} | token_acc={float(accuracy):.1%} | visual_first_byte={float(visual_accuracy):.1%}"
            )

    final_loss, _ = eval_step(state, batch, inputs)

    print(f"Final loss: {float(final_loss):.4f}")

    assert float(final_loss) < float(initial_loss), (
        "The multimodal model did not reduce its loss"
    )

    print("Pass")


if __name__ == "__main__":
    main()
