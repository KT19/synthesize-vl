from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.legacy.synthetic import make_dataset
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import eval_step, train_step


def make_tiny_batch() -> TrainingBatch:
    examples = make_dataset(seed=42, count=4)

    samples = [
        TrainingSample(
            task=example.kind, prompt=example.prompt + " ", answer=example.answer
        )
        for example in examples
    ]

    return pack_examples(samples, max_text_length=128)


def create_state(
    model: CausalLM, batch: TrainingBatch, seed: int = 0, learning_rate: float = 2e-3
):
    return create_train_state(
        model,
        (batch.input_ids,),
        seed=seed,
        config=OptimizerConfig(learning_rate=learning_rate),
    )


def main():
    batch = make_tiny_batch()

    model = CausalLM()
    state = create_state(model, batch)

    initial_loss, _ = eval_step(state, batch)

    print(f"Initial loss: {initial_loss:.4f}")

    # Train repeatedly on the same four examples
    for step in range(1, 201):
        state, _ = train_step(state, batch)

        if step % 50 == 0:
            loss, accuracy = eval_step(state, batch)

            print(f"Step {step:3d}: loss={loss:.4f}, accuracy={float(accuracy):.1%}")

    final_loss, final_accuracy = eval_step(state, batch)

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")

    print(f"Answer-token accuracy (teacher forced): {float(final_accuracy):.1%}")

    assert final_loss < initial_loss, "Training did not reduce loss"

    print("Pass")


if __name__ == "__main__":
    main()
