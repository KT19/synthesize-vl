import jax.numpy as jnp
from flax.training.train_state import TrainState

from scripts.smoke.experiment_utils import (
    count_parameters,
    first_answer_accuracy,
    make_tiny_multimodal_batch,
    model_inputs,
)
from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.models.vision_language import MultimodalCausalLM
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import eval_step, train_step


def create_experiment_state(
    batch: TrainingBatch, use_gate: bool, seed: int = 0
) -> TrainState:
    model = MultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        intermediate_size=128,
        max_text_length=128,
        position_mode="rope",
        use_attention_gate=use_gate,
    )

    return create_train_state(
        model,
        model_inputs(batch),
        seed=seed,
        config=OptimizerConfig(learning_rate=2e-3),
    )


def run_experiment(
    batch: TrainingBatch, heldout: TrainingBatch, use_gate: bool, steps: int = 200
) -> tuple[float, float, float, int]:
    """Train one condition and return evaluation results."""
    state = create_experiment_state(batch, use_gate=use_gate)

    name = "Gated" if use_gate else "Standard"
    inputs = model_inputs(batch)

    initial_loss, _ = eval_step(state, batch, inputs)

    print(f"\n{name} attention")
    parameters = count_parameters(state.params)
    print(f"Parameters: {parameters:,}")
    print(f"Initial loss: {float(initial_loss):.4f}")

    for step in range(1, steps + 1):
        state, _ = train_step(state, batch, inputs)

        if step % 50 == 0 or step == steps:
            loss, accuracy = eval_step(state, batch, inputs)

            print(
                f"Step {step:3d}: loss={float(loss):.4f}, accuracy={float(accuracy):.1%}"
            )

    final_loss, _ = eval_step(state, batch, inputs)

    heldout_loss, _ = eval_step(state, heldout, model_inputs(heldout))

    visual_accuracy = first_answer_accuracy(state, heldout)

    assert bool(jnp.isfinite(final_loss))
    assert bool(jnp.isfinite(heldout_loss))

    return (float(final_loss), float(heldout_loss), float(visual_accuracy), parameters)


def main() -> None:
    train_batch = make_tiny_multimodal_batch(seed=42)
    heldout_batch = make_tiny_multimodal_batch(seed=123)

    standard = run_experiment(train_batch, heldout_batch, use_gate=False)
    gated = run_experiment(train_batch, heldout_batch, use_gate=True)

    print("\n---Comparison---")
    print("Metric   Standard    Gated")
    print(f"Train loss  {standard[0]:8.4f}  {gated[0]:8.4f}")
    print(f"Heldout loss    {standard[1]:8.4f}  {gated[1]:8.4f}")
    print(f"Heldout visual acc. {standard[2]:8.1%}  {gated[2]:8.1%}")
    print(f"Parameters  {standard[3]:8,d}   {gated[3]:8,d}")
    print("PASS")


if __name__ == "__main__":
    main()
