from scripts.smoke.experiment_utils import (
    ExperimentResult,
    make_tiny_multimodal_batch,
    run_experiment,
)
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


def main() -> None:
    train_batch = make_tiny_multimodal_batch(seed=42)
    heldout_batch = make_tiny_multimodal_batch(seed=123)

    full_model = RecurrentMultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        max_text_length=128,
        recurrent_steps=2,
        position_mode="rope",
        use_attention_gate=True,
        residual_type="mhc",
        num_streams=2,
        sinkhorn_iterations=20,
        core_attention="full",
    )

    hybrid_model = RecurrentMultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        max_text_length=128,
        recurrent_steps=2,
        position_mode="rope",
        use_attention_gate=True,
        residual_type="mhc",
        num_streams=2,
        sinkhorn_iterations=20,
        core_attention="hybrid",
        local_window=16,
        global_text_tokens=2,
    )

    results: list[ExperimentResult] = [
        run_experiment(
            "Full recurrent attention",
            model=full_model,
            train_batch=train_batch,
            heldout_batch=heldout_batch,
            steps=75,
        ),
        run_experiment(
            "Hybrid recurrent attention",
            model=hybrid_model,
            train_batch=train_batch,
            heldout_batch=heldout_batch,
            steps=75,
        ),
    ]

    print("\n--- Answer comparison ---")

    for result in results:
        print(
            f"\n{result.name}\n"
            f"Parameters: {result.parameters:,}\n"
            f"Training loss: {result.train_loss:.4f}\n"
            f"Held-out loss: {result.heldout_loss:.4f}\n"
            f"Held-out visual: {result.visual_accuracy:.1%}"
        )

    # Attention masks must not add parameters
    assert results[0].parameters == results[1].parameters

    print("PASS")


if __name__ == "__main__":
    main()
