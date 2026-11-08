from scripts.smoke.experiment_utils import make_tiny_multimodal_batch, run_experiment
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


def main() -> None:
    train_batch = make_tiny_multimodal_batch(seed=42)
    heldout_batch = make_tiny_multimodal_batch(seed=123)

    # Prelude(1) + Core(2) + Coda(1)
    baseline = RecurrentMultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        recurrent_steps=2,
        position_mode="rope",
        use_attention_gate=True,
        residual_type="standard",
    )
    mhc = RecurrentMultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        recurrent_steps=2,
        position_mode="rope",
        use_attention_gate=True,
        residual_type="mhc",
        num_streams=2,
        sinkhorn_iterations=20,
    )

    results = [
        run_experiment(
            "Standard recurrent Core (1 + 2 + 1)", baseline, train_batch, heldout_batch
        ),
        run_experiment(
            "Two-stream mHC core (1 + 2 + 1)", mhc, train_batch, heldout_batch
        ),
    ]

    print("\n--- Architecture comparison ---")

    for result in results:
        print(f"{result.name}\n")
        print(f"Params: {result.parameters:,}\n")
        print(f"Train loss: {result.train_loss:.4f}\n")
        print(f"Heldout loss: {result.heldout_loss:.4f}\n")
        print(f"Visual acc.: {result.visual_accuracy:.1%}")

    print("PASS")


if __name__ == "__main__":
    main()
