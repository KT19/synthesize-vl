from scripts.smoke.experiment_utils import make_tiny_multimodal_batch, run_experiment
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.models.vision_language import MultimodalCausalLM


def main() -> None:
    train_batch = make_tiny_multimodal_batch(seed=42)
    heldout_batch = make_tiny_multimodal_batch(seed=123)

    # For independent Transformer blocks
    dense = MultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        num_layers=4,
        intermediate_size=128,
        use_attention_gate=True,
    )

    # Prelude(1) + Core(2) + Coda(1)
    recurrent = RecurrentMultimodalCausalLM(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        recurrent_steps=2,
        use_attention_gate=True,
    )

    results = [
        run_experiment("Dense (4 unique blocks)", dense, train_batch, heldout_batch),
        run_experiment("Recurrent (1 + 2 + 1)", recurrent, train_batch, heldout_batch),
    ]

    print("\n--- Architecture comparison ---")

    for result in results:
        print(f"{result.name}\n")
        print(f"Params: {result.parameters:,}\n")
        print(f"Train loss: {result.train_loss:.4f}\n")
        print(f"Heldout loss: {result.heldout_loss:.4f}\n")
        print(f"Visual acc.: {result.visual_accuracy:.1%}")

    assert all(
        result.train_loss >= 0 and result.heldout_loss >= 0 for result in results
    )
    assert results[1].parameters < results[0].parameters

    print("PASS")


if __name__ == "__main__":
    main()
