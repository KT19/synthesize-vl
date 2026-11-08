import argparse

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from scripts.smoke.experiment_utils import ModelKind
from synthesize_vl.data.schema import Split, TrainingBatch
from synthesize_vl.legacy.generators import generate_documents
from synthesize_vl.legacy.pretraining import bits_per_byte, pack_pretraining_documents
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.training.losses import masked_lm_loss
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import train_step


def make_batch(
    *, seed: int, split: Split, start: int, count: int, max_length: int
) -> TrainingBatch:
    documents = generate_documents(seed=seed, split=split, start=start, count=count)

    return pack_pretraining_documents(documents, max_length=max_length)


def make_model(kind: ModelKind, max_length: int) -> nn.Module:
    if kind == "text":
        return CausalLM(
            hidden_size=32,
            num_heads=4,
            num_layers=1,
            intermediate_size=64,
            max_length=max_length,
            position_mode="rope",
            use_attention_gate=True,
        )

    if kind == "recurrent":
        return RecurrentMultimodalCausalLM(
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            max_text_length=max_length,
            recurrent_steps=1,
            position_mode="rope",
            use_attention_gate=True,
            residual_type="mhc",
            num_streams=2,
            sinkhorn_iterations=10,
            core_attention="hybrid",
            local_window=16,
            global_text_tokens=2,
        )

    raise ValueError(f"Unknown model kind: {kind}")


def model_inputs(batch: TrainingBatch, kind: ModelKind) -> tuple[jax.Array, ...]:
    if kind == "text":
        return (batch.input_ids,)

    return (batch.input_ids, batch.images, batch.has_image)


@jax.jit
def evaluate_batch(
    state: TrainState, batch: TrainingBatch, inputs: tuple[jax.Array, ...]
) -> tuple[jax.Array, jax.Array]:
    """Compute full LM loss and bytes-only BPB."""
    logits = state.apply_fn({"params": state.params}, *inputs)

    return (masked_lm_loss(logits, batch), bits_per_byte(logits, batch))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=("text", "recurrent"), default="recurrent")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if args.validation_size < 1 or args.log_every < 1:
        raise ValueError("validation-size and log-every must be positive")

    kind: ModelKind = args.model
    seed: int = args.seed
    max_length: int = args.max_length

    first_batch = make_batch(
        seed=seed, split="train", start=0, count=args.batch_size, max_length=max_length
    )
    validation = make_batch(
        seed=seed,
        split="validation",
        start=0,
        count=args.validation_size,
        max_length=max_length,
    )

    model = make_model(kind, max_length)

    state = create_train_state(
        model,
        model_inputs(first_batch, kind),
        seed=0,
        config=OptimizerConfig(learning_rate=2e-3),
    )
    val_inputs = model_inputs(validation, kind)

    initial_loss, initial_bpb = evaluate_batch(state, validation, val_inputs)

    print(f"Model: {kind}")
    print(f"Initial validation LM loss: {float(initial_loss):.4f}")
    print(f"Initial validation BPB: {float(initial_bpb):.4f}")

    for update_index in range(args.steps):
        batch = make_batch(
            seed=seed,
            split="train",
            start=update_index * args.batch_size,
            count=args.batch_size,
            max_length=max_length,
        )

        state, train_loss = train_step(state, batch, model_inputs(batch, kind))

        if ((update_index + 1) % args.log_every == 0) or (
            update_index + 1 == args.steps
        ):
            val_loss, val_bpb = evaluate_batch(state, validation, val_inputs)

            print(
                f"Update {int(state.step):4d} | train_nats={float(train_loss):.4f} | val_nats={float(val_loss):.4f} | val_bpb={float(val_bpb):.4f}"
            )

            assert bool(jnp.isfinite(train_loss))
            assert bool(jnp.isfinite(val_loss))
            assert bool(jnp.isfinite(val_bpb))

    print("PASS")


if __name__ == "__main__":
    main()
