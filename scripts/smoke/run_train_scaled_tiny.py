"""Example
uv run python -m scripts.smoke.run_train_scaled_tiny --devices 1 --steps 50 --microbatch-size 2
uv run python -m scripts.smoke.run_train_scaled_tiny --devices 2 --steps 50 --microbatch-size 2
uv run python -m scripts.smoke.run_train_scaled_tiny --devices 2 --steps 50 --microbatch-size 2 --mhc
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from scripts.smoke.experiment_utils import make_tiny_multimodal_batch, model_inputs
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM
from synthesize_vl.training.accumulation import (
    make_accumulated_step,
    split_microbatches,
)
from synthesize_vl.training.checkpoints import restore_checkpoint, save_checkpoint
from synthesize_vl.training.distributed import prepare_data_parallel
from synthesize_vl.training.optimizers import (
    AdvancedOptimizerConfig,
    create_advanced_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--mhc", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/scaled_tiny_state")
    )

    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")

    # This is a fixed synthetic smoke-test dataset
    batch = make_tiny_multimodal_batch(seed=42)
    microbatches = split_microbatches(batch, args.microbatch_size)

    model = RecurrentMultimodalCausalLM(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        max_text_length=128,
        recurrent_steps=1,
        position_mode="rope",
        use_attention_gate=True,
        residual_type="mhc" if args.mhc else "standard",
        num_streams=2,
        core_attention="hybrid",
        local_window=16,
        global_text_tokens=2,
    )

    optimizer_config = AdvancedOptimizerConfig(
        muon_lr=3e-3, adamw_lr=1e-3, total_steps=100, warmup_steps=2
    )
    state = create_advanced_state(
        model, model_inputs(batch), seed=0, config=optimizer_config
    )
    step_fn = make_accumulated_step(multimodal=True)

    compiled_step, state, microbatches = prepare_data_parallel(
        step_fn, state, microbatches, num_devices=args.devices
    )

    if args.resume:
        state = restore_checkpoint(args.checkpoint, state)
        print("Resumed from step:", int(state.step))

    print("Devices:", jax.local_devices()[: args.devices])
    print("Microbatch shape:", microbatches.input_ids.shape)

    for _ in range(args.steps):
        state, loss = compiled_step(state, microbatches)

        assert bool(jnp.isfinite(loss))

        print(f"Update {int(state.step):3d} | loss={float(loss):.4f}")

    save_checkpoint(
        args.checkpoint,
        state,
        overwrite=True,
        metadata={
            "dataset_seed": 42,
            "architecture": "recurrent_mhc" if args.mhc else "recurrent_standard",
            "attention": "hybrid",
            "schedule_steps": optimizer_config.total_steps,
            "devices": args.devices,
        },
    )

    print("Checkpoint: ", args.checkpoint.resolve())
    print("PASS")


if __name__ == "__main__":
    main()
