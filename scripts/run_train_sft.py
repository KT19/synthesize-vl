import argparse
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.models.architecture import build_model
from synthesize_vl.pretraining.checkpoints import (
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.pretraining.evaluation import evaluate_families, print_report
from synthesize_vl.sft.checkpoints import (
    initialize_sft_from_pretraining,
    resolve_pretrained_checkpoint,
)
from synthesize_vl.sft.model_evaluation import evaluate_sft_generation
from synthesize_vl.sft.stream import make_sft_validation_batches, next_sft_batch
from synthesize_vl.sft.train_config import load_sft_train_config
from synthesize_vl.training.accumulation import split_microbatches
from synthesize_vl.training.distributed import prepare_data_parallel
from synthesize_vl.training.joint_objective import make_joint_step
from synthesize_vl.training.optimizers import create_advanced_state


def move_like(source: Any, template: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value, reference: jax.device_put(value, reference.sharding),
        source,
        template,
    )


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, ...]:
    return (batch.input_ids, batch.images, batch.has_image)


def count_parameters(params) -> int:
    return sum(array.size for array in jax.tree_util.tree_leaves(params))


def run_validation(state: TrainState, validation: Mapping[str, TrainingBatch], *, label: str) -> None:
    report = evaluate_families(state=state, batches=validation, inputs_fn=model_inputs)

    print_report(report, label=label)


def main(config_path: str) -> None:
    config = load_sft_train_config(config_path)

    source_checkpoint = resolve_pretrained_checkpoint(config.pretrained_checkpoint_root)
    identity = config.checkpoint_identity(source_checkpoint=str(source_checkpoint))

    print(f"Pretrained path: {source_checkpoint}")
    print(f"SFT output: {config.runtime.checkpoint_root}")

    model = build_model(config.architecture)

    cursor = StreamCursor()
    first_batch, _ = next_sft_batch(config.data, cursor)

    # Parameter shapes do not depend on batch size
    first_inputs = tuple(array[:1] for array in model_inputs(first_batch))
    # Initialize optimizer
    state = create_advanced_state(model, first_inputs, seed=0, config=config.optimizer)

    print(f"Trainable parameters: {count_parameters(state.params):,}")

    if config.runtime.resume_from is None:
        state = initialize_sft_from_pretraining(state, source_checkpoint)

        print("Pretrained parameters loaded.")

    step_fn = make_joint_step(visual_loss_share=0.5)
    first_micro = split_microbatches(first_batch, config.runtime.microbatch_size)

    completed_step, state, sharded_template = prepare_data_parallel(
        step_fn, state, first_micro, num_devices=config.runtime.devices
    )

    if config.runtime.resume_from is not None:
        state, cursor = restore_pretraining_checkpoint(config.runtime.resume_from, state, identity)
        print(f"Resumed SFT at update {cursor.updates}, next example {cursor.next_index}")

    evaluation_data = replace(config.data, seed=config.evaluation.seed)
    validation = make_sft_validation_batches(evaluation_data, examples_per_group=config.evaluation.examples_per_group)
    print("\nInitial SFT evaluation")

    run_validation(state, validation, label=f"SFT validation at step {cursor.updates}")

    if cursor.updates == 0:
        evaluate_sft_generation(model, state.params, evaluation_data, config.evaluation)

    last_saved_step = cursor.updates

    while cursor.updates < config.runtime.max_updates:
        batch, next_cursor = next_sft_batch(config.data, cursor)
        micro = split_microbatches(batch, config.runtime.microbatch_size)

        device_micro = move_like(micro, sharded_template)

        state, metrics = completed_step(state, device_micro)

        if not bool(jnp.isfinite(metrics.loss)):
            raise FloatingPointError("Non-finite SFT training loss")

        cursor = next_cursor

        step = int(jax.device_get(state.step))

        if step != cursor.updates:
            raise RuntimeError("SFT step and data cursor disagree")

        if step % config.runtime.log_every == 0:
            print(
                f"Step {step:>6} | joint={float(metrics.loss):.4f} | text={float(metrics.vm_nats):.4f} | visual={float(metrics.visual_nats):.4f}"
            )

        if step % config.runtime.evaluate_every == 0:
            run_validation(state, validation, label=f"SFT validation at step {step}")

        if step % config.runtime.save_every == 0:
            path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)
            last_saved_step = step
            print("Saved SFT checkpoint:", path)

    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)
        print("Saved final SFT checkpoint:", path)

    print("\nFinal SFT validation")
    run_validation(state, validation, label=f"SFT validation at step {cursor.updates}")

    evaluate_sft_generation(model, state.params, evaluation_data, config.evaluation)

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_student.yaml")
    args = parser.parse_args()
    main(args.config)
