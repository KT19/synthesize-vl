import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.models.architecture import build_model
from synthesize_vl.pretraining.checkpoints import (
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.pretraining.config import PretrainConfig, load_pretrain_config
from synthesize_vl.pretraining.evaluation import (
    evaluate_families,
    print_report,
)
from synthesize_vl.pretraining.history import TrainingHistory
from synthesize_vl.pretraining.joint_evaluation import (
    make_joint_evaluation_documents,
    pack_evaluation_groups,
)
from synthesize_vl.pretraining.task_stream import make_task_validation, next_task_batch
from synthesize_vl.training.accumulation import split_microbatches
from synthesize_vl.training.distributed import prepare_data_parallel
from synthesize_vl.training.joint_objective import make_joint_step
from synthesize_vl.training.optimizers import create_advanced_state

DEFAULT_CONFIG = Path("configs/pretrain.yaml")


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, ...]:
    return (batch.input_ids, batch.images, batch.has_image)


def move_like(source: Any, template: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value, reference: jax.device_put(value, reference.sharding),
        source,
        template,
    )


def count_parameters(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


def evaluate_validation(state, validation, history: TrainingHistory) -> None:
    report = evaluate_families(state, validation, model_inputs)

    print_report(report, label=f"Validation at update {int(state.step)}")
    history.record_validation(update=int(state.step), report=report)


def write_resolved_configuration(config: PretrainConfig) -> None:
    root = Path(config.runtime.checkpoint_root).resolve()

    root.mkdir(parents=True, exist_ok=True)

    destination = root / "resolved_pretrain.yaml"

    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(config), stream, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config

    config = load_pretrain_config(config_path)

    if config.runtime.max_updates > config.optimizer.total_steps:
        raise ValueError("max_updates exceeds optimizer schedule")

    if jax.local_device_count() < config.runtime.devices:
        raise RuntimeError(f"Requested {config.runtime.devices} devices; JAX sees {jax.local_device_count()}")

    print("Synthesize-VL pretraining with fixed task sampling")
    print("Config:", Path(config_path).resolve())
    print("Devices:", jax.local_devices()[: config.runtime.devices])

    write_resolved_configuration(config)

    identity = config.checkpoint_identity()
    model = build_model(config.architecture)

    cursor = StreamCursor()
    first_batch, _ = next_task_batch(data=config.data, max_length=config.architecture.max_text_length, cursor=cursor)

    init_inputs = tuple(x[:1] for x in model_inputs(first_batch))
    state = create_advanced_state(model, init_inputs, seed=0, config=config.optimizer)

    print(f"Trainable parameters: {count_parameters(state.params):,}")

    first_micro = split_microbatches(first_batch, config.data.microbatch_size)
    step_fn = make_joint_step(visual_loss_share=config.data.visual_loss_share)

    compiled_step, state, sharded_template = prepare_data_parallel(
        step_fn, state, first_micro, num_devices=config.runtime.devices
    )

    # Restore after constructing the exact same structure
    if config.runtime.resume_from is not None:
        state, cursor = restore_pretraining_checkpoint(config.runtime.resume_from, state, identity)

        print(f"Restored update {cursor.updates} next sample {cursor.next_index}")

    history = TrainingHistory(
        config.runtime.checkpoint_root,
        resume_update=cursor.updates if config.runtime.resume_from is not None else None,
    )

    validation = make_task_validation(data=config.data, max_length=config.architecture.max_text_length)

    if config.runtime.resume_from is None:
        evaluate_validation(state, validation, history)

    last_saved_step = cursor.updates

    while cursor.updates < config.runtime.max_updates:
        batch, advanced = next_task_batch(
            data=config.data,
            max_length=config.architecture.max_text_length,
            cursor=cursor,
        )

        microbatches = split_microbatches(batch, config.data.microbatch_size)
        placed = move_like(microbatches, sharded_template)

        state, metrics = compiled_step(state, placed)

        # Wait for the update before committing its cursor.
        if not bool(jnp.isfinite(metrics.loss)):
            raise FloatingPointError(f"Non-finite loss at update {cursor.updates + 1}")

        cursor = advanced

        if int(state.step) != cursor.updates:
            raise RuntimeError("Optimizer step and stream cursor disagree")

        step = cursor.updates

        history.record_training(
            update=step,
            joint_nats=float(metrics.loss),
            vm_nats=float(metrics.vm_nats),
            visual_nats=float(metrics.visual_nats),
            vm_targets=int(metrics.vm_targets),
            visual_targets=int(metrics.visual_targets),
            targets_seen=cursor.targets_seen,
            bytes_seen=cursor.bytes_seen,
        )

        if step % config.runtime.log_every == 0:
            print(
                f"Step {step:>6} | joint={float(metrics.loss):.4f} | vm={float(metrics.vm_nats):.4f} | vision={float(metrics.visual_nats):.4f} | targets={cursor.targets_seen:,} | bytes={cursor.bytes_seen:,}"
            )

        if step % config.runtime.evaluate_every == 0:
            evaluate_validation(state, validation, history)

        if step % config.runtime.save_every == 0:
            path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)
            last_saved_step = cursor.updates
            print(f"Saved: {path}")

    # save the last update
    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)

        print(f"Saved final: {path}")

    if config.runtime.evaluate_ood_at_end:
        ood_documents = make_joint_evaluation_documents(
            seed=config.data.seed,
            split="test",
            examples_per_family=config.data.validation_per_arm,
            max_length=config.architecture.max_text_length,
            out_of_distribution=True,
        )
        ood_batches = pack_evaluation_groups(ood_documents, max_length=config.architecture.max_text_length)
        report = evaluate_families(state, ood_batches, model_inputs)
        print_report(report, label="Final held-out difficulty evaluation")

    print("Complete")


if __name__ == "__main__":
    main()
