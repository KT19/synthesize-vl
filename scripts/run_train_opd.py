import argparse
from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.models.architecture import build_model
from synthesize_vl.opd.config import load_opd_config
from synthesize_vl.opd.core import make_opd_update, make_student_sampler
from synthesize_vl.opd.data import OPDRollouts, make_opd_batch
from synthesize_vl.pretraining.checkpoints import (
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.pretraining.evaluation import evaluate_families, print_report
from synthesize_vl.sft.checkpoints import (
    initialize_sft_from_pretraining,
    resolve_pretrained_checkpoint,
    restore_pretrained_parameters,
)
from synthesize_vl.sft.model_evaluation import evaluate_sft_generation
from synthesize_vl.sft.stream import make_sft_validation_batches
from synthesize_vl.training.optimizers import create_advanced_state


def replicated_shardings(tree, sharding):
    return jax.tree_util.tree_map(
        lambda _: sharding,
        tree,
    )


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, ...]:
    return (batch.input_ids, batch.images, batch.has_image)


def count_parameters(params) -> int:
    return sum(array.size for array in jax.tree_util.tree_leaves(params))


def run_validation(state: TrainState, validation: Mapping[str, TrainingBatch], *, label: str) -> None:
    report = evaluate_families(state=state, batches=validation, inputs_fn=model_inputs)

    print_report(report, label=label)


def main(config_path: str) -> None:
    config = load_opd_config(config_path)

    available = jax.local_devices()
    if len(available) < config.runtime.devices:
        raise RuntimeError(f"Requested {config.runtime.devices} devices, JAX sees {len(available)}")

    student_source = resolve_pretrained_checkpoint(config.student_checkpoint_root)
    teacher_source = resolve_pretrained_checkpoint(config.teacher_checkpoint_root)

    identity = config.checkpoint_identity(student_source=str(student_source), teacher_source=str(teacher_source))

    print("Student source:", student_source)
    print("Teacher source:", teacher_source)
    print("On-policy distillation")

    student_model = build_model(config.student_architecture)
    teacher_model = build_model(config.teacher_architecture)

    cursor = StreamCursor()

    first_batch = make_opd_batch(
        config.data, cursor, context_length=config.rollout.context_length, rollout_tokens=config.rollout.new_tokens
    )

    # Initialize parameters using one representative row
    init_inputs = (first_batch.input_ids[:1], first_batch.images[:1], first_batch.has_image[:1])
    # Fresh OPD optimizer
    state = create_advanced_state(student_model, init_inputs, seed=0, config=config.optimizer)

    if config.runtime.resume_from is None:
        state = initialize_sft_from_pretraining(state, student_source)
    else:
        state, cursor = restore_pretraining_checkpoint(config.runtime.resume_from, state, identity)

    # Teacher parameters are restored independently
    teacher_template = teacher_model.init(jax.random.key(0), *init_inputs)["params"]
    teacher_params = restore_pretrained_parameters(teacher_source, teacher_template)

    print(f"Student parameters: {count_parameters(state.params):,}")
    print(f"Teacher parameters: {count_parameters(teacher_params):,}")

    print(f"OPD starting optimizer step: {int(jax.device_get(state.step))}")

    # Dara parallelism
    mesh = Mesh(np.asarray(available[: config.runtime.devices]), ("data",))
    replica = NamedSharding(mesh, P())
    data_shard = NamedSharding(mesh, P("data"))

    state_shardings = replicated_shardings(state, replica)
    teacher_shardings = replicated_shardings(teacher_params, replica)
    prompt_shardings = replicated_shardings(first_batch, data_shard)

    rollout_template = OPDRollouts(
        input_ids=first_batch.input_ids,
        prefix_lengths=first_batch.prefix_lengths,
        images=first_batch.images,
        has_image=first_batch.has_image,
        sampled_tokens=jnp.zeros((config.data.batch_size, config.rollout.new_tokens), dtype=jnp.int32),
        valid_mask=jnp.zeros((config.data.batch_size, config.rollout.new_tokens), dtype=jnp.bool_),
    )

    rollout_shardings = replicated_shardings(rollout_template, data_shard)

    state = jax.device_put(state, state_shardings)
    teacher_params = jax.device_put(teacher_params, teacher_shardings)

    sampler = jax.jit(
        make_student_sampler(
            student_model, new_tokens=config.rollout.new_tokens, temperature=config.rollout.sampling_temperature
        ),
        in_shardings=(replicated_shardings(state.params, replica), prompt_shardings, replica),
        out_shardings=rollout_shardings,
    )
    train_step = jax.jit(
        make_opd_update(
            student_model,
            teacher_model,
            distillation_temperature=config.rollout.distillation_temperature,
            visual_loss_share=config.rollout.visual_loss_share,
        ),
        in_shardings=(state_shardings, teacher_shardings, rollout_shardings),
        out_shardings=(state_shardings, None),
    )

    evaluation_data = replace(config.data, seed=config.evaluation.seed)

    validation = make_sft_validation_batches(evaluation_data, examples_per_group=config.evaluation.examples_per_group)
    run_validation(state, validation, label=f"Initial OPD validation at step {cursor.updates}")

    if cursor.updates == 0:
        evaluate_sft_generation(student_model, state.params, evaluation_data, config.evaluation)

    last_saved_step = cursor.updates

    while cursor.updates < config.runtime.max_updates:
        batch = make_opd_batch(
            config.data, cursor, context_length=config.rollout.context_length, rollout_tokens=config.rollout.new_tokens
        )
        placed = jax.device_put(batch, prompt_shardings)

        # Reproducible sampling key
        sampling_key = jax.random.fold_in(jax.random.key(config.rollout.seed), cursor.updates)
        sampling_key = jax.device_put(sampling_key, replica)

        # 1. sample from the current student policy
        rollouts = sampler(state.params, placed, sampling_key)

        # 2. Evaluate frozen teacher and train student
        updated, metrics = train_step(state, teacher_params, rollouts)

        if not bool(jnp.isfinite(metrics.loss)):
            raise FloatingPointError("Non-finite SFT training loss")

        state = updated

        # Count only prompt bytes and valid studnet rollout tokens
        prompt_bytes = int(jax.device_get(jnp.sum(placed.prefix_lengths - 1 - placed.has_image.astype(jnp.int32))))

        sampled_targets = int(jax.device_get(jnp.sum(rollouts.valid_mask)))

        cursor = StreamCursor(
            next_index=cursor.next_index + config.data.batch_size,
            updates=cursor.updates + 1,
            documents_seen=cursor.documents_seen + config.data.batch_size,
            bytes_seen=cursor.bytes_seen + prompt_bytes,
            targets_seen=cursor.targets_seen + sampled_targets,
        )

        step = int(jax.device_get(state.step))

        if step != cursor.updates:
            raise RuntimeError("OPD optimizer/data cursor mismatch")

        if step % config.runtime.log_every == 0:
            print(
                f"Step {step:>6} | KL={float(metrics.loss):.5f} | text={float(metrics.text_kl):.5f} | visual={float(metrics.visual_kl):.5f}"
            )

        if step % config.runtime.evaluate_every == 0:
            run_validation(state, validation, label=f"OPD validation at step {step}")

        if step % config.runtime.save_every == 0:
            path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)
            last_saved_step = step
            print("Saved OPD checkpoint:", path)

    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(config.runtime.checkpoint_root, state, cursor, identity)
        print("Saved final OPD checkpoint:", path)

    print("\nFinal OPD validation")
    run_validation(state, validation, label=f"Final OPD at step {cursor.updates}")

    evaluate_sft_generation(student_model, state.params, evaluation_data, config.evaluation)

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/opd.yaml")
    args = parser.parse_args()
    main(args.config)
