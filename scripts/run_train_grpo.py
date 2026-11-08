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
from synthesize_vl.grpo.config import load_grpo_config
from synthesize_vl.grpo.core import make_grpo_update, rollout_log_probs
from synthesize_vl.grpo.data import make_grop_prompts, score_rollout_groups
from synthesize_vl.models.architecture import build_model
from synthesize_vl.opd.core import make_student_sampler
from synthesize_vl.opd.data import OPDRollouts
from synthesize_vl.pretraining.checkpoints import restore_pretraining_checkpoint, save_pretraining_checkpoint
from synthesize_vl.pretraining.evaluation import evaluate_families, print_report
from synthesize_vl.sft.checkpoints import initialize_sft_from_pretraining, resolve_pretrained_checkpoint
from synthesize_vl.sft.model_evaluation import evaluate_sft_generation
from synthesize_vl.sft.stream import make_sft_validation_batches
from synthesize_vl.training.optimizers import create_advanced_state


def replicated_shardings(tree, sharding):
    return jax.tree_util.tree_map(lambda _: sharding, tree)


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, ...]:
    return (batch.input_ids, batch.images, batch.has_image)


def count_parameters(params) -> int:
    return sum(array.size for array in jax.tree_util.tree_leaves(params))


def run_validation(state: TrainState, validation: Mapping[str, TrainingBatch], *, label: str) -> None:
    report = evaluate_families(state=state, batches=validation, inputs_fn=model_inputs)

    print_report(report, label=label)


def replace_stream_updates(cursor: StreamCursor, new_updates: int) -> StreamCursor:
    """Preserve stream counts while advancing optimizer step."""
    return StreamCursor(
        next_index=cursor.next_index,
        updates=new_updates,
        documents_seen=cursor.documents_seen,
        bytes_seen=cursor.bytes_seen,
        targets_seen=cursor.targets_seen,
    )


def main(config_path: str) -> None:
    config = load_grpo_config(config_path)
    available = jax.local_devices()

    if len(available) < config.runtime.devices:
        raise RuntimeError(f"Need {config.runtime.devices} devices; JAX sees {len(available)}")

    source = resolve_pretrained_checkpoint(config.policy_checkpoint_root)
    identity = config.checkpoint_identity(str(source))

    print("GRPO")
    print("Initial policy:", source)

    model = build_model(config.architecture)
    cursor = StreamCursor()

    _, first_prompts = make_grop_prompts(
        config.data,
        start=0,
        group_size=config.grpo.group_size,
        context_length=config.rollout.context_length,
        new_tokens=config.rollout.new_tokens,
    )

    init_inputs = (first_prompts.input_ids[:1], first_prompts.images[:1], first_prompts.has_image[:1])

    # Fresh optimizer
    state = create_advanced_state(model, init_inputs, seed=0, config=config.optimizer)

    if config.runtime.resume_from is None:
        state = initialize_sft_from_pretraining(state, source)
    else:
        state, cursor = restore_pretraining_checkpoint(config.runtime.resume_from, state, identity)

    print(f"Model parameters: {count_parameters(state.params):,}")

    epochs = config.rollout.ppo_epochs
    print("Starting GRPO step:", cursor.updates // epochs)

    if cursor.updates % epochs:
        raise ValueError("Checkpoint is not at a completed rollout boundary")

    if cursor.next_index % config.data.batch_size:
        raise ValueError("Restored data cursor is not at a prompt-batch boundary")

    # Parameters, states: replicated. Response -> sharded across GPUs
    mesh = Mesh(np.asarray(available[: config.runtime.devices]), ("data",))

    replica = NamedSharding(mesh, P())
    data_shard = NamedSharding(mesh, P("data"))

    state_shardings = replicated_shardings(state, replica)
    param_shardings = replicated_shardings(state.params, replica)
    prompt_shardings = replicated_shardings(first_prompts, data_shard)

    total_responses = config.data.batch_size * config.grpo.group_size

    rollout_template = OPDRollouts(
        input_ids=first_prompts.input_ids,
        prefix_lengths=first_prompts.prefix_lengths,
        images=first_prompts.images,
        has_image=first_prompts.has_image,
        sampled_tokens=jnp.zeros((total_responses, config.rollout.new_tokens), dtype=jnp.int32),
        valid_mask=jnp.zeros((total_responses, config.rollout.new_tokens), dtype=jnp.bool_),
    )

    rollout_shardings = replicated_shardings(rollout_template, data_shard)

    state = jax.device_put(state, state_shardings)
    sampler = jax.jit(
        make_student_sampler(model, new_tokens=config.rollout.new_tokens, temperature=1.0),
        in_shardings=(param_shardings, prompt_shardings, replica),
        out_shardings=rollout_shardings,
    )
    old_logprob_fn = jax.jit(
        lambda params, rollouts: rollout_log_probs(model, params, rollouts),
        in_shardings=(param_shardings, rollout_shardings),
        out_shardings=data_shard,
    )

    update_fn = jax.jit(
        make_grpo_update(model, group_size=config.grpo.group_size, clip_epsilon=config.rollout.clip_epsilon),
        in_shardings=(state_shardings, rollout_shardings, data_shard, data_shard),
        out_shardings=(state_shardings, None),
    )

    evaluation_data = replace(config.data, seed=config.evaluation.seed)
    validation = make_sft_validation_batches(evaluation_data, examples_per_group=config.evaluation.examples_per_group)
    run_validation(state, validation, label=f"Initial GRPO validation at step {cursor.updates // epochs}")

    if cursor.updates == 0:
        evaluate_sft_generation(model, state.params, evaluation_data, config.evaluation)

    last_saved_step = cursor.updates
    attempted_rounds = 0
    uninformative_streak = 0

    while cursor.updates // epochs < config.runtime.max_updates:
        attempted_rounds += 1

        examples, prompts = make_grop_prompts(
            config.data,
            start=cursor.next_index,
            group_size=config.grpo.group_size,
            context_length=config.rollout.context_length,
            new_tokens=config.rollout.new_tokens,
        )

        placed = jax.device_put(prompts, prompt_shardings)

        round_index = cursor.next_index // config.data.batch_size

        key = jax.random.fold_in(jax.random.key(config.rollout.seed), round_index)
        key = jax.device_put(key, replica)

        # 1. Sample from the current student policy
        rollouts = sampler(state.params, placed, key)

        # 2. Score sampled responses with the exact oracle
        report = score_rollout_groups(examples, rollouts, group_size=config.grpo.group_size)
        prompt_bytes = sum(len(example.prompt.encode("utf-8")) for example in examples)

        sampled_target = int(jax.device_get(jnp.sum(rollouts.valid_mask)))

        advanced = StreamCursor(
            next_index=cursor.next_index + len(examples),
            updates=cursor.updates,
            documents_seen=cursor.documents_seen + len(examples),
            bytes_seen=cursor.bytes_seen + prompt_bytes,
            targets_seen=cursor.targets_seen + sampled_target,
        )

        if not report.has_signal:
            cursor = advanced
            uninformative_streak += 1

            if attempted_rounds % 10 == 0:
                print(
                    f"Rollout round {attempted_rounds}: no group-relative signal; exact reward rate={report.correct_rate:.1%}"
                )

            if uninformative_streak >= config.runtime.max_uninformative_streak:
                break

            continue

        uninformative_streak = 0

        # 3. Save the behavior policy's token log probs
        # These remain fixed during both PPO passes
        old_log_probs = old_logprob_fn(state.params, rollouts)

        advantages = jnp.asarray(report.advantages.reshape(-1), dtype=jnp.float32)
        advantages = jax.device_put(advantages, data_shard)

        # 4. Two clipped updates
        for _ in range(epochs):
            state, metrics = update_fn(state, rollouts, old_log_probs, advantages)

            if not bool(jnp.isfinite(metrics.loss)):
                raise FloatingPointError("Non-finite GRPO loss")

        cursor = replace_stream_updates(advanced, advanced.updates + epochs)
        step = cursor.updates // epochs

        if step % config.runtime.log_every == 0:
            print(
                f"Step {step:>5} | GRPO loss={float(metrics.loss):.5f} | reward={report.correct_rate:.1%} | clip={float(metrics.clip_fraction):.1%}"
            )

        if step % config.runtime.evaluate_every == 0:
            run_validation(state, validation, label=f"GRPO validation at step {step}")

        if step % config.runtime.save_every == 0:
            path = save_pretraining_checkpoint(
                config.runtime.checkpoint_root, state, cursor, identity, checkpoint_step=step
            )

            last_saved_step = cursor.updates
            print("Saved GRPO checkpoint:", path)

    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(
            config.runtime.checkpoint_root, state, cursor, identity, checkpoint_step=cursor.updates // epochs
        )

        print("Saved final GRPO checkpoint:", path)

    run_validation(state, validation, label=f"Final GRPO validation at step {cursor.updates // epochs}")

    evaluate_sft_generation(model, state.params, evaluation_data, config.evaluation)

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    args = parser.parse_args()
    main(args.config)
