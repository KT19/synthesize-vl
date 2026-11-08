import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.legacy.joint_stream import next_joint_batch
from synthesize_vl.legacy.multimodal_stream import MixedStreamConfig
from synthesize_vl.models.architecture import ModelArchitecture, build_model
from synthesize_vl.pretraining.checkpoints import (
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.pretraining.evaluation import (
    evaluate_families,
    print_report,
)
from synthesize_vl.pretraining.joint_evaluation import (
    evaluate_exact_tasks,
    make_joint_evaluation_documents,
    pack_evaluation_groups,
    print_exact_report,
)
from synthesize_vl.training.accumulation import (
    split_microbatches,
)
from synthesize_vl.training.distributed import prepare_data_parallel
from synthesize_vl.training.joint_objective import make_joint_step
from synthesize_vl.training.optimizers import (
    AdvancedOptimizerConfig,
    create_advanced_state,
)


def inputs(batch: TrainingBatch) -> tuple[jax.Array, ...]:
    return (batch.input_ids, batch.images, batch.has_image)


def count_parameters(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


def move_like(source: Any, template: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value, reference: jax.device_put(value, reference.sharding),
        source,
        template,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--vision-slots", type=int, default=4)
    parser.add_argument("--visual-loss-share", type=float, default=0.5)

    parser.add_argument("--schedule-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=10)

    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validation-per-family", type=int, default=4)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--exact-probes", type=int, default=1)

    args = parser.parse_args()

    if (
        min(
            args.steps,
            args.microbatch_size,
            args.devices,
            args.eval_every,
            args.save_every,
            args.validation_per_family,
        )
        < 1
    ):
        raise ValueError("Expected positive run settings")

    if args.exact_probes < 0:
        raise ValueError("--exact-probes must be nonnegative")

    if args.batch_size % args.microbatch_size:
        raise ValueError("batch-size must be divisible by microbatch-size")

    if args.microbatch_size % args.devices:
        raise ValueError("microbatch-size must be divisible by devices")

    model_architecture = ModelArchitecture()

    config = MixedStreamConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=model_architecture.max_text_length,
        vision_slots=args.vision_slots,
        vm_profile="mixed",
        visual_profile="mixed",
    )

    optimizer_config = AdvancedOptimizerConfig(
        muon_lr=3e-3,
        adamw_lr=1e-3,
        total_steps=args.schedule_steps,
        warmup_steps=args.warmup_steps,
    )

    fingerprint = {
        "format_version": 1,
        "dataset": "joint_vm_visual",
        "architecture": model_architecture.to_dict(),
        "stream": asdict(config),
        "optimizer": asdict(optimizer_config),
        "visual_loss_share": args.visual_loss_share,
        "microbatch_size": args.microbatch_size,
    }

    model = build_model(model_architecture)

    # Initialize a model and a batch with final shapes
    cursor = StreamCursor()

    first_batch, _ = next_joint_batch(config, cursor)

    state = create_advanced_state(
        model, inputs(first_batch), seed=0, config=optimizer_config
    )

    print(f"Student parameters: {count_parameters(state.params):,}")

    # One optimizer update after several microbatches
    micro = split_microbatches(first_batch, args.microbatch_size)

    compiled_step, state, sharded_template = prepare_data_parallel(
        make_joint_step(visual_loss_share=args.visual_loss_share),
        state,
        micro,
        num_devices=args.devices,
    )

    if args.resume_from is not None:
        state, cursor = restore_pretraining_checkpoint(
            args.resume_from, state, fingerprint
        )

        print(
            f"Restored update {cursor.updates}, next sample index {cursor.next_index}"
        )

    validation_documents = make_joint_evaluation_documents(
        seed=config.seed,
        split="validation",
        examples_per_family=args.validation_per_family,
        max_length=config.max_length,
    )

    validation = pack_evaluation_groups(
        validation_documents, max_length=config.max_length
    )

    print_report(
        evaluate_families(state, validation, inputs), label="Initial joint validation"
    )

    last_saved_step = cursor.updates

    for _ in range(args.steps):
        batch, next_cursor = next_joint_batch(config, cursor)

        micro = split_microbatches(batch, args.microbatch_size)
        device_micro = move_like(micro, sharded_template)
        state, metrics = compiled_step(state, device_micro)

        # Wait for the update before committing its cursor.
        if not bool(jnp.isfinite(metrics.loss)):
            raise FloatingPointError(f"Non-finite loss at update {cursor.updates + 1}")

        cursor = next_cursor

        if int(state.step) != cursor.updates:
            raise RuntimeError("Optimizer step and stream cursor disagree")

        if cursor.updates % args.eval_every == 0:
            print(
                f"\nUpdate {cursor.updates} | joint={float(metrics.loss):.4f} | VM={float(metrics.vm_nats):.4f} | vision={float(metrics.visual_nats):.4f}"
            )

            report = evaluate_families(state, validation, inputs)
            print_report(report, label="Joint validation")

        if cursor.updates % args.save_every == 0:
            path = save_pretraining_checkpoint(
                args.checkpoint_root, state, cursor, fingerprint
            )
            last_saved_step = cursor.updates
            print(f"Saved: {path}")

    # save the last update
    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(
            args.checkpoint_root, state, cursor, fingerprint
        )

        print(f"Saved final: {path}")

    print(
        f"\nTotal: {cursor.updates} optimizer updates, {cursor.documents_seen:,} documents, {cursor.bytes_seen:,} bytes, {cursor.targets_seen:,} targets"
    )

    ood = make_joint_evaluation_documents(
        seed=config.seed,
        split="test",
        examples_per_family=args.validation_per_family,
        max_length=config.max_length,
        out_of_distribution=True,
    )

    ood_batches = pack_evaluation_groups(ood, max_length=config.max_length)
    print_report(
        evaluate_families(state, ood_batches, inputs),
        label="Held-out difficulty evaluation",
    )

    if args.exact_probes:
        probes = make_joint_evaluation_documents(
            seed=config.seed,
            split="test",
            examples_per_family=args.exact_probes,
            max_length=config.max_length,
        )
        exact = evaluate_exact_tasks(
            model, state.params, probes, examples_per_family=args.exact_probes
        )
        print_exact_report(exact)

    print("PASS")


if __name__ == "__main__":
    main()
