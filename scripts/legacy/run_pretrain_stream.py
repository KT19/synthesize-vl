import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from scripts.smoke.run_pretrain_tiny import ModelKind, make_model
from synthesize_vl.data.schema import StreamCursor, TrainingBatch
from synthesize_vl.legacy.evaluation import make_evaluation_batches
from synthesize_vl.legacy.stream import StreamConfig, next_training_batch
from synthesize_vl.legacy.vm_evaluation import make_vm_evaluation_batches
from synthesize_vl.pretraining.checkpoints import (
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.pretraining.evaluation import evaluate_families, print_report
from synthesize_vl.training.accumulation import (
    make_accumulated_step,
    split_microbatches,
)
from synthesize_vl.training.distributed import prepare_data_parallel
from synthesize_vl.training.optimizers import (
    AdvancedOptimizerConfig,
    create_advanced_state,
)


def run_configuration(
    stream: StreamConfig,
    optimizer: AdvancedOptimizerConfig,
    model_kind: ModelKind,
    microbatch_size: int,
) -> dict[str, Any]:
    stream_fields = asdict(stream)

    if stream.generator == "legacy":
        stream_fields.pop("generator")
        stream_fields.pop("vm_profile")

    return {
        "format_version": 1,
        "generator_version": 2 if stream.generator == "vm" else 1,
        "model_version": 1,
        "model_kind": model_kind,
        "stream": stream_fields,
        "optimizer": asdict(optimizer),
        "microbatch_size": microbatch_size,
    }


def move_like(source: Any, template: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value, reference: jax.device_put(value, reference.sharding),
        source,
        template,
    )


def model_inputs(batch: TrainingBatch, kind: ModelKind) -> tuple[jax.Array, ...]:
    if kind == "text":
        return (batch.input_ids,)

    return (batch.input_ids, batch.images, batch.has_image)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=("text", "recurrent"), default="text")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--target-bytes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--validation-per-family", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--schedule-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/stream")
    )
    parser.add_argument("--resume-from", type=Path, default=None)

    parser.add_argument("--generator", choices=("legacy", "vm"), default="vm")
    parser.add_argument(
        "--vm-profile",
        choices=("mixed", "foundation", "intermediate", "advanced"),
        default="mixed",
    )
    parser.add_argument("--vm-ood-test", action="store_true")

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

    if args.target_bytes is not None and args.target_bytes <= 0:
        raise ValueError("target-bytes must be positive")

    if args.batch_size % args.microbatch_size:
        raise ValueError("batch-size must be divisible by microbatch-size")

    if args.microbatch_size % args.devices:
        raise ValueError("microbatch-size must be divisible by devices")

    kind: ModelKind = args.model

    stream_config = StreamConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        generator=args.generator,
        vm_profile=args.vm_profile,
    )
    optimizer_config = AdvancedOptimizerConfig(
        muon_lr=3e-3,
        adamw_lr=1e-3,
        total_steps=args.schedule_steps,
        warmup_steps=args.warmup_steps,
    )

    configuration = run_configuration(
        stream_config, optimizer_config, kind, args.microbatch_size
    )

    # Initialize a model and a batch with final shapes
    cursor = StreamCursor()

    first_batch, _ = next_training_batch(stream_config, cursor)
    model = make_model(kind, stream_config.max_length)
    state = create_advanced_state(
        model, model_inputs(first_batch, kind), seed=0, config=optimizer_config
    )

    # One optimizer update after several microbatches
    step_fn = make_accumulated_step(multimodal=(kind == "recurrent"))
    first_micro = split_microbatches(first_batch, args.microbatch_size)

    compiled_step, state, sharded_template = prepare_data_parallel(
        step_fn, state, first_micro, num_devices=args.devices
    )

    if args.resume_from is not None:
        state, cursor = restore_pretraining_checkpoint(
            args.resume_from, state, configuration
        )

        print(
            f"Restored update {cursor.updates}, next sample index {cursor.next_index}"
        )

    if args.generator == "vm":
        validation = make_vm_evaluation_batches(
            seed=stream_config.seed,
            split="validation",
            examples_per_profile=args.validation_per_family,
            max_length=stream_config.max_length,
        )

    else:
        validation = make_evaluation_batches(
            seed=stream_config.seed,
            split="validation",
            examples_per_family=args.validation_per_family,
            max_length=stream_config.max_length,
        )

    def input_fn(batch):
        return model_inputs(batch, kind)

    initial_report = evaluate_families(state, validation, input_fn)
    print_report(initial_report, label=f"Validation at update {cursor.updates}")
    last_saved_step = cursor.updates

    for _ in range(args.steps):
        if args.target_bytes is not None and cursor.bytes_seen >= args.target_bytes:
            print("Data budget reached")
            break

        batch, advanced_cursor = next_training_batch(stream_config, cursor)

        micro = split_microbatches(batch, args.microbatch_size)
        device_micro = move_like(micro, sharded_template)
        state, loss = compiled_step(state, device_micro)

        # Wait for the update before committing its cursor.
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at update {cursor.updates + 1}")

        cursor = advanced_cursor

        if int(state.step) != cursor.updates:
            raise RuntimeError("Optimizer step and stream cursor disagree")

        if cursor.updates % args.eval_every == 0:
            print(
                f"\nUpdate {cursor.updates} | Train loss {float(loss):.4f} | Bytes {cursor.bytes_seen:,} | Targets {cursor.targets_seen:,}"
            )

            report = evaluate_families(state, validation, input_fn)
            print_report(report, label="Validation")

        if cursor.updates % args.save_every == 0:
            path = save_pretraining_checkpoint(
                args.checkpoint_root, state, cursor, configuration
            )
            last_saved_step = cursor.updates
            print(f"Saved: {path}")

    # save the last update
    if cursor.updates != last_saved_step:
        path = save_pretraining_checkpoint(
            args.checkpoint_root, state, cursor, configuration
        )

        print(f"Saved final: {path}")

    print(
        f"\nTotal: {cursor.updates} optimizer updates, {cursor.documents_seen:,} documents, {cursor.bytes_seen:,} bytes, {cursor.targets_seen:,} targets"
    )

    if args.evaluate_test:
        if args.generator == "vm":
            test_batches = make_vm_evaluation_batches(
                seed=stream_config.seed,
                split="test",
                examples_per_profile=args.validation_per_family,
                max_length=stream_config.max_length,
            )

        else:
            test_batches = make_evaluation_batches(
                seed=stream_config.seed,
                split="test",
                examples_per_family=args.validation_per_family,
                max_length=stream_config.max_length,
            )

        test_report = evaluate_families(state, test_batches, input_fn)
        print_report(test_report, label="Final held-out test")

    if args.vm_ood_test:
        if args.generator != "vm":
            raise ValueError("--vm-ood-test requires --generator vm")

        ood_batches = make_vm_evaluation_batches(
            seed=stream_config.seed,
            split="test",
            examples_per_profile=args.validation_per_family,
            max_length=stream_config.max_length,
            out_of_distribution=True,
        )

        ood_report = evaluate_families(state, ood_batches, input_fn)

        print_report(ood_report, label="Held-out VM difficulty test")

    print("PASS")


if __name__ == "__main__":
    main()
