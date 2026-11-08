from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scripts import run_train_sft
from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.sft.dataset import generate_sft_example
from synthesize_vl.sft.train_config import load_sft_train_config


@pytest.mark.parametrize("show_work", [False, True])
def test_teacher_and_student_share_heldout_examples(show_work: bool) -> None:
    configs = [
        load_sft_train_config(path)
        for path in ("configs/sft_student.yaml", "configs/sft_teacher.yaml")
    ]

    for task in ("text_arithmetic", "visual_arithmetic"):
        examples = [
            generate_sft_example(
                replace(config.data, seed=config.evaluation.seed),
                split="test",
                index=700000,
                task_override=task,
                difficulty_override="medium",
                cot_override=show_work,
            )
            for config in configs
        ]
        student, teacher = examples
        assert student.prompt == teacher.prompt
        assert student.oracle_answer == teacher.oracle_answer
        if task == "visual_arithmetic":
            np.testing.assert_array_equal(
                np.asarray(student.image), np.asarray(teacher.image)
            )


def test_sft_runner_separates_training_and_evaluation_seeds(monkeypatch) -> None:
    config = load_sft_train_config("configs/sft_teacher.yaml")
    config = replace(
        config,
        data=replace(config.data, seed=17),
        evaluation=replace(config.evaluation, seed=23),
        runtime=replace(
            config.runtime,
            max_updates=1,
            evaluate_every=1,
            save_every=1,
            resume_from=None,
        ),
    )
    batch_size = config.data.batch_size
    batch = TrainingBatch(
        input_ids=jnp.zeros((batch_size, 4), dtype=jnp.int32),
        target_ids=jnp.zeros((batch_size, 4), dtype=jnp.int32),
        loss_mask=jnp.ones((batch_size, 4), dtype=jnp.bool_),
        images=jnp.zeros((batch_size, 64, 64, 3), dtype=jnp.float32),
        has_image=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )
    training_seeds = []
    validation_seeds = []
    generation_seeds = []
    sources = []

    def next_batch(data, cursor):
        training_seeds.append(data.seed)
        return batch, replace(
            cursor,
            next_index=cursor.next_index + batch_size,
            documents_seen=cursor.documents_seen + batch_size,
            updates=cursor.updates + 1,
        )

    def initialize(state, source):
        sources.append(source)
        return state

    def prepare(step_fn, state, microbatches, *, num_devices):
        def step(current, micro):
            updated = SimpleNamespace(params=current.params, step=current.step + 1)
            return updated, SimpleNamespace(loss=0.0, vm_nats=0.0, visual_nats=0.0)

        return step, state, microbatches

    monkeypatch.setattr(run_train_sft, "load_sft_train_config", lambda path: config)
    monkeypatch.setattr(run_train_sft, "resolve_pretrained_checkpoint", Path)
    monkeypatch.setattr(run_train_sft, "build_model", lambda architecture: None)
    monkeypatch.setattr(run_train_sft, "next_sft_batch", next_batch)
    monkeypatch.setattr(
        run_train_sft,
        "create_advanced_state",
        lambda *args, **kwargs: SimpleNamespace(params={}, step=0),
    )
    monkeypatch.setattr(run_train_sft, "initialize_sft_from_pretraining", initialize)
    monkeypatch.setattr(run_train_sft, "prepare_data_parallel", prepare)
    monkeypatch.setattr(
        run_train_sft,
        "make_sft_validation_batches",
        lambda data, **kwargs: {"seed": data.seed},
    )
    monkeypatch.setattr(
        run_train_sft,
        "run_validation",
        lambda state, validation, **kwargs: validation_seeds.append(validation["seed"]),
    )
    monkeypatch.setattr(
        run_train_sft,
        "evaluate_sft_generation",
        lambda model, params, data, settings: generation_seeds.append(data.seed),
    )
    monkeypatch.setattr(
        run_train_sft,
        "save_pretraining_checkpoint",
        lambda root, *args: Path(root) / "step_00000001",
    )

    run_train_sft.main("configs/sft_teacher.yaml")

    assert training_seeds == [17, 17]
    assert validation_seeds == [23, 23, 23]
    assert generation_seeds == [23, 23]
    assert sources == [Path(config.pretrained_checkpoint_root)]
