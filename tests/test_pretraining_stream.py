import jax
import numpy as np

from synthesize_vl.data.schema import StreamCursor
from synthesize_vl.legacy.evaluation import make_evaluation_batches
from synthesize_vl.legacy.generators import FAMILIES, generate_documents
from synthesize_vl.legacy.stream import StreamConfig, next_training_batch
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.pretraining.checkpoints import (
    configuration_digest,
    restore_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import train_step


def smal_model() -> CausalLM:
    return CausalLM(
        hidden_size=16,
        num_heads=2,
        num_layers=1,
        intermediate_size=32,
        max_length=128,
        position_mode="rope",
    )


def test_stream_is_reproducible() -> None:
    config = StreamConfig(seed=17, batch_size=4, max_length=128)

    start = StreamCursor()

    first, after_first = next_training_batch(config, start)
    second, after_second = next_training_batch(config, after_first)

    # Generating the same cursor twice yields identical input and target tokens
    replay, replay_cursor = next_training_batch(config, start)

    np.testing.assert_array_equal(first.input_ids, replay.input_ids)

    assert replay_cursor == after_first

    expected = generate_documents(seed=17, split="train", start=0, count=8)

    assert after_second.next_index == 8
    assert after_second.documents_seen == 8
    assert after_second.updates == 2

    assert after_second.bytes_seen == sum(len(doc.payload) for doc in expected)
    assert after_second.targets_seen == (after_second.bytes_seen + 8)
    assert not np.array_equal(first.input_ids, second.input_ids)


def test_stream_consumes_exact_supervised_targets() -> None:
    config = StreamConfig(seed=2, batch_size=4, max_length=128)

    batch, cursor = next_training_batch(config, StreamCursor())

    assert cursor.targets_seen == int(batch.loss_mask.sum())
    assert cursor.bytes_seen == (cursor.targets_seen - cursor.documents_seen)


def test_evaluation_is_balanced_and_deterministic() -> None:
    first = make_evaluation_batches(seed=42, split="validation", examples_per_family=3, max_length=128)
    second = make_evaluation_batches(seed=42, split="validation", examples_per_family=3, max_length=128)

    assert set(first) == set(FAMILIES)

    for family in FAMILIES:
        assert first[family].input_ids.shape == (3, 128)

        np.testing.assert_array_equal(first[family].target_ids, second[family].target_ids)


def test_configuration_fingerprint_is_stable() -> None:
    first = configuration_digest({"seed": 7, "batch_size": 8})
    same = configuration_digest({"batch_size": 8, "seed": 7})
    changed = configuration_digest({"seed": 8, "batch_size": 8})

    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, changed)


def test_checkpoint_restores_stream_and_optimizer(tmp_path) -> None:
    config = StreamConfig(seed=5, batch_size=4, max_length=128)

    batch, advanced = next_training_batch(config, StreamCursor())

    model = smal_model()
    template = create_train_state(model, (batch.input_ids,), seed=0, config=OptimizerConfig(learning_rate=2e-3))

    trained, _ = train_step(template, batch)

    settings = {
        "test": "stream_resume",
        "seed": config.seed,
        "batch_size": config.batch_size,
        "optimizer": "adamw",
    }

    path = save_pretraining_checkpoint(tmp_path, trained, advanced, settings)
    restored, restored_cursor = restore_pretraining_checkpoint(path, template, settings)

    assert restored_cursor == advanced
    assert int(restored.step) == 1

    for before, after in zip(
        jax.tree_util.tree_leaves(trained.opt_state),
        jax.tree_util.tree_leaves(restored.opt_state),
    ):
        np.testing.assert_array_equal(before, after)

    next_a, cursor_a = next_training_batch(config, advanced)
    next_b, cursor_b = next_training_batch(config, restored_cursor)

    assert cursor_a == cursor_b

    np.testing.assert_array_equal(next_a.input_ids, next_b.input_ids)
