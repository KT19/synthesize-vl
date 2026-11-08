import csv
from pathlib import Path

import numpy as np

from synthesize_vl.data.schema import StreamCursor
from synthesize_vl.pretraining.config import DataConfig, load_pretrain_config
from synthesize_vl.pretraining.evaluation import EvaluationReport, FamilyMetrics
from synthesize_vl.pretraining.history import TrainingHistory
from synthesize_vl.pretraining.task_stream import make_task_validation, next_task_batch
from synthesize_vl.pretraining.tasks import ARMS, TASK_PROBABILITIES, sample_arm


def test_fixed_task_mixture() -> None:
    assert len(ARMS) == 15
    assert sum(arm.modality == "vm" for arm in ARMS) == 3
    assert sum(arm.modality == "visual" for arm in ARMS) == 12

    np.testing.assert_allclose(sum(TASK_PROBABILITIES), 1.0)

    for modality in ("vm", "visual"):
        probabilities = [
            p for arm, p in zip(ARMS, TASK_PROBABILITIES) if arm.modality == modality
        ]
        np.testing.assert_allclose(sum(probabilities), 0.5)
        np.testing.assert_allclose(probabilities, 0.5 / len(probabilities))


def test_task_sampling_is_reproducible() -> None:
    first = [sample_arm(seed=42, index=i) for i in range(1024)]
    replay = [sample_arm(seed=42, index=i) for i in range(1024)]

    assert first == replay
    assert {arm.name for arm in first} == {arm.name for arm in ARMS}
    assert abs(sum(arm.modality == "vm" for arm in first) / len(first) - 0.5) < 0.05


def test_yaml_configuration_loads_nested_settings() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "pretrain.yaml"
    config = load_pretrain_config(path)
    identity = config.checkpoint_identity()

    assert config.architecture.position_mode == "rope"
    assert config.optimizer.total_steps == config.runtime.max_updates
    assert identity["format"] == "fixed_pretrain"
    assert "curriculum" not in identity
    assert identity["task_catalog"] == [arm.name for arm in ARMS]
    assert identity["task_probabilities"] == list(TASK_PROBABILITIES)


def test_task_stream_replays_from_cursor() -> None:
    data = DataConfig(seed=17)
    first, advanced = next_task_batch(data=data, max_length=1024, cursor=StreamCursor())
    replay, replay_cursor = next_task_batch(
        data=data, max_length=1024, cursor=StreamCursor()
    )

    np.testing.assert_array_equal(first.input_ids, replay.input_ids)
    np.testing.assert_array_equal(first.images, replay.images)
    assert advanced == replay_cursor
    assert advanced.next_index == advanced.documents_seen == data.batch_size
    assert advanced.targets_seen == int(first.loss_mask.sum())

    second, after_second = next_task_batch(data=data, max_length=1024, cursor=advanced)
    replay_second, restored_cursor = next_task_batch(
        data=data, max_length=1024, cursor=replay_cursor
    )

    np.testing.assert_array_equal(second.target_ids, replay_second.target_ids)
    assert after_second == restored_cursor
    assert after_second.updates == 2
    assert after_second.targets_seen == advanced.targets_seen + int(
        second.loss_mask.sum()
    )


def test_task_validation_covers_all_arms() -> None:
    data = DataConfig(validation_per_arm=2)
    groups = make_task_validation(data=data, max_length=1024)

    assert set(groups) == {arm.name for arm in ARMS}

    for arm in ARMS:
        batch = groups[arm.name]

        assert batch.input_ids.shape == (data.validation_per_arm, 1024)
        assert int(batch.loss_mask.sum()) > 0

        if arm.modality == "vm":
            # Validation keeps execution-only supervision; the program prefix is masked.
            assert not bool(batch.loss_mask[:, 0].any())


def test_history_records_validation_and_trims_on_resume(tmp_path: Path) -> None:
    families = tuple(FamilyMetrics(arm.name, 2, 16, 1.5, 2.16) for arm in ARMS)
    report = EvaluationReport(families, 1.5, 2.16)
    history = TrainingHistory(tmp_path, resume_update=None)
    history.record_validation(update=0, report=report)

    for update in (1, 2):
        history.record_training(
            update=update,
            joint_nats=1.0,
            vm_nats=1.5,
            visual_nats=0.5,
            vm_targets=10,
            visual_targets=10,
            targets_seen=20 * update,
            bytes_seen=18 * update,
        )

    history.record_validation(update=2, report=report)
    restored = TrainingHistory(tmp_path, resume_update=1)

    with restored.train_path.open() as stream:
        training = list(csv.DictReader(stream))

    assert [int(row["update"]) for row in training] == [1]
    assert float(training[0]["vm_sampling_probability"]) == 0.5

    with restored.validation_path.open() as stream:
        validation = list(csv.DictReader(stream))

    assert len(validation) == len(ARMS)
    assert {row["update"] for row in validation} == {"0"}
    assert {row["task"] for row in validation} == {arm.name for arm in ARMS}
    assert "progress_ema" not in validation[0]
    assert "feedback_round" not in validation[0]
