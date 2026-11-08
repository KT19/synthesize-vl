import csv
from pathlib import Path
from typing import Any

from synthesize_vl.pretraining.evaluation import EvaluationReport
from synthesize_vl.pretraining.tasks import ARMS, TASK_PROBABILITIES

TRAIN_FIELDS = (
    "update",
    "joint_nats",
    "vm_nats",
    "visual_nats",
    "vm_targets",
    "visual_targets",
    "vm_sampling_probability",
    "visual_sampling_probability",
    "targets_seen",
    "bytes_seen",
)

VALIDATION_FIELDS = (
    "update",
    "task",
    "modality",
    "validation_nats_per_target",
    "validation_bits_per_byte",
    "bytes_evaluated",
    "sampling_probability",
)


class TrainingHistory:
    """Append raw training and validation metrics as CSV."""

    def __init__(self, root: str | Path, *, resume_update: int | None) -> None:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)

        self.train_path = directory / "train_metrics.csv"
        self.validation_path = directory / "validation_history.csv"

        self._prepare_file(self.train_path, TRAIN_FIELDS, resume_update=resume_update)
        self._prepare_file(
            self.validation_path, VALIDATION_FIELDS, resume_update=resume_update
        )

    @staticmethod
    def _prepare_file(
        path: Path, fields: tuple[str, ...], *, resume_update: int | None
    ) -> None:
        rows = []

        if resume_update is not None and path.exists():
            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = [
                    row
                    for row in csv.DictReader(stream)
                    if int(row["update"]) <= resume_update
                ]

        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _append(
        path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerows(rows)

    def record_training(
        self,
        *,
        update: int,
        joint_nats: float,
        vm_nats: float,
        visual_nats: float,
        vm_targets: int,
        visual_targets: int,
        targets_seen: int,
        bytes_seen: int,
    ) -> None:
        vm_probability = sum(
            probability
            for arm, probability in zip(ARMS, TASK_PROBABILITIES)
            if arm.modality == "vm"
        )

        row = {
            "update": update,
            "joint_nats": joint_nats,
            "vm_nats": vm_nats,
            "visual_nats": visual_nats,
            "vm_targets": vm_targets,
            "visual_targets": visual_targets,
            "vm_sampling_probability": vm_probability,
            "visual_sampling_probability": 1.0 - vm_probability,
            "targets_seen": targets_seen,
            "bytes_seen": bytes_seen,
        }

        self._append(self.train_path, TRAIN_FIELDS, [row])

    def record_validation(self, *, update: int, report: EvaluationReport) -> None:
        metrics = {item.family: item for item in report.families}
        rows = [
            {
                "update": update,
                "task": arm.name,
                "modality": arm.modality,
                "validation_nats_per_target": metrics[arm.name].nats_per_target,
                "validation_bits_per_byte": metrics[arm.name].bits_per_byte,
                "bytes_evaluated": metrics[arm.name].bytes_evaluated,
                "sampling_probability": probability,
            }
            for arm, probability in zip(ARMS, TASK_PROBABILITIES)
        ]

        self._append(self.validation_path, VALIDATION_FIELDS, rows)
