from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from synthesize_vl.models.architecture import ModelArchitecture, load_model_architecture
from synthesize_vl.sft.config import DifficultyWeights, SFTDataConfig
from synthesize_vl.sft.train_config import SFTEvaluationConfig
from synthesize_vl.training.optimizers import AdvancedOptimizerConfig


@dataclass(frozen=True, slots=True)
class OPDRolloutConfig:
    seed: int = 2407
    context_length: int = 512
    new_tokens: int = 256
    sampling_temperature: float = 1.0
    distillation_temperature: float = 1.0
    visual_loss_share: float = 0.5

    def __post_init__(self) -> None:
        if not (1 <= self.new_tokens < self.context_length):
            raise ValueError("Invalid rollout length")
        if not 0 < self.visual_loss_share < 1:
            raise ValueError("Invalid visual loss share")


@dataclass(frozen=True, slots=True)
class OPDRuntimeConfig:
    devices: int = 2
    max_updates: int = 500
    log_every: int = 50
    evaluate_every: int = 100
    save_every: int = 50
    checkpoint_root: str = "checkpoints/opd"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if min(self.devices, self.max_updates, self.log_every, self.evaluate_every, self.save_every) < 1:
            raise ValueError("Invalid OPD runtime settings")


@dataclass(frozen=True, slots=True)
class OPDConfig:
    student_checkpoint_root: str
    teacher_checkpoint_root: str
    teacher_architecture: ModelArchitecture
    student_architecture: ModelArchitecture
    data: SFTDataConfig
    optimizer: AdvancedOptimizerConfig
    rollout: OPDRolloutConfig
    runtime: OPDRuntimeConfig
    evaluation: SFTEvaluationConfig

    def __post_init__(self) -> None:
        if self.data.batch_size % self.runtime.devices:
            raise ValueError("Batch size must divide across devices")

        if self.runtime.max_updates > self.optimizer.total_steps:
            raise ValueError("OPD run exceeds optimizer schedule")

    def checkpoint_identity(self, *, student_source: str, teacher_source: str) -> dict[str, Any]:
        return {
            "format": "synthesize_vl_opd",
            "student_source": student_source,
            "teacher_source": teacher_source,
            "student_architecture": asdict(self.student_architecture),
            "teacher_architecture": asdict(self.teacher_architecture),
            "data": asdict(self.data),
            "optimizer": asdict(self.optimizer),
            "rollout": asdict(self.rollout),
            "devices": self.runtime.devices,
        }


def load_opd_config(path: str | Path) -> OPDConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    sft_data_config = values["data"]
    difficulty_weights = sft_data_config["difficulty_weights"]
    sft_data_config.pop("difficulty_weights", None)

    return OPDConfig(
        student_checkpoint_root=values["student_checkpoint_root"],
        teacher_checkpoint_root=values["teacher_checkpoint_root"],
        teacher_architecture=load_model_architecture(values["teacher_architecture_config"]),
        student_architecture=load_model_architecture(values["student_architecture_config"]),
        data=SFTDataConfig(difficulty_weights=DifficultyWeights(**difficulty_weights), **sft_data_config),
        optimizer=AdvancedOptimizerConfig(**values["optimizer"]),
        rollout=OPDRolloutConfig(**values["rollout"]),
        runtime=OPDRuntimeConfig(**values["runtime"]),
        evaluation=SFTEvaluationConfig(**values["evaluation"]),
    )
