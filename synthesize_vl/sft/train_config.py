from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from synthesize_vl.models.architecture import (
    ModelArchitecture,
    load_model_architecture,
)
from synthesize_vl.sft.config import DifficultyWeights, SFTDataConfig
from synthesize_vl.training.optimizers import AdvancedOptimizerConfig


@dataclass(frozen=True, slots=True)
class SFTRuntimeConfig:
    devices: int = 2
    microbatch_size: int = 4
    max_updates: int = 3000
    log_every: int = 50
    evaluate_every: int = 500
    save_every: int = 500
    checkpoint_root: str = "checkpoints/sft"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.devices,
            self.microbatch_size,
            self.max_updates,
            self.log_every,
            self.evaluate_every,
            self.save_every,
        )
        if any(value < 1 for value in values):
            raise ValueError("Runtime values must be positive")

        if not self.checkpoint_root:
            raise ValueError("Missing SFT checkpoint root")


@dataclass(frozen=True, slots=True)
class SFTEvaluationConfig:
    examples_per_group: int = 2
    exact_examples_per_level: int = 2
    max_new_tokens: int = 512
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.seed < 0 or self.examples_per_group < 1 or self.max_new_tokens < 1:
            raise ValueError("Invalid SFT evaluation settings")


@dataclass(frozen=True, slots=True)
class SFTTrainConfig:
    architecture: ModelArchitecture
    pretrained_checkpoint_root: str
    data: SFTDataConfig
    optimizer: AdvancedOptimizerConfig
    runtime: SFTRuntimeConfig
    evaluation: SFTEvaluationConfig

    def checkpoint_identity(self, *, source_checkpoint: str) -> dict[str, Any]:
        return {
            "format": "synthesized_sft",
            "architecture": asdict(self.architecture),
            "pretrained_source": source_checkpoint,
            "data": asdict(self.data),
            "optimizer": asdict(self.optimizer),
            "microbatch_size": self.runtime.microbatch_size,
            "devices": self.runtime.devices,
        }


def load_sft_train_config(path: str | Path) -> SFTTrainConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    architecture_path = Path(values["architecture_config"]).expanduser()
    architecture = load_model_architecture(path.parent / architecture_path)
    sft_data_config = values["data"]
    difficulty_weights = sft_data_config["difficulty_weights"]
    sft_data_config.pop("difficulty_weights", None)

    return SFTTrainConfig(
        architecture=architecture,
        pretrained_checkpoint_root=values["pretrained_checkpoint_root"],
        data=SFTDataConfig(
            difficulty_weights=DifficultyWeights(**difficulty_weights),
            **sft_data_config,
        ),
        optimizer=AdvancedOptimizerConfig(**values["optimizer"]),
        runtime=SFTRuntimeConfig(**values["runtime"]),
        evaluation=SFTEvaluationConfig(**values["evaluation"]),
    )
