from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from synthesize_vl.models.architecture import ModelArchitecture, load_model_architecture
from synthesize_vl.sft.config import DifficultyWeights, SFTDataConfig
from synthesize_vl.sft.train_config import SFTEvaluationConfig
from synthesize_vl.training.optimizers import AdvancedOptimizerConfig


@dataclass(frozen=True, slots=True)
class GRPODataConfig:
    group_size: int = 8

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("Group-relative advantages need group_size >= 2")


@dataclass(frozen=True, slots=True)
class GRPORolloutConfig:
    seed: int = 2507
    context_length: int = 512
    new_tokens: int = 256
    clip_epsilon: float = 0.2
    ppo_epochs: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.new_tokens < self.context_length:
            raise ValueError("Invalid rollout length")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("Invalid clip epsilon")


@dataclass(frozen=True, slots=True)
class GRPORuntimeConfig:
    devices: int = 2
    max_updates: int = 500
    log_every: int = 10
    evaluate_every: int = 50
    save_every: int = 50
    max_uninformative_streak: int = 10

    checkpoint_root: str = "checkpoints/grpo"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if min(self.devices, self.max_updates, self.log_every, self.evaluate_every, self.save_every) < 1:
            raise ValueError("Runtime values must be positive")


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    policy_checkpoint_root: str
    architecture: ModelArchitecture
    data: SFTDataConfig
    grpo: GRPODataConfig
    optimizer: AdvancedOptimizerConfig
    rollout: GRPORolloutConfig
    runtime: GRPORuntimeConfig
    evaluation: SFTEvaluationConfig

    def __post_init__(self) -> None:
        if (self.data.batch_size * self.grpo.group_size) % self.runtime.devices:
            raise ValueError("Rollout count must divide across devices")

    def checkpoint_identity(self, source_checkpoint: str) -> dict[str, Any]:
        return {
            "format": "synthesize_vl_grpo",
            "architecture": asdict(self.architecture),
            "policy_source": source_checkpoint,
            "data": asdict(self.data),
            "group_size": self.grpo.group_size,
            "optimizer": asdict(self.optimizer),
            "rollout": asdict(self.rollout),
            "devices": self.runtime.devices,
        }


def load_grpo_config(path: str | Path) -> GRPOConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    sft_data_config = values["data"]
    difficulty_weights = sft_data_config["difficulty_weights"]
    sft_data_config.pop("difficulty_weights", None)

    return GRPOConfig(
        policy_checkpoint_root=values["policy_checkpoint_root"],
        architecture=load_model_architecture(values["architecture_config"]),
        data=SFTDataConfig(difficulty_weights=DifficultyWeights(**difficulty_weights), **sft_data_config),
        grpo=GRPODataConfig(**values["grpo"]),
        optimizer=AdvancedOptimizerConfig(**values["optimizer"]),
        rollout=GRPORolloutConfig(**values["rollout"]),
        runtime=GRPORuntimeConfig(**values["runtime"]),
        evaluation=SFTEvaluationConfig(**values["evaluation"]),
    )
