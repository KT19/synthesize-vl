from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from synthesize_vl.models.architecture import (
    ModelArchitecture,
    load_model_architecture,
)
from synthesize_vl.pretraining.tasks import ARMS, TASK_PROBABILITIES
from synthesize_vl.training.optimizers import AdvancedOptimizerConfig


@dataclass(frozen=True, slots=True)
class DataConfig:
    seed: int = 2026
    batch_size: int = 8
    microbatch_size: int = 4
    visual_loss_share: float = 0.5
    vm_execute_probability: float = 0.75
    validation_per_arm: int = 2

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if self.batch_size < 1 or self.microbatch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.batch_size % self.microbatch_size:
            raise ValueError("Invalid microbatch partition")
        if not 0 < self.visual_loss_share < 1:
            raise ValueError("Invalid visual loss share")
        if not 0 <= self.vm_execute_probability <= 1:
            raise ValueError("Invalid VM execute probability")
        if self.validation_per_arm < 1:
            raise ValueError("Invalid validation sample count")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    devices: int = 2
    max_updates: int = 10000
    log_every: int = 20
    evaluate_every: int = 400
    save_every: int = 500
    checkpoint_root: str = "checkpoints/pretrain_fixed"
    resume_from: str | None = None
    evaluate_ood_at_end: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.devices,
                self.max_updates,
                self.log_every,
                self.evaluate_every,
                self.save_every,
            )
            < 1
        ):
            raise ValueError("Invalid runtime settings")
        if not self.checkpoint_root:
            raise ValueError("Missing checkpoint root")


@dataclass(frozen=True, slots=True)
class PretrainConfig:
    architecture: ModelArchitecture
    data: DataConfig
    optimizer: AdvancedOptimizerConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.data.microbatch_size % self.runtime.devices:
            raise ValueError("Microbatch size must be divisible by devices")

    def checkpoint_identity(self) -> dict[str, Any]:
        return {
            "format": "fixed_pretrain",
            "architecture": asdict(self.architecture),
            "data": asdict(self.data),
            "optimizer": asdict(self.optimizer),
            "devices": self.runtime.devices,
            "evaluate_every": self.runtime.evaluate_every,
            "task_catalog": [arm.name for arm in ARMS],
            "task_probabilities": list(TASK_PROBABILITIES),
        }


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    if "architecture_config" in values:
        architecture_path = Path(values["architecture_config"]).expanduser()
        architecture = load_model_architecture(path.parent / architecture_path)
    else:
        # Previously saved resolved configurations contain inline architecture values.
        architecture = ModelArchitecture(**values["architecture"])

    return PretrainConfig(
        architecture=architecture,
        data=DataConfig(**values["data"]),
        optimizer=AdvancedOptimizerConfig(**values["optimizer"]),
        runtime=RuntimeConfig(**values["runtime"]),
    )
