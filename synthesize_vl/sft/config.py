from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DifficultyWeights:
    easy: float = 0.30
    medium: float = 0.45
    hard: float = 0.25

    def __post_init__(self) -> None:
        weights = (self.easy, self.medium, self.hard)

        if any(not 0 <= w <= 1 for w in weights):
            raise ValueError("Invalid difficulty probability")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Difficulty probabilities must sum to 1")

    def probabilities(self) -> tuple[float, float, float]:
        return self.easy, self.medium, self.hard


@dataclass(frozen=True, slots=True)
class SFTDataConfig:
    seed: int = 2026
    batch_size: int = 8
    max_text_length: int = 1024
    text_probability: float = 0.5
    cot_probability: float = 0.85
    difficulty_weights: DifficultyWeights = field(default_factory=DifficultyWeights)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if not 16 <= self.max_text_length <= 1024:
            raise ValueError("Text length exceeds the frozen student context")

        if not 0 <= self.text_probability <= 1:
            raise ValueError("Invalid text probability")

        if not 0 <= self.cot_probability <= 1:
            raise ValueError("Invalid CoT probability")


@dataclass(frozen=True, slots=True)
class SFTConfig:
    pretrained_checkpoint_root: str
    data: SFTDataConfig

    def __post_init__(self) -> None:
        if not self.pretrained_checkpoint_root:
            raise ValueError("Missing pretrained checkpoint path")


def _validated_fields(cls: type, values: Any) -> dict:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a YAML mapping")  # noqa: TRY004

    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed

    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")

    return values


def load_sft_config(path: str | Path) -> SFTConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    _validated_fields(SFTConfig, raw)

    if set(raw) != {"pretrained_checkpoint_root", "data"}:
        raise ValueError("Missing SFT configuration section")

    data_values = dict(_validated_fields(SFTDataConfig, raw["data"]))

    difficulty = DifficultyWeights(
        **_validated_fields(
            DifficultyWeights, data_values.pop("difficulty_weights", {})
        )
    )

    return SFTConfig(
        pretrained_checkpoint_root=raw["pretrained_checkpoint_root"],
        data=SFTDataConfig(difficulty_weights=difficulty, **data_values),
    )
