from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from flax import linen as nn

from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    hidden_size: int = 896
    num_heads: int = 14
    intermediate_size: int = 3584

    recurrent_steps: int = 4
    core_layers: int = 1
    residual_type: str = "mhc"
    num_streams: int = 2
    sinkhorn_iterations: int = 20

    position_mode: str = "rope"
    use_attention_gate: bool = True

    core_attention: str = "hybrid"
    local_window: int = 64
    global_text_tokens: int = 2

    max_text_length: int = 1024
    image_size: int = 64
    patch_size: int = 8
    vocab_size: int = 260

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("Hidden size must divide evenly into heads")

        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")

        if self.max_text_length < 128:
            raise ValueError("Context length is unexpectedly small")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_model_architecture(path: str | Path) -> ModelArchitecture:
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    return ModelArchitecture(**values)


def build_model(config: ModelArchitecture) -> nn.Module:
    return RecurrentMultimodalCausalLM(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        recurrent_steps=config.recurrent_steps,
        core_layers=config.core_layers,
        residual_type=config.residual_type,
        num_streams=config.num_streams,
        sinkhorn_iterations=config.sinkhorn_iterations,
        position_mode=config.position_mode,
        use_attention_gate=config.use_attention_gate,
        core_attention=config.core_attention,
        local_window=config.local_window,
        global_text_tokens=config.global_text_tokens,
        max_text_length=config.max_text_length,
        image_size=config.image_size,
        patch_size=config.patch_size,
        vocab_size=config.vocab_size,
    )
