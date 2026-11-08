from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

from synthesize_vl.models.architecture import build_model
from synthesize_vl.pretraining.config import load_pretrain_config
from synthesize_vl.sft.train_config import load_sft_train_config


@pytest.mark.parametrize(
    "filename,loader",
    [
        ("pretrain.yaml", load_pretrain_config),
        ("sft_student.yaml", load_sft_train_config),
        ("sft_teacher.yaml", load_sft_train_config),
    ],
)
def test_external_architecture_controls_model(tmp_path, filename, loader) -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    architecture = yaml.safe_load((configs / "student_architecture.yaml").read_text())
    architecture.update(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        recurrent_steps=2,
        core_layers=3,
        max_text_length=512,
        position_mode="nope",
        use_attention_gate=False,
        core_attention="full",
    )
    (tmp_path / "architecture.yaml").write_text(yaml.safe_dump(architecture))

    values = yaml.safe_load((configs / filename).read_text())
    values["architecture_config"] = "architecture.yaml"
    if filename != "pretrain.yaml":
        values["data"]["max_text_length"] = 512
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(values))

    config = loader(path)
    assert asdict(config.architecture) == architecture

    model = build_model(config.architecture)
    assert model.hidden_size == 32
    assert model.recurrent_steps == 2
    assert model.core_layers == 3
    assert model.max_text_length == 512
    assert model.position_mode == "nope"
    assert not model.use_attention_gate
    assert model.core_attention == "full"

    variables = jax.eval_shape(
        model.init,
        jax.random.key(0),
        jax.ShapeDtypeStruct((1, 4), jnp.int32),
        jax.ShapeDtypeStruct((1, 64, 64, 3), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.bool_),
    )
    embedding = variables["params"]["multimodal_embedding"]["text_embedding"]["token_embedding"]["embedding"]
    assert embedding.shape == (260, 32)
    assert {name for name in variables["params"]["core"] if name.startswith("shared_mhc_")} == {
        "shared_mhc_0",
        "shared_mhc_1",
        "shared_mhc_2",
    }

    identity = (
        config.checkpoint_identity(source_checkpoint="source")
        if filename != "pretrain.yaml"
        else config.checkpoint_identity()
    )
    assert identity["architecture"] == architecture
