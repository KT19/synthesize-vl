import jax
import jax.numpy as jnp

from scripts.smoke.experiment_utils import (
    count_parameters,
    make_tiny_multimodal_batch,
    model_inputs,
)
from synthesize_vl.models.recurrent import RecurrentMultimodalCausalLM


def main() -> None:
    batch = make_tiny_multimodal_batch(seed=42)
    inputs = model_inputs(batch)

    def make_model(steps: int) -> RecurrentMultimodalCausalLM:
        return RecurrentMultimodalCausalLM(
            hidden_size=32, num_heads=4, intermediate_size=64, recurrent_steps=steps
        )

    model_1 = make_model(1)
    model_3 = make_model(3)

    vars_1 = model_1.init(jax.random.key(0), *inputs)
    vars_3 = model_3.init(jax.random.key(0), *inputs)

    logits_1 = model_1.apply(vars_1, *inputs)
    logits_3 = model_3.apply(vars_3, *inputs)

    params_1 = count_parameters(vars_1["params"])  # type: ignore
    params_3 = count_parameters(vars_3["params"])  # type: ignore

    print("1-loop logits:", logits_1.shape)  # type: ignore
    print("3-loop logits:", logits_3.shape)  # type: ignore
    print("1-loop parameters:", params_1)
    print("3-loop parameters:", params_3)

    assert logits_1.shape == (8, 128, 260)  # type: ignore
    assert logits_3.shape == logits_1.shape  # type: ignore
    assert params_1 == params_3

    difference = jnp.linalg.norm(logits_1 - logits_3)  # type: ignore

    print("Output difference:", float(difference))
    assert float(difference) > 0

    print("PASS")


if __name__ == "__main__":
    main()
