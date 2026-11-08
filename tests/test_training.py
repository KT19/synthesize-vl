import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.models.transformer import CausalLM
from synthesize_vl.training.losses import masked_lm_loss
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import eval_step, train_step


def make_test_setup() -> tuple[TrainingBatch, TrainState]:
    batch = pack_examples([TrainingSample("math", "Q: 1+1=", "2")], max_text_length=32)

    model = CausalLM(
        hidden_size=32, num_heads=4, num_layers=1, intermediate_size=64, max_length=32
    )

    state = create_train_state(
        model, (batch.input_ids,), seed=0, config=OptimizerConfig(learning_rate=3e-3)
    )

    return batch, state


def test_loss_mask_ignores_unsupervised_target() -> None:
    batch, _ = make_test_setup()

    logits = jax.random.normal(jax.random.key(4), (1, 32, 260))
    original = masked_lm_loss(logits, batch)
    changed = batch.replace(target_ids=batch.target_ids.at[0, -1].set(123))  # type: ignore

    np.testing.assert_allclose(original, masked_lm_loss(logits, changed))


def test_shared_step_updates_parameters() -> None:
    batch, state = make_test_setup()

    updated, loss = train_step(state, batch)

    assert int(updated.step) == 1
    assert bool(jnp.isfinite(loss))

    assert any(
        not bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(updated.params),
        )
    )


def test_explicit_model_inputs_match_default() -> None:
    batch, state = make_test_setup()

    default = eval_step(state, batch, (batch.input_ids,))
    explicit = eval_step(state, batch, (batch.input_ids,))

    for a, b in zip(default, explicit):
        np.testing.assert_allclose(a, b)
