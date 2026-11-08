from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from synthesize_vl.data.packing import pack_examples
from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.legacy.synthetic import make_dataset
from synthesize_vl.legacy.vision import ObjectSpec, make_question, render_scene
from synthesize_vl.training.state import OptimizerConfig, create_train_state
from synthesize_vl.training.steps import eval_step, train_step

ModelKind = Literal["text", "recurrent"]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    name: str
    parameters: int
    train_loss: float
    heldout_loss: float
    visual_accuracy: float


def make_tiny_multimodal_batch(seed: int = 42) -> TrainingBatch:
    """Create reproducible text, math, code, and vision examples."""
    text_examples = make_dataset(seed=seed, count=2)
    samples = [
        TrainingSample(
            task=example.kind, prompt=example.prompt + " ", answer=example.answer
        )
        for example in text_examples
    ]

    # Cross digit identity and color so color alone cannot distinguish 3 from 7.
    visual_specs = [
        ObjectSpec("digit", "red", "3"),
        ObjectSpec("digit", "red", "7"),
        ObjectSpec("digit", "blue", "3"),
        ObjectSpec("digit", "blue", "7"),
        ObjectSpec("circle", "green"),
        ObjectSpec("square", "purple"),
    ]

    root_key = jax.random.key(seed)

    for index, spec in enumerate(visual_specs):
        key = jax.random.fold_in(root_key, index + 100)
        scene = render_scene([spec], key)
        qa = make_question(scene, "identify")
        samples.append(
            TrainingSample(
                task=qa.task,
                prompt=qa.question + "\nAnswer: ",
                answer=qa.answer,
                image=scene.image,
            )
        )

    return pack_examples(samples, max_text_length=128)


def model_inputs(batch: TrainingBatch) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the inputs accepted by the multimodal language models."""
    return (batch.input_ids, batch.images, batch.has_image)


def count_parameters(tree: object) -> int:
    """Count array elements in a parameter pytree or variables tree."""
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree))


def first_answer_accuracy(state: TrainState, batch: TrainingBatch) -> jax.Array:
    """Return first-answer-token accuracy over visual samples."""
    logits = state.apply_fn({"params": state.params}, *model_inputs(batch))
    first = jnp.argmax(batch.loss_mask, axis=1)
    rows = jnp.arange(batch.input_ids.shape[0])
    predictions = jnp.argmax(logits[rows, first, :], axis=-1)
    targets = batch.target_ids[rows, first]
    correct = (predictions == targets) & batch.has_image

    return jnp.sum(correct) / jnp.maximum(jnp.sum(batch.has_image), 1)


def run_experiment(
    name: str,
    model: nn.Module,
    train_batch: TrainingBatch,
    heldout_batch: TrainingBatch,
    steps: int = 150,
) -> ExperimentResult:
    """Train and evaluate one architecture on the shared multimodal task."""
    inputs = model_inputs(train_batch)
    state = create_train_state(
        model, inputs, seed=0, config=OptimizerConfig(learning_rate=1e-3)
    )
    parameters = count_parameters(state.params)
    initial_loss, _ = eval_step(state, train_batch, inputs)

    print(f"\n{name}")
    print(f"Parameters: {parameters:,}")
    print(f"Initial loss: {float(initial_loss):.4f}")

    for step in range(1, steps + 1):
        state, _ = train_step(state, train_batch, inputs)

        if step % 50 == 0 or step == steps:
            loss, accuracy = eval_step(state, train_batch, inputs)
            print(
                f"Step {step:3d} | loss={float(loss):.4f} | accuracy={float(accuracy):.1%}"
            )

    train_loss, _ = eval_step(state, train_batch, inputs)
    heldout_loss, _ = eval_step(state, heldout_batch, model_inputs(heldout_batch))
    visual_accuracy = first_answer_accuracy(state, heldout_batch)

    return ExperimentResult(
        name=name,
        parameters=parameters,
        train_loss=float(train_loss),
        heldout_loss=float(heldout_loss),
        visual_accuracy=float(visual_accuracy),
    )
