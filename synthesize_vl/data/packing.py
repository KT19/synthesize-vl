from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from synthesize_vl.data.schema import TrainingBatch, TrainingSample
from synthesize_vl.tokenizer import ByteTokenizer


def pack_examples(
    samples: Sequence[TrainingSample], max_text_length: int = 128
) -> TrainingBatch:
    if not samples:
        raise ValueError("Cannot pack an empty batch")

    if max_text_length < 1:
        raise ValueError("max_text_length must be positive")

    tok = ByteTokenizer()
    batch_size = len(samples)

    inputs = np.full((batch_size, max_text_length), tok.PAD, dtype=np.int32)
    targets = np.full_like(inputs, tok.PAD)
    loss_mask = np.zeros_like(inputs, dtype=np.bool_)

    images = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    has_image = np.zeros(batch_size, dtype=np.bool_)

    for i, sample in enumerate(samples):
        visual = sample.image is not None

        prompt_bytes = list(sample.prompt.encode("utf-8"))
        answer_bytes = list(sample.answer.encode("utf-8"))

        # Complete sequence before next-token shifting
        sequence = (
            [tok.BOS]
            + ([tok.IMAGE] if visual else [])
            + prompt_bytes
            + answer_bytes
            + [tok.EOS]
        )

        length = len(sequence) - 1

        if length > max_text_length:
            raise ValueError(f"Sample {i} exceeds max_text_length")

        # Teacher-forcing inputs and next-token targets
        inputs[i, :length] = sequence[:-1]
        targets[i, :length] = sequence[1:]

        # Supervise answer bytes and the final EOS token
        answer_start = len(prompt_bytes) + int(visual)
        loss_mask[i, answer_start:length] = True

        if visual:
            if sample.image.size != (64, 64):  # type: ignore
                raise ValueError("Expected 64x64 image")

            image = sample.image.convert("RGB")  # type: ignore

            images[i] = np.asarray(image, dtype=np.float32) / 255.0
            has_image[i] = True

    return TrainingBatch(
        input_ids=jnp.asarray(inputs),
        target_ids=jnp.asarray(targets),
        loss_mask=jnp.asarray(loss_mask),
        images=jnp.asarray(images),
        has_image=jnp.asarray(has_image),
    )
