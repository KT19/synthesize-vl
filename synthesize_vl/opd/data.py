from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from synthesize_vl.data.schema import StreamCursor
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.dataset import generate_sft_examples
from synthesize_vl.sft.schema import SFTExample
from synthesize_vl.tokenizer import ByteTokenizer


@struct.dataclass
class OPDPrompts:
    input_ids: jax.Array
    prefix_lengths: jax.Array
    images: jax.Array
    has_image: jax.Array


@struct.dataclass
class OPDRollouts:
    input_ids: jax.Array
    prefix_lengths: jax.Array
    images: jax.Array
    has_image: jax.Array
    sampled_tokens: jax.Array
    valid_mask: jax.Array


def pack_opd_prompts(examples: Sequence[SFTExample], *, context_length: int, rollout_tokens: int) -> OPDPrompts:
    if not examples:
        raise ValueError("Empty OPD example list")

    input_rows: list[np.ndarray] = []
    lengths: list[int] = []
    images: list[np.ndarray] = []
    flags: list[bool] = []

    for example in examples:
        visual = example.image is not None

        prefix = [ByteTokenizer.BOS] + ([ByteTokenizer.IMAGE] if visual else []) + list(example.prompt.encode("utf-8"))

        # Reserve space for every sampled continuation token
        if len(prefix) + rollout_tokens > context_length:
            raise ValueError(f"{example.sample_id} prompt + rollout exceeds context length {context_length}")

        row = np.full((context_length,), ByteTokenizer.PAD, dtype=np.int32)
        row[: len(prefix)] = prefix

        if visual:
            pixels = np.asarray(example.image.convert("RGB"), dtype=np.float32) / 255.0  # type: ignore
        else:
            pixels = np.zeros((64, 64, 3), dtype=np.float32)

        if pixels.shape != (64, 64, 3):
            raise ValueError("Invalid image shape")

        input_rows.append(row)
        lengths.append(len(prefix))
        images.append(pixels)
        flags.append(visual)

    return OPDPrompts(
        input_ids=jnp.asarray(np.stack(input_rows), dtype=jnp.int32),
        prefix_lengths=jnp.asarray(lengths, dtype=jnp.int32),
        images=jnp.asarray(np.stack(images), dtype=jnp.float32),
        has_image=jnp.asarray(flags, dtype=jnp.bool_),
    )


def make_opd_batch(
    config: SFTDataConfig, cursor: StreamCursor, *, context_length: int, rollout_tokens: int
) -> OPDPrompts:

    examples = generate_sft_examples(config, split="train", start=cursor.next_index, count=config.batch_size)

    return pack_opd_prompts(examples, context_length=context_length, rollout_tokens=rollout_tokens)
