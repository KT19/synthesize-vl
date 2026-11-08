from dataclasses import dataclass
from typing import Literal

import jax
from flax import struct
from PIL import Image

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class PretrainingDocument:
    sample_id: str
    split: Split
    family: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class TrainingSample:
    task: str
    prompt: str
    answer: str
    image: Image.Image | None = None


# Jax PyTree
@struct.dataclass
class TrainingBatch:
    input_ids: jax.Array
    target_ids: jax.Array
    loss_mask: jax.Array
    images: jax.Array
    has_image: jax.Array


@dataclass(frozen=True, slots=True)
class StreamCursor:
    """Position in a deterministic training stream."""

    next_index: int = 0
    updates: int = 0
    documents_seen: int = 0
    bytes_seen: int = 0
    targets_seen: int = 0
