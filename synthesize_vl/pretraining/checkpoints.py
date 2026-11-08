import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import StreamCursor
from synthesize_vl.training.checkpoints import checkpoint_payload


def configuration_digest(configuration: Mapping[str, Any]) -> np.ndarray:
    """Stable SHA-256 digest of a JSON-compatible run config."""
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

    return np.frombuffer(hashlib.sha256(encoded).digest(), dtype=np.uint8).copy()


def _snapshot(
    state: TrainState, cursor: StreamCursor, digest: np.ndarray
) -> dict[str, Any]:
    return {
        "learner": checkpoint_payload(state),
        "cursor": asdict(cursor),
        "config_digest": digest,
    }


def save_pretraining_checkpoint(
    root: str | Path,
    state: TrainState,
    cursor: StreamCursor,
    configuration: Mapping[str, Any],
    *,
    checkpoint_step: int | None = None,
) -> Path:
    """Save state, optionally using a separate step for directory naming."""
    step = int(jax.device_get(state.step))

    if step != cursor.updates:
        raise ValueError("Optimizer step and stream cursor disagree")

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    name_step = step if checkpoint_step is None else checkpoint_step
    destination = root_path / f"step_{name_step:08d}"

    if destination.exists():
        raise FileExistsError(destination)

    payload = _snapshot(state, cursor, configuration_digest(configuration))

    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(destination, payload)
        checkpointer.wait_until_finished()

    return destination


def restore_pretraining_checkpoint(
    path: str | Path, template: TrainState, configuration: Mapping[str, Any]
) -> tuple[TrainState, StreamCursor]:
    source = Path(path).resolve()

    if not source.is_dir():
        raise FileNotFoundError(source)

    expected = configuration_digest(configuration)
    target = _snapshot(template, StreamCursor(), expected)

    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(source, target)

    actual = np.asarray(restored["config_digest"], dtype=np.uint8)

    if not np.array_equal(actual, expected):
        raise ValueError("Checkpoint configuration mismatch.")

    learner = restored["learner"]

    state = template.replace(
        step=learner["step"], params=learner["params"], opt_state=learner["opt_state"]
    )

    cursor_values = {key: int(value) for key, value in restored["cursor"].items()}

    cursor = StreamCursor(**cursor_values)

    if int(jax.device_get(state.step)) != cursor.updates:
        raise ValueError("Restored step and cursor disagree")

    if cursor.next_index != cursor.documents_seen:
        raise ValueError("Invalid restored sample cursor")

    return state, cursor
