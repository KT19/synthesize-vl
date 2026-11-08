import re
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from orbax.checkpoint import checkpoint_utils

_STEP_PATTERN = re.compile(r"step_(\d+)")


def resolve_pretrained_checkpoint(root: str | Path) -> Path:
    """Accept an exact step directory or a parent directory"""
    root = Path(root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)

    if _STEP_PATTERN.fullmatch(root.name):
        return root

    candidates: list[tuple[int, Path]] = []

    for child in root.iterdir():
        if not child.is_dir():
            continue

        match = _STEP_PATTERN.fullmatch(child.name)

        if match is not None:
            candidates.append((int(match.group(1)), child))

    if not candidates:
        raise FileNotFoundError(f"No step_XXXX checkpoints under {root}")

    return max(candidates, key=lambda pair: pair[0])[1]


def _parameter_signature(params) -> dict[str, tuple]:
    """Compare parameter paths, shapes, and dtypes"""
    result = {}

    leaves, _ = jax.tree_util.tree_flatten_with_path(params)

    for path, array in leaves:
        key = jax.tree_util.keystr(path)

        result[key] = (tuple(array.shape), str(array.dtype))

    return result


def restore_pretrained_parameters(checkpoint: str | Path, expected_params):
    """Restore only model parameters"""
    checkpoint = resolve_pretrained_checkpoint(checkpoint)

    target = {"learner": {"params": expected_params}}
    restore_args = checkpoint_utils.construct_restore_args(target)

    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            checkpoint, args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args, partial_restore=True)
        )

    params = restored["learner"]["params"]

    if _parameter_signature(params) != _parameter_signature(expected_params):
        raise ValueError("Pretrained parameters do not match")

    return params


def initialize_sft_from_pretraining(state: TrainState, checkpoint: str | Path) -> TrainState:
    if int(jax.device_get(state.step)) != 0:
        raise ValueError("Expected a newly initialized SFT optimizer")

    params = restore_pretrained_parameters(checkpoint, state.params)

    return state.replace(params=params)
